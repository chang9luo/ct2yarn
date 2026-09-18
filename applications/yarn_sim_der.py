"""
Discrete Elastic Rod (Bergou et al., SIGGRAPH 2008) + Linearly-Implicit Euler.

A re-implementation of an earlier position-based prototype on a proper
elastic-rod model. Key differences from that PBD solver:

  PBD prototype                        DER + implicit Euler (this file)
  -----------------                    --------------------------------
  Constraints projected on positions   Forces from energy gradients
  Stiffness ~ # iterations             Stiffness = real material constant k
  v re-derived from (p-x)/dt           v from M·v_{n+1} = M·v_n + dt·F
  No proper mass / inertia             Mass matrix M; momentum conserved
  Collisions: distance projection      Collisions: penalty forces in F

State per node: position x (N,3), velocity v (N,3).
State per edge: rest length L_rest, rest curvature binormal κb_rest (interior).

Energies:
    E_stretch = Σ ½ k_s (||e_j|| - L_j_rest)²  / L_j_rest
    E_bend    = Σ ½ α/ℓ_i · ||κb_i - κb_i_rest||²
        with  κb_i = 2 (e_{i-1} × e_i) / (||e_{i-1}||·||e_i|| + e_{i-1}·e_i)
        and   ℓ_i  = ½(||e_{i-1}|| + ||e_i||)  (Voronoi length at node i)

Forces (= -∇E):
    Stretch:   analytic, vectorised closed form.
    Bending:   vectorised per-triple finite-difference gradient of the local
               energy at each interior node (3 nodes × 3 dims = 9 evals per
               triple, with M=N-2 triples; each eval is one vectorised
               kernel pass, so the cost is still O(N)).

Integration (linearly-implicit Euler / Baraff-Witkin):
        (M - dt²·J)·v_{n+1} = M·v_n + dt·F(x_n)
        x_{n+1} = x_n + dt·v_{n+1}
    where  J = ∂F/∂x  is approximated matrix-free via
        J·δv ≈ (F(x_n + ε·δv) - F(x_n)) / ε
    so we never assemble it. Linear system solved with scipy CG.

Collisions:
    * Ground / obstacles / bowl walls: penalty forces  F_c = -k_c · pen · n
      computed inside F(x), so they enter the implicit solve correctly.
    * Self-collision: still cKDTree pairs, but treated as PBD-style position
      projection AFTER the implicit step (hybrid; full IPC is out of scope).

Trajectory output format: yarn_sim_traj_v1.npz.

Usage:
    python yarn_sim_der.py yarn_likely_20260527_052928.npz --mode drop \\
        --frames 120 --preview
"""
from __future__ import annotations
import argparse
import time

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse.linalg import cg, LinearOperator
from scipy.ndimage import map_coordinates

import yarn_plies as Y          # load_centerlines (no GUI imported)


# ---------------------------------------------------------------------------
# Preview helpers, inlined from the earlier PBD prototype so this file stands
# alone. They only draw the polyscope preview and are not part of the physics.
# ---------------------------------------------------------------------------
def _hemisphere(cx, cy, cz, R, z_rim, n_theta=24, n_phi=40):
    """Build a triangle mesh of the bowl-shaped lower cap (theta=0 at the
    south pole; goes up to where the sphere meets z = z_rim)."""
    import numpy as np
    cos_max = float(np.clip((cz - z_rim) / R, -1.0, 1.0))
    theta_max = float(np.arccos(cos_max))           # north angle (from -z)
    th = np.linspace(0.0, theta_max, n_theta)
    ph = np.linspace(0.0, 2 * np.pi, n_phi, endpoint=False)
    T, P = np.meshgrid(th, ph, indexing='ij')
    vx = cx + R * np.sin(T) * np.cos(P)
    vy = cy + R * np.sin(T) * np.sin(P)
    vz = cz - R * np.cos(T)                          # theta=0 -> south pole
    verts = np.stack([vx, vy, vz], -1).reshape(-1, 3)
    i = np.arange(n_theta - 1)[:, None]
    j = np.arange(n_phi)[None, :]
    jn = (j + 1) % n_phi
    v00 = i * n_phi + j; v01 = i * n_phi + jn
    v10 = (i + 1) * n_phi + j; v11 = (i + 1) * n_phi + jn
    f1 = np.stack([v00, v10, v11], -1).reshape(-1, 3)
    f2 = np.stack([v00, v11, v01], -1).reshape(-1, 3)
    return verts, np.vstack([f1, f2]).astype(np.int32)

def _upsample_chain(frames, edges, factor=3):
    """Viz-only smoothing: cubic-spline-resample a single chain to (factor-1)
    extra nodes between every consecutive pair. Removes corners at original
    nodes that linear interpolation would leave.

    `frames` can be (N,3) or (F,N,3); returns (new_frames, new_edges).
    Multi-strand / non-sequential edges fall back to no upsample.
    """
    if factor <= 1:
        return frames, edges
    squeeze = (frames.ndim == 2)
    if squeeze:
        frames = frames[None]
    F, N, D = frames.shape
    sequential = (len(edges) == N - 1
                  and np.array_equal(edges[:, 0], np.arange(N - 1))
                  and np.array_equal(edges[:, 1], np.arange(1, N)))
    if not sequential:
        return (frames[0] if squeeze else frames), edges
    new_N = (N - 1) * factor + 1
    new_idx = np.linspace(0.0, N - 1, new_N)
    new_frames = np.empty((F, new_N, D), dtype=frames.dtype)
    try:
        from scipy.interpolate import CubicSpline
        for f in range(F):
            new_frames[f] = CubicSpline(np.arange(N), frames[f],
                                        axis=0)(new_idx)
    except ImportError:
        lo = new_idx.astype(np.int64).clip(0, N - 2)
        a = (new_idx - lo).astype(frames.dtype)[None, :, None]
        new_frames[:] = (1 - a) * frames[:, lo] + a * frames[:, lo + 1]
    new_edges = np.column_stack([np.arange(new_N - 1), np.arange(1, new_N)])
    if squeeze:
        new_frames = new_frames[0]
    return new_frames, new_edges

def _tube_mesh(poly, radius, nseg=14):
    """Triangulated tube surface around a polyline -- the *rendered* yarn.
    Reuses yarn_plies' RMF-based mesh builder so the cross-section doesn't
    twist. Returns (verts(M*nseg,3), faces). Replaces polyscope's segmented
    curve_network display with one continuous smooth-shaded surface."""
    if len(poly) < 2:
        return (np.zeros((0, 3), np.float32),
                np.zeros((0, 3), np.int32))
    v, f = Y.tube_mesh(np.asarray(poly, np.float64), float(radius), int(nseg))
    return v.astype(np.float32), f.astype(np.int32)

def _show_obstacles(ps, obstacles):
    """Draw each cylindrical bar as a thick curve segment along its y-axis."""
    import numpy as np
    for i, (cx, cy, cz, R, L) in enumerate(obstacles):
        v = np.array([[cx, cy - L * 0.5, cz], [cx, cy + L * 0.5, cz]])
        e = np.array([[0, 1]])
        cn = ps.register_curve_network(f'obs_{i}', v, e)
        cn.set_radius(float(R), relative=False)
        cn.set_color((0.55, 0.55, 0.62))

def _show_bowls(ps, bowls):
    import numpy as np
    for i, (cx, cy, cz, R, z_rim) in enumerate(bowls):
        v, f = _hemisphere(cx, cy, cz, R, z_rim)
        m = ps.register_surface_mesh(f'bowl_{i}', v, f, color=(0.7, 0.65, 0.55),
                                     smooth_shade=True)
        m.set_transparency(0.55)

def _show_ground(ps, x_ref, ground_z, name='ground'):
    """Register a translucent ground rectangle sized to the geometry."""
    import numpy as np
    ctr = x_ref.mean(0)
    ext = float((x_ref.max(0) - x_ref.min(0)).max()) * 1.3
    v = np.array([[ctr[0] - ext, ctr[1] - ext, ground_z],
                  [ctr[0] + ext, ctr[1] - ext, ground_z],
                  [ctr[0] + ext, ctr[1] + ext, ground_z],
                  [ctr[0] - ext, ctr[1] + ext, ground_z]])
    f = np.array([[0, 1, 2], [0, 2, 3]])
    m = ps.register_surface_mesh(name, v, f, color=(0.75, 0.75, 0.78),
                                 smooth_shade=True)
    m.set_transparency(0.45)
    return m



# --------------------------------------------------------------------------
# Resampling + rod setup
# --------------------------------------------------------------------------
def resample(P, ds):
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    al = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(al[-1])
    k = max(int(round(total / max(ds, 1e-6))) + 1, 2)
    targ = np.linspace(0.0, total, k)
    return np.column_stack([np.interp(targ, al, P[:, i]) for i in range(3)])


def build_rod(curves, ds):
    """Concatenate resampled strands. Returns (x0, edges, strand, tri,
    rest_lens, rest_kb, rest_vor) where `tri` is the (M,3) array of
    (i-1, i, i+1) node-index triples used for bending."""
    xs, edges_list, strand_list = [], [], []
    base = 0
    for sid, P in enumerate(curves):
        Q = resample(P, ds)
        n = len(Q)
        xs.append(Q)
        edges_list.append(np.column_stack([np.arange(n - 1), np.arange(1, n)])
                          + base)
        strand_list.append(np.full(n, sid))
        base += n
    x0 = np.vstack(xs)
    edges = np.vstack(edges_list)
    strand = np.concatenate(strand_list)
    N = len(x0)

    # rest edge lengths
    rest_lens = np.linalg.norm(x0[edges[:, 1]] - x0[edges[:, 0]], axis=1)

    # bending triples (i-1, i, i+1) within a strand
    same = (strand[:-2] == strand[1:-1]) & (strand[1:-1] == strand[2:])
    tri_mid = np.where(same)[0] + 1                 # middle-node indices
    tri = np.column_stack([tri_mid - 1, tri_mid, tri_mid + 1])

    # rest curvature binormal at each interior node
    e_prev = x0[tri[:, 1]] - x0[tri[:, 0]]
    e_next = x0[tri[:, 2]] - x0[tri[:, 1]]
    cross = np.cross(e_prev, e_next)
    nL = np.linalg.norm(e_prev, axis=1)
    nR = np.linalg.norm(e_next, axis=1)
    denom = nL * nR + (e_prev * e_next).sum(1) + 1e-12
    rest_kb = 2.0 * cross / denom[:, None]
    rest_vor = 0.5 * (nL + nR)

    return x0, edges, strand, tri, rest_lens, rest_kb, rest_vor


# --------------------------------------------------------------------------
# Bending energy per interior triple (vectorised)
# --------------------------------------------------------------------------
def bending_energy_per_triple(x, tri, rest_kb, rest_vor, alpha):
    """Vector of E_b for each interior triple. (M,)."""
    e_prev = x[tri[:, 1]] - x[tri[:, 0]]
    e_next = x[tri[:, 2]] - x[tri[:, 1]]
    cross = np.cross(e_prev, e_next)
    nL = np.linalg.norm(e_prev, axis=1) + 1e-12
    nR = np.linalg.norm(e_next, axis=1) + 1e-12
    denom = nL * nR + (e_prev * e_next).sum(1) + 1e-12
    kb = 2.0 * cross / denom[:, None]
    diff = kb - rest_kb
    return 0.5 * alpha * (diff * diff).sum(1) / np.maximum(rest_vor, 1e-9)


def bending_forces_fd(x, tri, rest_kb, rest_vor, alpha, h=1e-4):
    """Local FD gradient of the bending energy per triple. Each of the
    3·3=9 (node, axis) perturbations is one vectorised kernel pass, so the
    whole bending-force computation is 9 × O(M) ≈ O(N)."""
    M = len(tri)
    if M == 0:
        return np.zeros_like(x)
    E0 = bending_energy_per_triple(x, tri, rest_kb, rest_vor, alpha)
    F = np.zeros_like(x)
    # Loop over which node of the triple (0,1,2) and which axis (0,1,2):
    for k in range(3):                              # 0 = prev, 1 = mid, 2 = nxt
        node_idx = tri[:, k]
        for d in range(3):
            xh = x.copy()
            xh[node_idx, d] += h
            Eh = bending_energy_per_triple(xh, tri, rest_kb, rest_vor, alpha)
            grad = (Eh - E0) / h                    # ∂E/∂x_{tri[:,k], d}
            np.add.at(F[:, d], node_idx, -grad)     # F = -∇E
    return F


# --------------------------------------------------------------------------
# Full force F(x) -- gravity + stretch + bending + contact penalties
# --------------------------------------------------------------------------
def compute_forces(x, args, edges, tri, rest_lens, rest_kb, rest_vor,
                   ground, obstacles, bowls, mass):
    """All conservative forces evaluated at x. Returns (N,3)."""
    N = len(x)
    F = np.zeros_like(x)

    # gravity (force = m·g)
    F[:, 2] += mass[:, 0] * args.gravity

    # stretch:  F = -k_s (L - L_rest) t  on each endpoint
    ia, ib = edges[:, 0], edges[:, 1]
    e = x[ib] - x[ia]
    L = np.linalg.norm(e, axis=1) + 1e-12
    t = e / L[:, None]
    mag = args.k_stretch * (L - rest_lens) / rest_lens   # divide by rest for unit-consistent
    Fa = mag[:, None] * t                                 # pulls a toward b when stretched
    np.add.at(F, ia, Fa)
    np.add.at(F, ib, -Fa)

    # bending (FD per triple)
    if args.k_bend > 0:
        F += bending_forces_fd(x, tri, rest_kb, rest_vor, args.k_bend)

    # contact penalties: ground (z >= ground + r)
    r = args.radius
    pen_g = (ground + r) - x[:, 2]
    in_g = pen_g > 0.0
    F[in_g, 2] += args.k_contact * pen_g[in_g]

    # cylinder obstacles (axis along y; xz-radial penalty)
    for cx, cy, cz, R_o, L_o in obstacles:
        dx = x[:, 0] - cx
        dz = x[:, 2] - cz
        d2 = dx * dx + dz * dz
        in_y = np.abs(x[:, 1] - cy) < L_o * 0.5
        tgt = R_o + r
        pen = tgt - np.sqrt(d2 + 1e-12)
        mask = (pen > 0.0) & in_y
        if mask.any():
            idx = np.where(mask)[0]
            d = np.sqrt(d2[idx] + 1e-12)
            nx = dx[idx] / d
            nz = dz[idx] / d
            F[idx, 0] += args.k_contact * pen[idx] * nx
            F[idx, 2] += args.k_contact * pen[idx] * nz

    # bowls (push inward when below rim and outside the inner-radius R-r)
    for cx, cy, cz, R_b, z_rim in bowls:
        d = x - np.array([cx, cy, cz])
        dist = np.linalg.norm(d, axis=1) + 1e-12
        tgt = R_b - r
        below = x[:, 2] < z_rim
        pen = dist - tgt
        mask = (pen > 0.0) & below
        if mask.any():
            idx = np.where(mask)[0]
            n_hat = d[idx] / dist[idx, None]
            F[idx] -= args.k_contact * pen[idx, None] * n_hat   # push inward

    return F


# --------------------------------------------------------------------------
# Linearly-implicit Euler step (Baraff-Witkin)
# --------------------------------------------------------------------------
def implicit_step(x, v, mass, dt, args, edges, tri, rest_lens, rest_kb,
                  rest_vor, ground, obstacles, bowls, pin_mask):
    """One time step. Returns (x_new, v_new). Solves
        (M - dt²·J)·v_{n+1} = M·v_n + dt·F(x_n)
    matrix-free, where J·δv ≈ (F(x_n + ε·δv·dt) - F(x_n))/(ε·dt)."""
    N = len(x)
    F0 = compute_forces(x, args, edges, tri, rest_lens, rest_kb, rest_vor,
                        ground, obstacles, bowls, mass)
    # zero forces on pinned nodes so they stay put
    F0[pin_mask] = 0.0

    rhs = (mass * v + dt * F0).reshape(-1)
    eps = max(1e-4, 1e-3 * dt)

    def A_matvec(dv_flat):
        dv = dv_flat.reshape(N, 3)
        dv = dv.copy()
        dv[pin_mask] = 0.0                          # pin velocities to 0
        F_pert = compute_forces(x + (eps * dt) * dv, args, edges, tri,
                                rest_lens, rest_kb, rest_vor, ground,
                                obstacles, bowls, mass)
        F_pert[pin_mask] = 0.0
        J_dv = (F_pert - F0) / (eps * dt)           # finite-diff Jacobian
        out = mass * dv - (dt * dt) * J_dv
        out[pin_mask] = 0.0
        return out.reshape(-1)

    A = LinearOperator((3 * N, 3 * N), matvec=A_matvec, dtype=np.float64)
    v_new_flat, _ = cg(A, rhs, x0=v.reshape(-1), rtol=1e-3,
                       maxiter=args.cg_iters)
    v_new = v_new_flat.reshape(N, 3)
    v_new[pin_mask] = 0.0
    v_new *= (1.0 - args.damping)                   # velocity damping
    x_new = x + dt * v_new
    return x_new, v_new


# --------------------------------------------------------------------------
# Post-step self-collision projection (PBD-style; hybrid)
# --------------------------------------------------------------------------
def project_self_collision(x, v, r, strand_local_skip, dt):
    """Push apart pairs of nodes closer than 2*r, skipping same-strand near
    neighbours. Position correction; velocity recomputed from delta x."""
    strand, local, skip = strand_local_skip
    pr = cKDTree(x).query_pairs(2.0 * r, output_type='ndarray')
    if len(pr) == 0:
        return x, v
    si, sj = pr[:, 0], pr[:, 1]
    keep = ~((strand[si] == strand[sj]) & (np.abs(local[si] - local[sj]) <= skip))
    pr = pr[keep]
    if len(pr) == 0:
        return x, v
    x_new = x.copy()
    for _ in range(2):                              # 2 Gauss-Seidel sweeps
        d = x_new[pr[:, 1]] - x_new[pr[:, 0]]
        dist = np.linalg.norm(d, axis=1) + 1e-12
        bad = dist < 2.0 * r
        if not bad.any():
            break
        idx = np.where(bad)[0]
        n_hat = d[idx] / dist[idx, None]
        push = 0.5 * (2.0 * r - dist[idx])[:, None] * n_hat
        np.add.at(x_new, pr[idx, 0], -push)
        np.add.at(x_new, pr[idx, 1], +push)
    v_new = v + (x_new - x) / dt
    return x_new, v_new


# --------------------------------------------------------------------------
# Scene primitive helpers
# --------------------------------------------------------------------------
def auto_obstacles(x0, lift, ground, n, R_o, off):
    if n <= 0:
        return []
    bb_min, bb_max = x0.min(0), x0.max(0)
    ctr = 0.5 * (bb_min + bb_max)
    z_top = bb_min[2] + lift - 120.0
    z_bot = ground + 100.0
    if z_top <= z_bot + 50:
        return []
    z_levels = np.linspace(z_top, z_bot, n)
    L = float(bb_max[1] - bb_min[1]) * 1.5
    out = []
    for i, cz in enumerate(z_levels):
        t = i / max(n - 1, 1)
        cx = float(ctr[0]) + float(off) * (2.0 * t - 1.0)
        out.append([cx, float(ctr[1]), float(cz), float(R_o), L])
    return out


def auto_bowl(x0, lift, ground, R=None):
    bb_min, bb_max = x0.min(0), x0.max(0)
    ctr = 0.5 * (bb_min + bb_max)
    rim_max = float(bb_min[2] + lift) - 100.0
    avail = rim_max - float(ground)
    if R is None:
        R = min(float(np.max(bb_max[:2] - bb_min[:2])) * 0.6, avail * 0.95)
    R = float(max(R, 50.0))
    cz = float(ground) + R
    return [float(ctr[0]), float(ctr[1]), cz, R, cz]      # rim = equator


# --------------------------------------------------------------------------
# Mesh bowl collision via a baked signed-distance field
# --------------------------------------------------------------------------
class BowlSDF:
    """Penalty/projection collision against a glass-bowl SDF.

    sdf < 0 inside the glass solid, > 0 in air/cavity.  A node is kept at
    least `radius` away from the solid (sdf >= radius), pushed out along the
    SDF gradient (which points into the cavity from the inner wall)."""
    def __init__(self, path):
        d = np.load(path)
        self.sdf = d['sdf'].astype(np.float32)
        self.origin = d['origin'].astype(np.float64)
        self.sp = float(d['spacing'])
        gx, gy, gz = np.gradient(self.sdf, self.sp)
        self.g = (gx, gy, gz)
        self.n = np.array(self.sdf.shape)

    def _samp(self, grid, idx):
        return map_coordinates(grid, idx, order=1, mode='nearest')

    def collide(self, x, v, radius):
        idx = ((x - self.origin) / self.sp).T            # (3, N) index coords
        inb = np.all((idx >= 0) & (idx <= (self.n - 1)[:, None]), axis=0)
        if not inb.any():
            return x, v
        s = self._samp(self.sdf, idx)
        hit = inb & (s < radius)
        if not hit.any():
            return x, v
        g = np.stack([self._samp(self.g[0], idx),
                      self._samp(self.g[1], idx),
                      self._samp(self.g[2], idx)], axis=1)
        nrm = g / (np.linalg.norm(g, axis=1, keepdims=True) + 1e-9)
        x = x.copy(); v = v.copy()
        x[hit] += ((radius - s)[:, None] * nrm)[hit]
        vn = (v * nrm).sum(1)                              # remove inward velocity
        inward = hit & (vn < 0)
        v[inward] -= vn[inward, None] * nrm[inward]
        return x, v


# --------------------------------------------------------------------------
# Top-level simulate
# --------------------------------------------------------------------------
def simulate(curves, args, frame_cb=None):
    x0, edges, strand, tri, rest_lens, rest_kb, rest_vor = build_rod(
        curves, args.ds)
    N = len(x0)
    local = np.arange(N) - np.cumsum(np.r_[0, np.bincount(strand)[:-1]])[strand]

    x = x0.copy()
    v = np.zeros_like(x)
    # uniform mass per node (kg-equivalent; just sets the inertia scale)
    mass = np.ones((N, 1)) * args.mass

    zmin = float(x0[:, 2].min())
    ground = (float(args.ground) if args.ground is not None else zmin)

    if args.mode in ('drop', 'bowl'):
        x[:, 2] += args.lift
    # Mesh bowl (baked SDF) takes over collision; disable analytic primitives.
    bowl_sdf = BowlSDF(args.bowl_sdf) if getattr(args, 'bowl_sdf', '') else None
    if bowl_sdf is not None:
        obstacles = []; bowls = []
        print(f'[yarn_sim_der] mesh bowl SDF: {args.bowl_sdf} '
              f'(grid {tuple(bowl_sdf.n)}, spacing {bowl_sdf.sp:.0f})')
    else:
        obstacles = (auto_obstacles(x0, args.lift, ground, args.obstacles,
                                    args.obs_radius, args.obs_offset)
                     if args.mode == 'drop' else [])
        bowls = ([auto_bowl(x0, args.lift, ground, args.bowl_radius)]
                 if args.mode == 'bowl' else [])

    # optional end-pin
    pin_mask = np.zeros(N, dtype=bool)
    if args.mode == 'drop' and args.pin > 0:
        idx0 = np.where(strand == 0)[0][:args.pin]
        pin_mask[idx0] = True

    dt_sub = args.dt / args.substeps

    frames = [x.astype(np.float32).copy()]
    t_pen = []

    t0 = time.time()
    for f in range(args.frames):
        for _ in range(args.substeps):
            x, v = implicit_step(x, v, mass, dt_sub, args, edges, tri,
                                 rest_lens, rest_kb, rest_vor, ground,
                                 obstacles, bowls, pin_mask)
            # post-step self-collision projection (hybrid)
            x, v = project_self_collision(
                x, v, args.radius, (strand, local, args.skip), dt_sub)
            # mesh bowl collision (keeps yarn inside the glass cavity)
            if bowl_sdf is not None:
                x, v = bowl_sdf.collide(x, v, args.radius)
            # ground clamp safety net (in case penalty wasn't stiff enough)
            below = x[:, 2] < ground + args.radius
            if below.any():
                x[below, 2] = ground + args.radius
                v[below, 2] = np.maximum(v[below, 2], 0.0)
        frames.append(x.astype(np.float32).copy())
        # residual penetration (informational)
        pr = cKDTree(x).query_pairs(2.0 * args.radius, output_type='ndarray')
        if len(pr):
            si, sj = pr[:, 0], pr[:, 1]
            keep = ~((strand[si] == strand[sj])
                     & (np.abs(local[si] - local[sj]) <= args.skip))
            pr = pr[keep]
            d = np.linalg.norm(x[pr[:, 1]] - x[pr[:, 0]], axis=1) if len(pr) \
                else np.array([2 * args.radius])
            t_pen.append(float(np.clip(2 * args.radius - d.min(), 0, None)))
        else:
            t_pen.append(0.0)
        if frame_cb is not None:
            frame_cb(f, x.astype(np.float64))

    dt_total = time.time() - t0
    print(f'[yarn_sim_der] {args.frames} frames, {N} nodes in '
          f'{dt_total:.1f}s ({dt_total / args.frames * 1000:.0f} ms/frame)')

    return (np.stack(frames, axis=0), edges, strand,
            np.zeros(N),                                # node_s (unused)
            float(ground), np.array(t_pen),
            np.asarray(obstacles, np.float32) if obstacles
            else np.zeros((0, 5), np.float32),
            np.asarray(bowls, np.float32) if bowls
            else np.zeros((0, 5), np.float32),
            np.zeros((args.frames + 1, len(edges)), np.float32))   # phi placeholder


# --------------------------------------------------------------------------
# Polyscope viz: reuse the helpers from yarn_sim (cubic-spline upsample +
# tube surface mesh + ground / obstacle / bowl primitives).
# --------------------------------------------------------------------------
def live(curves, args):
    """Open a polyscope window and update the tube mesh every solved frame
    (frame_tick, not the blocking show loop). Mirrors yarn_sim.live() but
    drives the DER solver instead."""
    import polyscope as ps
    # Build the rod ONCE so we know the edges / strand / bbox for viz.
    x0, edges, strand, *_ = build_rod(curves, args.ds)
    zmin = float(x0[:, 2].min())
    gz = float(args.ground) if args.ground is not None else zmin
    obs_list = (auto_obstacles(x0, args.lift, gz, args.obstacles,
                               args.obs_radius, args.obs_offset)
                if args.mode == 'drop' else [])
    bowl_list = ([auto_bowl(x0, args.lift, gz, args.bowl_radius)]
                 if args.mode == 'bowl' else [])
    ps.init(); ps.set_up_dir('z_up')
    ps.set_transparency_mode('pretty')
    x_show = x0.copy()
    if args.mode in ('drop', 'bowl'):
        x_show[:, 2] += args.lift
    x_show_v, _ = _upsample_chain(x_show, edges, args.viz_upsample)
    tube_r = args.radius * 0.6
    v0, faces = _tube_mesh(x_show_v, tube_r, 14)
    ps.register_surface_mesh('yarn', v0, faces, smooth_shade=True,
                             color=(0.85, 0.65, 0.45))
    _show_ground(ps, x_show, gz)
    if obs_list:
        _show_obstacles(ps, obs_list)
    if bowl_list:
        _show_bowls(ps, bowl_list)
    ps.frame_tick()

    def cb(f_idx, x):
        x_v, _ = _upsample_chain(x, edges, args.viz_upsample)
        verts, _ = _tube_mesh(x_v, tube_r, 14)
        ps.get_surface_mesh('yarn').update_vertex_positions(verts)
        ps.frame_tick()

    out = simulate(curves, args, frame_cb=cb)
    print('[yarn_sim_der] sim done -- close the window to exit.')
    ps.show()
    return out


def preview(frames, edges, strand, ground, radius, obstacles=(), bowls=(),
            viz_upsample=3, strand_colors=None):
    """Multi-strand DER playback — one tube mesh per strand so the
    polyline-tube generator doesn't join the tail of strand N to the
    head of strand N+1.  `strand_colors`: optional (S, 3) RGB per strand
    (defaults to PG2026 palette)."""
    import polyscope as ps
    import polyscope.imgui as psim

    _PG = np.array([
        [0.00, 0.66, 0.46],   # #00a974 teal
        [0.00, 0.78, 0.93],   # #01c7ee cyan
        [0.98, 0.87, 0.04],   # #fbdd0b yellow
        [0.91, 0.33, 0.32],   # #e75352 red
        [0.85, 0.40, 0.85],
        [0.30, 0.55, 0.95],
    ], np.float32)

    frames, edges = _upsample_chain(frames, edges, viz_upsample)
    # After upsample, edges array indexes into the upsampled node space.
    # Reconstruct the per-node strand id from the (still increasing-by-edge)
    # structure: each connected chain in `edges` belongs to one strand.
    N = frames.shape[1]
    # Greedy chain-walk: nodes that share an edge get the same strand id.
    parent = np.arange(N)
    def _find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for u, v in edges:
        u, v = int(u), int(v); ru, rv = _find(u), _find(v)
        if ru != rv: parent[ru] = rv
    roots = np.array([_find(i) for i in range(N)])
    # Re-number 0..S-1 in encounter order so strand 0 = first chain etc.
    uniq, remap_strand = np.unique(roots, return_inverse=True)
    S = int(remap_strand.max()) + 1

    if strand_colors is None or len(strand_colors) < S:
        strand_colors = np.vstack([_PG, _PG])[:S]
    else:
        strand_colors = np.asarray(strand_colors, np.float32)[:S]

    ps.init(); ps.set_up_dir('z_up')
    ps.set_transparency_mode('pretty')
    tube_r = float(radius) * 0.6
    nseg = 14

    # Per-strand node-index lists (stable order along the polyline so the
    # tube mesh sweep produces a smooth, kink-free strand).
    chain_idx = []
    for s in range(S):
        # Order nodes of this strand by walking the edge list.
        nodes = np.where(remap_strand == s)[0]
        # Build adjacency restricted to these nodes.
        nset = set(int(n) for n in nodes)
        nbrs = {n: [] for n in nset}
        for u, v in edges:
            u, v = int(u), int(v)
            if u in nset and v in nset:
                nbrs[u].append(v); nbrs[v].append(u)
        # Find an endpoint (deg 1) to start walking.
        start = next((n for n, ne in nbrs.items() if len(ne) == 1),
                     int(next(iter(nset))))
        order = [start]; seen = {start}; cur = start
        while True:
            nxt = next((n for n in nbrs[cur] if n not in seen), None)
            if nxt is None: break
            order.append(nxt); seen.add(nxt); cur = nxt
        chain_idx.append(np.asarray(order, np.int64))

    # Register one surface mesh per strand at frame 0.
    for s, idx in enumerate(chain_idx):
        poly0 = frames[0][idx]
        v, f = _tube_mesh(poly0, tube_r, nseg)
        ps.register_surface_mesh(
            f'yarn_{s}', v, f,
            smooth_shade=True,
            color=tuple(float(c) for c in strand_colors[s]))

    _show_ground(ps, frames[0], float(ground))
    if len(obstacles): _show_obstacles(ps, obstacles)
    if len(bowls):     _show_bowls(ps, bowls)

    state = {'f': 0, 'play': True, 'speed': 1}
    F = len(frames)

    def cb():
        if state['play']:
            state['f'] = (state['f'] + state['speed']) % F
        ch, v = psim.SliderInt('frame', state['f'], 0, F - 1)
        if ch: state['f'] = v
        ch, v = psim.Checkbox('play', state['play'])
        if ch: state['play'] = v
        psim.SameLine()
        ch, v = psim.SliderInt('speed', state['speed'], 1, 8)
        if ch: state['speed'] = v
        psim.TextUnformatted(f'frame {state["f"]+1} / {F}  '
                             f'({S} strand(s))')
        # Per-strand tube vertex update.
        for s, idx in enumerate(chain_idx):
            poly = frames[state['f']][idx]
            verts, _ = _tube_mesh(poly, tube_r, nseg)
            sm = ps.get_surface_mesh(f'yarn_{s}')
            sm.update_vertex_positions(verts)

    ps.set_user_callback(cb)
    ps.show()


# --------------------------------------------------------------------------
def save_traj(frames, edges, strand, node_s, ground, args, path,
              obstacles=None, bowls=None, phi=None):
    obs = obstacles if obstacles is not None else np.zeros((0, 5), np.float32)
    bw = bowls if bowls is not None else np.zeros((0, 5), np.float32)
    kw = dict(fmt='yarn_sim_traj_v1', frames=frames, edges=edges,
              strand=strand, node_s=node_s, ground=ground,
              radius=args.radius, ds=args.ds, mode=args.mode, dt=args.dt,
              obstacles=obs, bowls=bw, solver='der_implicit_euler')
    if phi is not None:
        kw['phi'] = phi
    np.savez_compressed(path, **kw)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('npz')
    ap.add_argument('--mode', choices=['drop', 'bowl'], default='drop')
    ap.add_argument('--radius', type=float, default=20.0)
    ap.add_argument('--ds', type=float, default=24.0)
    ap.add_argument('--frames', type=int, default=120)
    ap.add_argument('--substeps', type=int, default=4)
    ap.add_argument('--dt', type=float, default=1.0 / 60.0)
    ap.add_argument('--cg_iters', type=int, default=60,
                    help='max conjugate-gradient iterations per step')

    # material
    ap.add_argument('--mass', type=float, default=1.0,
                    help='node mass (sets inertia scale)')
    ap.add_argument('--k_stretch', type=float, default=4.0e5,
                    help='stretch stiffness (high enough to be inextensible, '
                         'low enough that implicit-Euler CG converges)')
    ap.add_argument('--k_bend', type=float, default=3.0e2,
                    help='bending stiffness (DER κb energy coefficient)')
    ap.add_argument('--k_contact', type=float, default=8.0e4,
                    help='penalty stiffness for ground/obstacle contact')

    ap.add_argument('--gravity', type=float, default=-1500.0)
    ap.add_argument('--damping', type=float, default=0.012,
                    help='velocity damping per substep')
    ap.add_argument('--skip', type=int, default=2,
                    help='exclude same-strand neighbours within this index gap')
    ap.add_argument('--ground', type=float, default=None)
    ap.add_argument('--lift', type=float, default=800.0)
    ap.add_argument('--pin', type=int, default=0)

    # scene
    ap.add_argument('--obstacles', type=int, default=3)
    ap.add_argument('--obs_radius', type=float, default=100.0)
    ap.add_argument('--obs_offset', type=float, default=300.0)
    ap.add_argument('--bowl_radius', type=float, default=None)
    ap.add_argument('--bowl_sdf', type=str, default='',
                    help='baked glass-bowl SDF npz (tmp/glassbowl_sdf.npz); '
                         'overrides analytic obstacles/bowl with mesh collision')

    ap.add_argument('--out', type=str, default='')
    ap.add_argument('--no_preview', dest='preview', action='store_false')
    ap.add_argument('--preview', action='store_true', default=True,
                    help='after sim, open a polyscope playback window')
    ap.add_argument('--live', action='store_true',
                    help='show polyscope window and update each frame DURING '
                         'the sim (slower per-frame but watch it solve)')
    ap.add_argument('--preview_traj', type=str, default='',
                    help='replay a saved trajectory npz; no sim')
    ap.add_argument('--viz_upsample', type=int, default=3,
                    help='cubic-spline upsample factor for the rendered tube')
    args = ap.parse_args()

    if args.preview_traj:
        d = np.load(args.preview_traj, allow_pickle=True)
        obs = d['obstacles'] if 'obstacles' in d.files else ()
        bowls = d['bowls'] if 'bowls' in d.files else ()
        sc = (d['strand_colors'] if 'strand_colors' in d.files else None)
        preview(d['frames'], d['edges'], d['strand'],
                float(d['ground']), float(d['radius']),
                obstacles=obs, bowls=bowls,
                viz_upsample=args.viz_upsample,
                strand_colors=sc)
        return

    curves = Y.load_centerlines(args.npz)
    # Optional per-strand colours saved by view_multi_yarn.py — carried
    # straight into the trajectory npz so re-playback paints correctly.
    try:
        _input_d = np.load(args.npz, allow_pickle=False)
        strand_colors_in = (_input_d['strand_colors']
                            if 'strand_colors' in _input_d.files else None)
    except Exception:
        strand_colors_in = None
    print(f'[yarn_sim_der] {len(curves)} strand(s); mode={args.mode} (DER)'
          + ('  [LIVE]' if args.live else ''))

    if args.live:
        out = live(curves, args)
    else:
        out = simulate(curves, args)
    frames, edges, strand, node_s, ground, pen, obs, bowls, phi = out

    print(f'[yarn_sim_der] penetration: peak {pen.max():.2f} vox, '
          f'final {pen[-1]:.2f} vox  (collision diameter {2*args.radius:.0f})')

    path = args.out or f'sim_der_{args.mode}_{time.strftime("%H%M%S")}.npz'
    save_traj(frames, edges, strand, node_s, ground, args, path,
              obstacles=obs, bowls=bowls, phi=phi)
    if strand_colors_in is not None:
        # Append strand_colors to the trajectory npz so playback can
        # restore them without going back to the input file.
        _td = dict(np.load(path, allow_pickle=False))
        _td['strand_colors'] = np.asarray(strand_colors_in, np.float32)
        np.savez_compressed(path, **_td)
    print(f'[yarn_sim_der] wrote {path}')

    if args.preview and not args.live:
        preview(frames, edges, strand, ground, args.radius,
                obstacles=obs, bowls=bowls,
                viz_upsample=args.viz_upsample,
                strand_colors=strand_colors_in)


if __name__ == '__main__':
    main()
