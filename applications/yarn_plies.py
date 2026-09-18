"""
Twisted plies around a fitted yarn centerline.

Reads a yarn-npz centerline (the `yarn_state_v1` format written by
view_yarn / toy_chain_spring, i.e. per-curve `curve_<sid>` arrays, plus a
single `yarn` key for one strand) and wraps N plies helically around each
centerline, the *ply level* of the Schroeder/Zhao coaxial-helix procedural
yarn model that Zhao et al. 2016 ("Fitting Procedural Yarn Models for
Realistic Cloth Rendering") fits to micro-CT scans.

Construction (per centerline C(s), s = arclength):
    * a rotation-minimising frame {N(s), B(s)} (carry the normal forward,
      projecting out each tangent) gives a non-twisting reference, so the
      only spin is the twist we add on purpose;
    * ply p in 0..K-1 has base phase phi_p = 2*pi*p/K and rides a helix
        theta(s)      = sign * 2*pi * s / pitch + phi_p
        ply_p(s)      = C(s) + R * (cos(theta) N(s) + sin(theta) B(s))
      so `pitch` is the arclength of one full 2*pi revolution (the model's
      `yarn_alpha`) and R is the ply offset from the centre (`yarn_radius`).

Both the twist (via `pitch`) and the ply count K are adjustable: as CLI
flags, and as live sliders in the polyscope viewer. An "Export" button (or
--export) writes ply centerlines to yarn_plies_<ts>.npz and the merged tube
mesh to yarn_plies_<ts>.obj.

Usage:
    python yarn_plies.py yarn_likely_20260527_052928.npz
    python yarn_plies.py F.npz --n_plies 3 --pitch 80 --radius 12 --export
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path

import numpy as np


def rainbow_rgb(t, sat=0.9, val=0.95):
    """Vectorised HSV->RGB rainbow (red->violet) for t in [0,1] -> (...,3)."""
    t = np.clip(np.asarray(t, float), 0.0, 1.0)
    h6 = (0.85 * t) * 6.0                       # hue 0(red)..0.85(violet)
    c = val * sat
    x = c * (1 - np.abs((h6 % 2) - 1))
    m = val - c
    z = np.zeros_like(t)
    seg = np.floor(h6).astype(int) % 6
    cond = [seg == k for k in range(6)]
    r = np.select(cond, [c, x, z, z, x, c]) + m
    g = np.select(cond, [x, c, c, x, z, z]) + m
    b = np.select(cond, [z, z, x, c, c, x]) + m
    return np.stack([r, g, b], -1)

# Distinct ply colours (RGB 0..1), cycled if K is larger.
_PLY_COLORS = np.array([
    [0.85, 0.30, 0.25], [0.25, 0.55, 0.85], [0.35, 0.70, 0.35],
    [0.85, 0.65, 0.20], [0.60, 0.40, 0.75], [0.30, 0.70, 0.70],
    [0.80, 0.45, 0.60], [0.55, 0.55, 0.55],
])


# --------------------------------------------------------------------------
# Perlin noise (improved Perlin, Ken Perlin) -- vectorised, dependency-free.
# Used as a smooth 3D vector field to perturb the otherwise-perfect helices,
# mirroring perlin_vector_field() in the reference FiberGenerator (the field
# that bends flyaway hairs / adds organic wander).
# --------------------------------------------------------------------------
_GRAD3 = np.array([
    [1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0],
    [1, 0, 1], [-1, 0, 1], [1, 0, -1], [-1, 0, -1],
    [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1]], dtype=np.float64)


def _make_perm(seed=0):
    rng = np.random.default_rng(seed)
    p = np.arange(256, dtype=np.int64)
    rng.shuffle(p)
    return np.concatenate([p, p])               # length 512


_PERM = _make_perm(0)


def _fade(t):
    return t * t * t * (t * (t * 6 - 15) + 10)


def _grad(h, x, y, z):
    g = _GRAD3[h % 12]
    return g[..., 0] * x + g[..., 1] * y + g[..., 2] * z


def _perlin3(coords, perm):
    """Scalar improved-Perlin noise at coords (N,3) -> (N,) in ~[-1,1]."""
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    xi = np.floor(x).astype(np.int64)
    yi = np.floor(y).astype(np.int64)
    zi = np.floor(z).astype(np.int64)
    xf, yf, zf = x - xi, y - yi, z - zi
    Xi, Yi, Zi = xi & 255, yi & 255, zi & 255
    u, v, w = _fade(xf), _fade(yf), _fade(zf)
    A = perm[Xi] + Yi; AA = perm[A] + Zi; AB = perm[A + 1] + Zi
    Bx = perm[Xi + 1] + Yi; BA = perm[Bx] + Zi; BB = perm[Bx + 1] + Zi

    def lerp(a, b, t):
        return a + t * (b - a)

    x1 = lerp(_grad(perm[AA], xf, yf, zf),
              _grad(perm[BA], xf - 1, yf, zf), u)
    x2 = lerp(_grad(perm[AB], xf, yf - 1, zf),
              _grad(perm[BB], xf - 1, yf - 1, zf), u)
    y1 = lerp(x1, x2, v)
    x3 = lerp(_grad(perm[AA + 1], xf, yf, zf - 1),
              _grad(perm[BA + 1], xf - 1, yf, zf - 1), u)
    x4 = lerp(_grad(perm[AB + 1], xf, yf - 1, zf - 1),
              _grad(perm[BB + 1], xf - 1, yf - 1, zf - 1), u)
    y2 = lerp(x3, x4, v)
    return lerp(y1, y2, w)


def perlin_field(coords, scale, perm=_PERM):
    """Smooth 3D Perlin vector field at coords (N,3) * scale -> (N,3) in
    ~[-1,1]^3. Three decorrelated channels give the x/y/z components -- the
    same field as the reference's perlin_vector_field()."""
    c = np.atleast_2d(np.asarray(coords, float)) * scale
    nx = _perlin3(c, perm)
    ny = _perlin3(c + 19.1, perm)
    nz = _perlin3(c + 53.7, perm)
    return np.stack([nx, ny, nz], axis=1)


def perlin_warp(curve, amount, scale, perm=_PERM):
    """Displace polyline `curve` (M,3) by `amount` * Perlin field at its
    world position * `scale`."""
    if amount <= 1e-9 or scale <= 0:
        return curve
    return curve + amount * perlin_field(curve, scale, perm)


def make_flyaways(P, src_curves, n_hairs, length, step, amt, scale,
                  outward=0.7, seed=0):
    """Stray "flyaway" hairs that give yarn its fuzz (Zhao 2016 sec. 4.4).

    Each hair starts at a random vertex of a random source curve (a fiber, or
    a ply if no fibers) and grows by following base_dir + a Perlin vector
    field -- exactly the reference's `flow += (perlin + tangent)*step`, but
    with `base_dir` biased radially OUTWARD (away from centerline P) by
    `outward` so the hair leaves the yarn surface. Vectorised over all hairs.

    Returns a list of (hair_pts (k,3), root_frac in [0,1]).
    """
    if n_hairs <= 0 or not len(src_curves):
        return []
    rng = np.random.default_rng(seed)
    M = len(P)
    H = int(n_hairs)
    ci = rng.integers(0, len(src_curves), H)
    starts = np.empty((H, 3)); base = np.empty((H, 3)); roots = np.empty(H)
    for h in range(H):
        cur = src_curves[ci[h]]
        n = len(cur)
        i = int(rng.integers(1, max(n - 1, 2)))
        t = cur[min(i + 1, n - 1)] - cur[max(i - 1, 0)]
        t /= (np.linalg.norm(t) + 1e-9)
        rad = cur[i] - P[min(i, M - 1)]
        rn = np.linalg.norm(rad)
        rad = rad / rn if rn > 1e-6 else t
        d = outward * rad + (1.0 - outward) * t
        base[h] = d / (np.linalg.norm(d) + 1e-9)
        starts[h] = cur[i]
        roots[h] = i / max(M - 1, 1)
    # per-hair length -> step count; grow all hairs together, freezing each
    # once it reaches its length.
    Ls = length * rng.uniform(0.5, 1.5, H)
    nst = np.maximum((Ls / max(step, 1e-6)).astype(int), 2)
    kmax = int(nst.max())
    pos = starts.copy()
    track = np.empty((H, kmax + 1, 3))
    track[:, 0] = pos
    for k in range(kmax):
        active = (k < nst)[:, None]
        pert = perlin_field(pos, scale) * amt
        pos = pos + (pert + base) * step * active
        track[:, k + 1] = pos
    return [(track[h, :nst[h] + 1], float(roots[h])) for h in range(H)]


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_centerlines(path: str) -> list[np.ndarray]:
    """Return a list of (M,3) centerlines from a yarn npz.

    Handles `yarn_state_v1` (per-curve `curve_<sid>` arrays; the lone `yarn`
    key is the single-strand alias and is skipped to avoid a duplicate) and
    falls back to any 2-D (*,3) float array in the file.
    """
    d = np.load(path, allow_pickle=True)
    keys = list(d.keys())
    curves = []
    curve_keys = sorted(k for k in keys if k.startswith('curve_'))
    if curve_keys:
        for k in curve_keys:
            a = np.asarray(d[k], np.float64)
            if a.ndim == 2 and a.shape[1] == 3 and len(a) >= 2:
                curves.append(a)
    if not curves:                              # generic fallback
        for k in keys:
            a = np.asarray(d[k])
            if a.ndim == 2 and a.shape[1] == 3 and len(a) >= 2:
                curves.append(a.astype(np.float64))
    if not curves:
        raise ValueError(f'No (*,3) centerline found in {path}; keys={keys}')
    return curves


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------
def rmf_frame(P: np.ndarray):
    """Rotation-minimising frame for polyline P (M,3).

    Returns (T, N, B, s): unit tangent, normal, binormal (each (M,3)) and
    cumulative arclength s (M,). The normal is carried forward and the
    tangent projected out at each step, reseeding on a ~180 deg flip, so the
    frame does not spin -- the same scheme as toy_chain_spring._tube_mesh.
    """
    P = np.asarray(P, np.float64)
    M = len(P)
    T = np.empty_like(P)
    T[1:-1] = P[2:] - P[:-2]
    T[0] = P[1] - P[0]
    T[-1] = P[-1] - P[-2]
    T /= (np.linalg.norm(T, axis=1, keepdims=True) + 1e-12)

    a = np.array([0.0, 0.0, 1.0])
    if abs(float(T[0] @ a)) > 0.9:
        a = np.array([0.0, 1.0, 0.0])
    N = np.empty_like(P)
    n = a - (a @ T[0]) * T[0]
    N[0] = n / (np.linalg.norm(n) + 1e-12)
    for k in range(1, M):
        n = N[k - 1] - (N[k - 1] @ T[k]) * T[k]
        nn = np.linalg.norm(n)
        if nn < 1e-8:                           # tangent flipped ~180 deg
            a2 = np.array([1.0, 0.0, 0.0])
            if abs(float(T[k] @ a2)) > 0.9:
                a2 = np.array([0.0, 1.0, 0.0])
            n = a2 - (a2 @ T[k]) * T[k]
            nn = np.linalg.norm(n)
        N[k] = n / (nn + 1e-12)
    B = np.cross(T, N)

    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    return T, N, B, s


def ply_centers(P, N, B, s, n_plies, pitch, radius, sign=1.0, phase0=0.0,
                core=False):
    """List of K (M,3) ply centerlines twisting around P (vectorised).

    With `core=True`, the bare centerline P is appended as one extra ply at
    radius 0, filling the otherwise-hollow centre of the ply ring.
    """
    turns = sign * 2.0 * np.pi * s / max(float(pitch), 1e-6)   # (M,)
    out = []
    for p in range(int(n_plies)):
        phi = phase0 + 2.0 * np.pi * p / max(int(n_plies), 1)
        th = turns + phi
        off = (np.cos(th)[:, None] * N + np.sin(th)[:, None] * B)
        out.append(P + radius * off)
    if core:
        out.append(P.copy())
    return out


_GOLDEN = np.pi * (3.0 - np.sqrt(5.0))          # phyllotaxis angle (rad)


def fiber_curves(ply, n_fibers, fiber_pitch, bundle_radius, sign=1.0,
                 migration=0.0, mig_freq=1.0):
    """List of F (M,3) fibers twisting around ONE ply centerline `ply`.

    This is the fiber level of the model (Zhao 2016 sec. 4.2): fibers fill
    the ply cross-section disk (phyllotaxis sampling of base radius/angle)
    and each rides a *finer* helix around the ply centre:
        theta_i(s) = sign*2pi*s/fiber_pitch + psi_i
        fiber_i(s) = ply(s) + r_i (cos theta_i Nf(s) + sin theta_i Bf(s))
    `migration` in [0,1] makes r_i oscillate down to (1-migration)*r_i along
    the strand (mig_freq oscillations per fiber turn), the model's fiber
    migration that gives the bundle its lived-in look.
    """
    F = max(int(n_fibers), 1)
    _, Nf, Bf, s = rmf_frame(ply)
    turns = sign * 2.0 * np.pi * s / max(float(fiber_pitch), 1e-6)   # (M,)
    out = []
    for i in range(F):
        r_i = bundle_radius * np.sqrt((i + 0.5) / F)   # ~uniform disk fill
        psi = i * _GOLDEN
        th = turns + psi
        base = np.full(len(ply), r_i)
        if migration > 1e-6:
            base = r_i * ((1.0 - migration)
                          + migration * 0.5 * (np.cos(mig_freq * th + psi) + 1.0))
        off = np.cos(th)[:, None] * Nf + np.sin(th)[:, None] * Bf
        out.append(ply + base[:, None] * off)
    return out


def tube_mesh(poly: np.ndarray, radius: float, nseg: int = 12):
    """(verts, faces) of an open tube around polyline `poly` (M>=2), using a
    rotation-minimising frame. Vectorised build of the ring vertices/faces."""
    P = np.asarray(poly, np.float64)
    M = len(P)
    if M < 2:
        return None
    _, Nrm, Bn, _ = rmf_frame(P)
    ang = np.linspace(0.0, 2.0 * np.pi, nseg, endpoint=False)
    cos = np.cos(ang)[None, :, None]            # (1,nseg,1)
    sin = np.sin(ang)[None, :, None]
    verts = (P[:, None, :]
             + radius * (cos * Nrm[:, None, :] + sin * Bn[:, None, :]))
    verts = verts.reshape(-1, 3)                # (M*nseg, 3)
    i = np.arange(M - 1)[:, None]
    j = np.arange(nseg)[None, :]
    jn = (j + 1) % nseg
    b0 = i * nseg
    b1 = (i + 1) * nseg
    f1 = np.stack([b0 + j, b1 + j, b1 + jn], -1).reshape(-1, 3)
    f2 = np.stack([b0 + j, b1 + jn, b0 + jn], -1).reshape(-1, 3)
    faces = np.vstack([f1, f2]).astype(np.int64)
    return verts, faces


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------
def export(curves_plies, params, stamp=None, curves_fibers=None,
           curves_flyaways=None):
    """Write ply (and optional fiber / flyaway) centerlines (npz) + merged
    tube mesh (obj of the plies). Returns paths."""
    stamp = stamp or time.strftime('%Y%m%d_%H%M%S')
    npz_path = f'yarn_plies_{stamp}.npz'
    obj_path = f'yarn_plies_{stamp}.obj'

    save = {'fmt': 'yarn_plies_v1', 'n_curves': len(curves_plies)}
    save.update({f'param_{k}': v for k, v in params.items()})
    for c, plies in enumerate(curves_plies):
        save[f'n_plies_{c}'] = len(plies)
        for p, ply in enumerate(plies):
            save[f'ply_{c}_{p}'] = ply.astype(np.float32)
    if curves_fibers:
        for c, per_ply in enumerate(curves_fibers):
            for p, fibers in enumerate(per_ply):
                save[f'n_fibers_{c}_{p}'] = len(fibers)
                for i, fc in enumerate(fibers):
                    save[f'fiber_{c}_{p}_{i}'] = fc.astype(np.float32)
    if curves_flyaways:
        for c, hairs in enumerate(curves_flyaways):
            save[f'n_fly_{c}'] = len(hairs)
            save[f'fly_root_{c}'] = np.array([rf for _, rf in hairs],
                                             np.float32)
            for i, (pts, _) in enumerate(hairs):
                save[f'fly_{c}_{i}'] = pts.astype(np.float32)
    np.savez_compressed(npz_path, **save)

    # Merged OBJ of all ply tubes.
    R = float(params['tube_radius'])
    nseg = int(params['nseg'])
    with open(obj_path, 'w') as fh:
        base = 0
        for plies in curves_plies:
            for ply in plies:
                tm = tube_mesh(ply, R, nseg)
                if tm is None:
                    continue
                v, f = tm
                for x, y, z in v:
                    fh.write(f'v {x:.5f} {y:.5f} {z:.5f}\n')
                for a, b, cc in f + 1 + base:    # OBJ is 1-indexed
                    fh.write(f'f {a} {b} {cc}\n')
                base += len(v)
    return npz_path, obj_path


# --------------------------------------------------------------------------
# Viewer
# --------------------------------------------------------------------------
def view(curves, state):
    import polyscope as ps
    import polyscope.imgui as psim

    ps.init()
    ps.set_up_dir('z_up')
    ps.set_transparency_mode('pretty')          # so translucent ply tubes show

    # Precompute frames once (twist params don't change them).
    frames = [rmf_frame(P) for P in curves]
    arclen = sum(float(s[-1]) for *_ , s in frames)
    print(f'[yarn_plies] {len(curves)} centerline(s), '
          f'total arclength {arclen:.0f}')

    # Show the original centerline(s) faintly for reference.
    for c, P in enumerate(curves):
        e = np.column_stack([np.arange(len(P) - 1), np.arange(1, len(P))])
        cn = ps.register_curve_network(f'centerline_{c}', P, e,
                                        radius=0.0015, color=(0.2, 0.2, 0.2))
        cn.set_enabled(state['show_center'])

    built = {'names': []}

    def rebuild():
        for nm in built['names']:
            if ps.has_surface_mesh(nm):
                ps.remove_surface_mesh(nm)
            if ps.has_curve_network(nm):
                ps.remove_curve_network(nm)
        built['names'] = []
        state['curves_plies'] = []
        state['curves_fibers'] = []
        state['curves_flyaways'] = []
        fib = state['fibers']
        amt = state['perlin_amt']
        sc = state['perlin_scale']
        # When fibers are on, draw ply tubes thinner/translucent so the
        # finer bundle inside is visible.
        ply_tube_r = state['tube_radius'] * (0.55 if fib else 1.0)
        for c, (P, fr) in enumerate(zip(curves, frames)):
            _, N, B, s = fr
            plies = ply_centers(P, N, B, s, state['n_plies'], state['pitch'],
                                state['radius'], state['sign'], state['phase0'],
                                core=state['core'])
            warped_plies = []
            per_ply_fibers = []
            fib_nodes, fib_edges, fib_cols = [], [], []
            base = 0
            for p, ply_raw in enumerate(plies):
                # Perlin field warps the ply centre -> coherent organic wobble.
                ply = perlin_warp(ply_raw, amt, sc)
                warped_plies.append(ply)
                col = tuple(_PLY_COLORS[p % len(_PLY_COLORS)])
                M = len(ply)
                if state['as_tubes']:
                    tm = tube_mesh(ply, ply_tube_r, state['nseg'])
                    if tm is not None:
                        v, f = tm
                        nm = f'ply_{c}_{p}'
                        m = ps.register_surface_mesh(nm, v, f, color=col,
                                                     smooth_shade=True)
                        if state['rainbow']:
                            ring = np.arange(len(v)) // state['nseg']
                            m.add_color_quantity(
                                'rainbow', rainbow_rgb(ring / max(M - 1, 1)),
                                enabled=True)
                        if fib:
                            m.set_transparency(0.4)
                        built['names'].append(nm)
                else:
                    e = np.column_stack([np.arange(M - 1), np.arange(1, M)])
                    nm = f'ply_{c}_{p}'
                    cn = ps.register_curve_network(nm, ply, e,
                                                   radius=0.002, color=col)
                    if state['rainbow']:
                        cn.add_color_quantity(
                            'rainbow', rainbow_rgb(np.arange(M) / max(M - 1, 1)),
                            defined_on='nodes', enabled=True)
                    built['names'].append(nm)
                if fib:
                    fibers = fiber_curves(
                        ply, state['fibers_per_ply'], state['fiber_pitch'],
                        state['bundle_radius'], state['sign'],
                        state['migration'])
                    # extra finer/higher-frequency fuzz per fiber
                    fibers = [perlin_warp(fc, amt * 0.5, sc * 4.0)
                              for fc in fibers]
                    per_ply_fibers.append(fibers)
                    for fc in fibers:
                        n = len(fc)
                        fib_nodes.append(fc)
                        fib_edges.append(np.column_stack(
                            [np.arange(n - 1), np.arange(1, n)]) + base)
                        if state['rainbow']:
                            fib_cols.append(rainbow_rgb(
                                np.arange(n) / max(n - 1, 1)))
                        else:
                            fib_cols.append(np.tile(col, (n, 1)))
                        base += n
            state['curves_plies'].append(warped_plies)
            state['curves_fibers'].append(per_ply_fibers)
            if fib and fib_nodes:
                nm = f'fibers_{c}'
                cn = ps.register_curve_network(
                    nm, np.vstack(fib_nodes), np.vstack(fib_edges),
                    radius=0.0009)
                cn.add_color_quantity('ply', np.vstack(fib_cols),
                                      enabled=True)
                built['names'].append(nm)

            # Flyaway hairs (fuzz): grown from the fibers (or plies) via the
            # Perlin field, biased outward so they leave the surface.
            hairs = []
            if state['fly'] and state['fly_n'] > 0:
                src = ([f for pf in per_ply_fibers for f in pf]
                       if (fib and per_ply_fibers) else warped_plies)
                hairs = make_flyaways(P, src, state['fly_n'], state['fly_len'],
                                      state['fly_step'], state['fly_amt'], sc,
                                      state['fly_outward'])
            state['curves_flyaways'].append(hairs)
            if hairs:
                hn, he, hc, bb = [], [], [], 0
                for pts, rf in hairs:
                    k = len(pts)
                    hn.append(pts)
                    he.append(np.column_stack([np.arange(k - 1),
                                               np.arange(1, k)]) + bb)
                    hc.append(np.tile(rainbow_rgb(rf) if state['rainbow']
                                      else [0.92, 0.90, 0.84], (k, 1)))
                    bb += k
                nm = f'flyaways_{c}'
                cnf = ps.register_curve_network(nm, np.vstack(hn),
                                                np.vstack(he), radius=0.0006)
                cnf.add_color_quantity('fly', np.vstack(hc), enabled=True)
                built['names'].append(nm)

    rebuild()

    def gui():
        ch = False
        psim.TextUnformatted('Ply level of the coaxial-helix yarn model')
        psim.Separator()

        c, v = psim.SliderInt('Ply count K', state['n_plies'], 1, 24)
        if c:
            state['n_plies'] = v; ch = True
        c, v = psim.SliderFloat('Twist pitch (len / turn)', state['pitch'],
                                20.0, 2000.0)
        if c:
            state['pitch'] = v; ch = True
        c, v = psim.SliderFloat('Ply offset radius', state['radius'],
                                0.0, 80.0)
        if c:
            state['radius'] = v; ch = True
        c, v = psim.SliderFloat('Tube radius', state['tube_radius'],
                                0.5, 40.0)
        if c:
            state['tube_radius'] = v; ch = True
        c, v = psim.SliderFloat('Phase offset', state['phase0'],
                                0.0, 2.0 * np.pi)
        if c:
            state['phase0'] = v; ch = True
        c, v = psim.SliderFloat('Perlin amount', state['perlin_amt'],
                                0.0, 15.0)
        if c:
            state['perlin_amt'] = v; ch = True
        c, v = psim.SliderFloat('Perlin scale (freq)', state['perlin_scale'],
                                0.002, 0.1)
        if c:
            state['perlin_scale'] = v; ch = True

        if psim.Button('Flip twist (S/Z)'):
            state['sign'] = -state['sign']; ch = True
        psim.SameLine()
        c, v = psim.Checkbox('Tubes', state['as_tubes'])
        if c:
            state['as_tubes'] = v; ch = True
        psim.SameLine()
        c, v = psim.Checkbox('Centerline', state['show_center'])
        if c:
            state['show_center'] = v
            for i in range(len(curves)):
                if ps.has_curve_network(f'centerline_{i}'):
                    ps.get_curve_network(f'centerline_{i}').set_enabled(v)

        c, v = psim.Checkbox('Center ply (core)', state['core'])
        if c:
            state['core'] = v; ch = True
        psim.SameLine()
        c, v = psim.Checkbox('Rainbow (along path)', state['rainbow'])
        if c:
            state['rainbow'] = v; ch = True

        # turns over the whole strand, for intuition
        turns = arclen / max(state['pitch'], 1e-6)
        psim.TextUnformatted(f'~{turns:.1f} full ply twists over the strand')

        psim.Separator()
        psim.TextUnformatted('Fiber level (finer helix inside each ply)')
        c, v = psim.Checkbox('Fibers', state['fibers'])
        if c:
            state['fibers'] = v; ch = True
        if state['fibers']:
            c, v = psim.SliderInt('Fibers / ply', state['fibers_per_ply'],
                                  1, 80)
            if c:
                state['fibers_per_ply'] = v; ch = True
            c, v = psim.SliderFloat('Fiber twist pitch', state['fiber_pitch'],
                                    8.0, 800.0)
            if c:
                state['fiber_pitch'] = v; ch = True
            c, v = psim.SliderFloat('Fiber bundle radius',
                                    state['bundle_radius'], 0.2, 30.0)
            if c:
                state['bundle_radius'] = v; ch = True
            c, v = psim.SliderFloat('Fiber migration', state['migration'],
                                    0.0, 1.0)
            if c:
                state['migration'] = v; ch = True
            ftot = (state['n_plies'] * state['fibers_per_ply']
                    * len(curves))
            psim.TextUnformatted(f'{ftot} fibers total (rebuild on change)')

        psim.Separator()
        psim.TextUnformatted('Flyaway hairs (fuzz)')
        c, v = psim.Checkbox('Flyaways', state['fly'])
        if c:
            state['fly'] = v; ch = True
        if state['fly']:
            c, v = psim.SliderInt('Hairs / strand', state['fly_n'], 0, 8000)
            if c:
                state['fly_n'] = v; ch = True
            c, v = psim.SliderFloat('Hair length', state['fly_len'], 5.0, 200.0)
            if c:
                state['fly_len'] = v; ch = True
            c, v = psim.SliderFloat('Hair wander', state['fly_amt'], 0.0, 3.0)
            if c:
                state['fly_amt'] = v; ch = True
            c, v = psim.SliderFloat('Hair outward', state['fly_outward'],
                                    0.0, 1.0)
            if c:
                state['fly_outward'] = v; ch = True

        if psim.Button('Export npz + obj'):
            params = dict(n_plies=state['n_plies'], pitch=state['pitch'],
                          radius=state['radius'],
                          tube_radius=state['tube_radius'],
                          nseg=state['nseg'], sign=state['sign'],
                          phase0=state['phase0'], fibers=state['fibers'],
                          fibers_per_ply=state['fibers_per_ply'],
                          fiber_pitch=state['fiber_pitch'],
                          bundle_radius=state['bundle_radius'],
                          migration=state['migration'],
                          perlin_amt=state['perlin_amt'],
                          perlin_scale=state['perlin_scale'],
                          core=state['core'], rainbow=state['rainbow'],
                          fly=state['fly'], fly_n=state['fly_n'])
            np_, ob_ = export(state['curves_plies'], params,
                              curves_fibers=state['curves_fibers'],
                              curves_flyaways=state.get('curves_flyaways'))
            print(f'[yarn_plies] wrote {np_} and {ob_}')

        if ch:
            rebuild()

    ps.set_user_callback(gui)
    ps.show()


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('npz', help='yarn centerline npz')
    ap.add_argument('--n_plies', type=int, default=8)
    ap.add_argument('--pitch', type=float, default=300.0,
                    help='arclength of one full ply twist (model yarn_alpha); '
                         'bigger = looser twist')
    ap.add_argument('--radius', type=float, default=16.0,
                    help='ply offset from centerline (model yarn_radius)')
    ap.add_argument('--tube_radius', type=float, default=9.0)
    ap.add_argument('--core', action='store_true', default=True,
                    help='add a center ply filling the hollow core')
    ap.add_argument('--no_core', dest='core', action='store_false')
    ap.add_argument('--rainbow', action='store_true',
                    help='colour the yarn as a rainbow along its path')
    # flyaway hairs (fuzz)
    ap.add_argument('--fly', action='store_true', help='add flyaway hairs')
    ap.add_argument('--fly_n', type=int, default=1500,
                    help='flyaway hairs per strand')
    ap.add_argument('--fly_len', type=float, default=45.0)
    ap.add_argument('--fly_step', type=float, default=5.0)
    ap.add_argument('--fly_amt', type=float, default=1.2,
                    help='hair Perlin wander strength')
    ap.add_argument('--fly_outward', type=float, default=0.7,
                    help='radial bias so hairs leave the surface (0..1)')
    ap.add_argument('--nseg', type=int, default=12)
    ap.add_argument('--sign', type=float, default=1.0, help='+1 / -1 twist dir')
    ap.add_argument('--phase0', type=float, default=0.0)
    # fiber level
    ap.add_argument('--fibers', action='store_true',
                    help='also build the finer fiber helix inside each ply')
    ap.add_argument('--fibers_per_ply', type=int, default=24)
    ap.add_argument('--fiber_pitch', type=float, default=80.0)
    ap.add_argument('--bundle_radius', type=float, default=6.0,
                    help='ply cross-section radius the fibers fill')
    ap.add_argument('--migration', type=float, default=0.3,
                    help='fiber radius oscillation in [0,1]')
    # Perlin noise (organic wander)
    ap.add_argument('--perlin_amt', type=float, default=2.0,
                    help='Perlin displacement magnitude (voxels); 0 = off')
    ap.add_argument('--perlin_scale', type=float, default=0.03,
                    help='Perlin spatial frequency (bigger = finer wobble)')
    ap.add_argument('--lines', action='store_true',
                    help='render plies as curves, not tubes')
    ap.add_argument('--export', action='store_true',
                    help='write npz+obj and exit (no viewer)')
    args = ap.parse_args()

    curves = load_centerlines(args.npz)
    state = dict(n_plies=args.n_plies, pitch=args.pitch, radius=args.radius,
                 tube_radius=args.tube_radius, nseg=args.nseg, sign=args.sign,
                 phase0=args.phase0, as_tubes=not args.lines,
                 show_center=True, curves_plies=[], curves_fibers=[],
                 fibers=args.fibers, fibers_per_ply=args.fibers_per_ply,
                 fiber_pitch=args.fiber_pitch, bundle_radius=args.bundle_radius,
                 migration=args.migration, perlin_amt=args.perlin_amt,
                 perlin_scale=args.perlin_scale, core=args.core,
                 rainbow=args.rainbow, fly=args.fly, fly_n=args.fly_n,
                 fly_len=args.fly_len, fly_step=args.fly_step,
                 fly_amt=args.fly_amt, fly_outward=args.fly_outward,
                 curves_flyaways=[])

    if args.export:
        frames = [rmf_frame(P) for P in curves]
        cps, cfs, cflys = [], [], []
        for P, (_, N, B, s) in zip(curves, frames):
            plies = [perlin_warp(ply, args.perlin_amt, args.perlin_scale)
                     for ply in ply_centers(P, N, B, s, args.n_plies,
                                            args.pitch, args.radius, args.sign,
                                            args.phase0, core=args.core)]
            cps.append(plies)
            per_ply = None
            if args.fibers:
                per_ply = [[perlin_warp(fc, args.perlin_amt * 0.5,
                                        args.perlin_scale * 4.0)
                            for fc in fiber_curves(
                                ply, args.fibers_per_ply, args.fiber_pitch,
                                args.bundle_radius, args.sign,
                                args.migration)]
                           for ply in plies]
                cfs.append(per_ply)
            if args.fly:
                src = ([f for pf in per_ply for f in pf] if per_ply
                       else plies)
                cflys.append(make_flyaways(
                    P, src, args.fly_n, args.fly_len, args.fly_step,
                    args.fly_amt, args.perlin_scale, args.fly_outward))
        params = dict(n_plies=args.n_plies, pitch=args.pitch,
                      radius=args.radius, tube_radius=args.tube_radius,
                      nseg=args.nseg, sign=args.sign, phase0=args.phase0,
                      fibers=args.fibers, fibers_per_ply=args.fibers_per_ply,
                      fiber_pitch=args.fiber_pitch,
                      bundle_radius=args.bundle_radius,
                      migration=args.migration, perlin_amt=args.perlin_amt,
                      perlin_scale=args.perlin_scale, core=args.core,
                      fly=args.fly, fly_n=args.fly_n)
        np_, ob_ = export(cps, params, curves_fibers=(cfs or None),
                          curves_flyaways=(cflys or None))
        print(f'[yarn_plies] wrote {np_} and {ob_}')
        return

    view(curves, state)


if __name__ == '__main__':
    main()
