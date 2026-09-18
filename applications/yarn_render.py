"""
Render the procedural yarn (plies / fibers) with Mitsuba 3.

The paper (Zhao et al. 2016) renders explicit fiber geometry with a *modified
Mitsuba 0.5* + a fiber-scattering BCSDF. That build is an unbuildable
Windows-era C++ project, so this uses stock Mitsuba 3 (pip, Python, GPU):
fibers are rendered as `linearcurve` hair primitives (their realistic
fiber-level look) and plies as swept tube meshes, under a clean constant
environment (path tracer -> soft ambient-occlusion shading) with an optional
backdrop. Reuses the geometry from yarn_plies.py so what you tuned in the
polyscope viewer is what gets rendered.

Input is either:
  * a centerline npz (yarn_state_v1) -> plies/fibers generated from the same
    CLI params as yarn_plies.py, or
  * a yarn_plies_v1 npz exported by yarn_plies.py -> rendered as-is.

Usage:
    python yarn_render.py yarn_likely_20260527_052928.npz --fibers
    python yarn_render.py yarn_plies_20260528_120000.npz --spp 256 --out r.png
"""
from __future__ import annotations
import argparse
import os
import tempfile
import time

import numpy as np

import yarn_plies as Y          # reuse geometry: rmf_frame, ply_centers, ...


# --------------------------------------------------------------------------
# Geometry: build (or load) warped plies + fibers
# --------------------------------------------------------------------------
def build_geometry(args):
    """Return (plies_by_curve, fibers_by_curve, flyaways_by_curve).

    plies_by_curve  : list over centerlines of [ (M,3) ply polyline, ... ]
    fibers_by_curve : same nesting, each ply -> [ (M,3) fiber, ... ]  (or None)
    flyaways_by_curve: list over centerlines of [ (pts, root_frac), ... ]
    """
    d = np.load(args.npz, allow_pickle=True)
    keys = list(d.keys())
    fmt = str(d['fmt']) if 'fmt' in keys else ''

    if fmt == 'yarn_plies_v1' or any(k.startswith('ply_') for k in keys):
        # Pre-generated geometry: load straight through.
        nc = int(d['n_curves'])
        plies_by_curve, fibers_by_curve, flyaways_by_curve = [], [], []
        has_fib = any(k.startswith('fiber_') for k in keys)
        for c in range(nc):
            np_c = int(d[f'n_plies_{c}'])
            plies_by_curve.append([np.asarray(d[f'ply_{c}_{p}'], np.float64)
                                   for p in range(np_c)])
            if has_fib and not args.plies_only:
                fibers_by_curve.append([
                    [np.asarray(d[f'fiber_{c}_{p}_{i}'], np.float64)
                     for i in range(int(d.get(f'n_fibers_{c}_{p}', 0)))]
                    for p in range(np_c)])
            else:
                fibers_by_curve.append(None)
            nfly = int(d.get(f'n_fly_{c}', 0))
            if nfly:
                rf = np.asarray(d[f'fly_root_{c}'], np.float64)
                flyaways_by_curve.append(
                    [(np.asarray(d[f'fly_{c}_{i}'], np.float64), float(rf[i]))
                     for i in range(nfly)])
            else:
                flyaways_by_curve.append([])
        return plies_by_curve, fibers_by_curve, flyaways_by_curve

    # Centerline npz -> generate, mirroring yarn_plies' pipeline.
    curves = Y.load_centerlines(args.npz)
    plies_by_curve, fibers_by_curve, flyaways_by_curve = [], [], []
    for P in curves:
        _, N, B, s = Y.rmf_frame(P)
        raw = Y.ply_centers(P, N, B, s, args.n_plies, args.pitch,
                            args.radius, args.sign, args.phase0,
                            core=args.core)
        plies = [Y.perlin_warp(pl, args.perlin_amt, args.perlin_scale)
                 for pl in raw]
        plies_by_curve.append(plies)
        per_ply = None
        if args.fibers and not args.plies_only:
            per_ply = []
            for pl in plies:
                fibers = Y.fiber_curves(pl, args.fibers_per_ply,
                                        args.fiber_pitch, args.bundle_radius,
                                        args.sign, args.migration)
                per_ply.append([Y.perlin_warp(fc, args.perlin_amt * 0.5,
                                              args.perlin_scale * 4.0)
                                for fc in fibers])
            fibers_by_curve.append(per_ply)
        else:
            fibers_by_curve.append(None)
        if args.fly:
            if args.color_plies and not args.rainbow:
                # Per-ply flyaway generation so each hair carries the
                # ply it grew from → render can colour it to match.
                n_pp = max(int(args.fly_n / max(1, len(plies))), 1)
                flat = []
                for ply_idx, pl in enumerate(plies):
                    src = ([per_ply[ply_idx]] if per_ply
                           else [pl])
                    src = src[0] if isinstance(src[0], list) else src
                    hairs = Y.make_flyaways(
                        P, src, n_pp, args.fly_len, args.fly_step,
                        args.fly_amt, args.perlin_scale, args.fly_outward,
                        seed=ply_idx * 9973)
                    flat.extend((pts, rf, ply_idx) for pts, rf in hairs)
                flyaways_by_curve.append(flat)
            else:
                src = ([f for pf in per_ply for f in pf] if per_ply else plies)
                flyaways_by_curve.append(Y.make_flyaways(
                    P, src, args.fly_n, args.fly_len, args.fly_step,
                    args.fly_amt, args.perlin_scale, args.fly_outward))
        else:
            flyaways_by_curve.append([])
    return plies_by_curve, fibers_by_curve, flyaways_by_curve


# --------------------------------------------------------------------------
# Mitsuba file writers
# --------------------------------------------------------------------------
def band_slice(poly, b, bands):
    """Contiguous arclength chunk b of `bands` from polyline `poly`, overlap
    one point so consecutive bands stay connected."""
    n = len(poly)
    i0 = int(round(b * n / bands))
    i1 = int(round((b + 1) * n / bands)) + 1
    return poly[i0:i1]


def write_curve_file(path, fibers, radius):
    """Mitsuba linearcurve file: 'x y z radius' per control point, fibers
    separated by a blank line. Returns how many fibers were written."""
    nwritten = 0
    with open(path, 'w') as fh:
        for fc in fibers:
            if len(fc) < 2:
                continue
            arr = np.column_stack([fc, np.full(len(fc), radius)])
            np.savetxt(fh, arr, fmt='%.4f')
            fh.write('\n')
            nwritten += 1
    return nwritten


def write_obj(path, plies, tube_radius, nseg):
    """OBJ of swept tubes for a set of ply polylines."""
    with open(path, 'w') as fh:
        base = 0
        for pl in plies:
            tm = Y.tube_mesh(pl, tube_radius, nseg)
            if tm is None:
                continue
            v, f = tm
            for x, y, z in v:
                fh.write(f'v {x:.5f} {y:.5f} {z:.5f}\n')
            for a, b, cc in f + 1 + base:
                fh.write(f'f {a} {b} {cc}\n')
            base += len(v)


def sigma_a_from_color(col, beta_n=0.35):
    """PBRT/Chiang inverse mapping: the hair absorption coefficient sigma_a
    that reproduces a target reflectance `col`, so the built-in `hair` BCSDF
    can take arbitrary (e.g. rainbow) colours rather than only melanin tones.
    """
    col = np.clip(np.asarray(col, float), 1e-3, 0.999)
    d = (5.969 - 0.215 * beta_n + 2.532 * beta_n ** 2 - 10.73 * beta_n ** 3
         + 5.574 * beta_n ** 4 + 0.245 * beta_n ** 5)
    return [float(x) for x in (np.log(col) / d) ** 2]


def synth_envmap(path, intensity=1.0, h=512, w=1024):
    """Write a simple procedural studio HDR (EXR): a cool->warm vertical sky
    gradient plus one soft bright 'sun' for directional highlights."""
    import mitsuba as mi
    y = np.linspace(0.0, 1.0, h)[:, None]
    grad = np.stack([0.55 + 0.45 * (1 - y) + 0 * np.zeros(w),
                     0.62 + 0.42 * (1 - y) + 0 * np.zeros(w),
                     0.80 + 0.30 * (1 - y) + 0 * np.zeros(w)], -1)
    yy, xx = np.mgrid[0:h, 0:w]
    dd = ((yy - 0.30 * h) / h) ** 2 + ((xx - 0.68 * w) / w) ** 2
    sun = np.exp(-dd / 0.0035)[..., None] * np.array([7.0, 6.6, 6.0])
    hdr = ((grad + sun) * intensity).astype(np.float32)
    mi.Bitmap(hdr).write(path)
    return path


# --------------------------------------------------------------------------
# Scene + render
# --------------------------------------------------------------------------
def render(plies_by_curve, fibers_by_curve, flyaways_by_curve, args,
           fixed_bbox=None, extra_shapes=None):
    """Render one frame. Optional `fixed_bbox=(lo, hi)` overrides the auto
    camera-framing bbox (use this when rendering an animation to keep the
    camera stable across frames). Optional `extra_shapes` is a dict of name
    -> mitsuba shape spec added to the scene (e.g. pachinko bars, bowl)."""
    import mitsuba as mi
    for v in (args.variant, 'cuda_ad_rgb', 'llvm_ad_rgb', 'scalar_rgb'):
        if v is None:
            continue
        try:
            mi.set_variant(v)
            print(f'[yarn_render] variant {v}')
            break
        except Exception:
            continue

    # bbox over all ply points (fibers live within), unless one was supplied
    # for stable cross-frame framing.
    if fixed_bbox is not None:
        lo, hi = np.asarray(fixed_bbox[0]), np.asarray(fixed_bbox[1])
    else:
        allp = np.vstack([pl for plies in plies_by_curve for pl in plies])
        lo, hi = allp.min(0), allp.max(0)
    center = 0.5 * (lo + hi)
    diag = float(np.linalg.norm(hi - lo)) + 1e-6

    az = np.radians(args.azim); el = np.radians(args.elev)
    d = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az),
                  np.sin(el)])
    origin = center + d * diag * args.dist_scale
    up = [0, 0, 1] if abs(np.sin(el)) < 0.97 else [0, 1, 0]

    tmp = tempfile.mkdtemp(prefix='yarn_mi_')
    pal = Y._PLY_COLORS
    if getattr(args, 'pastel', False):
        # Lighten the default vibrant palette by mixing 50% white.
        pal = pal * 0.5 + 0.5
    # Parse --curve_colors → per-curve palette (overrides per-ply colour).
    curve_pal = None
    if getattr(args, 'curve_colors', None):
        def _parse_col(s: str):
            s = s.strip().lstrip('#')
            if (len(s) == 6
                    and all(c in '0123456789abcdefABCDEF' for c in s)):
                return [int(s[0:2], 16) / 255.0,
                        int(s[2:4], 16) / 255.0,
                        int(s[4:6], 16) / 255.0]
            parts = [p for p in s.replace(',', ' ').split() if p]
            if len(parts) == 3:
                return [float(p) for p in parts]
            raise ValueError(f"can't parse colour {s!r}")
        curve_pal = np.array([_parse_col(c) for c in args.curve_colors],
                             dtype=np.float64)
    base_col = [float(x) for x in args.color.split(',')] \
        if args.color else [0.80, 0.62, 0.40]

    def mk_bsdf(col):
        col = [float(x) for x in col]
        if args.bsdf == 'hair':
            # Mitsuba 3's real fiber BCSDF (Chiang hair model). Colour is
            # driven through sigma_a so any hue works (not just melanin).
            # The white primary highlight desaturates hair, so deepen the
            # target colour (col**hair_sat, hair_sat>1) for more contrast.
            deep = [c ** args.hair_sat for c in col]
            return {'type': 'hair',
                    'longitudinal_roughness': args.long_rough,
                    'azimuthal_roughness': args.azim_rough,
                    'scale_tilt': args.scale_tilt,
                    'sigma_a': {'type': 'rgb',
                                'value': sigma_a_from_color(deep, args.azim_rough)}}
        if args.bsdf == 'dielectric':
            # a fibre is physically a translucent dielectric cylinder
            return {'type': 'roughdielectric', 'distribution': 'ggx',
                    'alpha': max(args.roughness * 0.4, 0.02),
                    'int_ior': args.eta, 'ext_ior': 1.0,
                    'specular_transmittance': {'type': 'rgb', 'value': col}}
        if args.bsdf == 'fiber':
            # principledthin with diffuse transmission + sheen: a translucent,
            # soft fiber look (a practical stand-in for a Marschner/Chiang
            # hair BCSDF, and it runs on the GPU -- see note in main()).
            return {'type': 'principledthin',
                    'base_color': {'type': 'rgb', 'value': col},
                    'roughness': args.roughness, 'diff_trans': args.diff_trans,
                    'spec_trans': 0.0, 'sheen': 0.6, 'flatness': 0.4}
        if args.bsdf == 'principled':
            return {'type': 'principled',
                    'base_color': {'type': 'rgb', 'value': col},
                    'roughness': args.roughness, 'sheen': 0.5,
                    'specular': 0.4}
        return {'type': 'roughplastic', 'distribution': 'ggx',
                'alpha': args.roughness,
                'diffuse_reflectance': {'type': 'rgb', 'value': col}}

    shapes = {}
    have_fibers = any(fb is not None for fb in fibers_by_curve)
    all_hairs = [hr for hairs in flyaways_by_curve for hr in hairs]
    fly_r = args.fly_thickness

    if args.rainbow:
        # Colour follows the path -> split every curve into arclength hue
        # bands (a fiber spans the whole strand, so per-shape colour alone
        # can't gradient along it). Pool all geometry, then one shape/band.
        bands = max(int(args.rainbow_bands), 2)
        fiber_pool, tube_pool = [], []
        for c, plies in enumerate(plies_by_curve):
            fibs = fibers_by_curve[c]
            for p, pl in enumerate(plies):
                if have_fibers and fibs is not None:
                    fiber_pool.extend(fibs[p])
                else:
                    tube_pool.append(pl)
        for b in range(bands):
            bsdf = mk_bsdf(Y.rainbow_rgb((b + 0.5) / bands))
            if fiber_pool:
                fpath = os.path.join(tmp, f'band_{b}.txt')
                if write_curve_file(fpath,
                                    [band_slice(fc, b, bands) for fc in fiber_pool],
                                    args.fiber_thickness):
                    shapes[f'band_{b}'] = {'type': 'linearcurve',
                                           'filename': fpath, 'bsdf': bsdf}
            if tube_pool:
                opath = os.path.join(tmp, f'band_{b}.obj')
                write_obj(opath, [band_slice(pl, b, bands) for pl in tube_pool],
                          args.tube_radius, args.nseg)
                shapes[f'tband_{b}'] = {'type': 'obj', 'filename': opath,
                                        'bsdf': bsdf}
            # flyaways whose root falls in this band get this hue
            bh = [pts for pts, rf in all_hairs
                  if min(int(rf * bands), bands - 1) == b]
            if bh:
                hpath = os.path.join(tmp, f'fly_{b}.txt')
                if write_curve_file(hpath, bh, fly_r):
                    shapes[f'fly_{b}'] = {'type': 'linearcurve',
                                          'filename': hpath, 'bsdf': bsdf}
    else:
        for c, plies in enumerate(plies_by_curve):
            fibs = fibers_by_curve[c]
            for p, pl in enumerate(plies):
                if curve_pal is not None:
                    col = list(curve_pal[c % len(curve_pal)])
                elif args.color_plies:
                    col = list(pal[p % len(pal)])
                else:
                    col = base_col
                bsdf = mk_bsdf(col)
                if have_fibers and fibs is not None:
                    fpath = os.path.join(tmp, f'fib_{c}_{p}.txt')
                    write_curve_file(fpath, fibs[p], args.fiber_thickness)
                    shapes[f'fib_{c}_{p}'] = {'type': 'linearcurve',
                                              'filename': fpath, 'bsdf': bsdf}
                else:
                    opath = os.path.join(tmp, f'ply_{c}_{p}.obj')
                    write_obj(opath, [pl], args.tube_radius, args.nseg)
                    shapes[f'ply_{c}_{p}'] = {'type': 'obj',
                                              'filename': opath, 'bsdf': bsdf}
        if all_hairs:
            # Three colouring paths for flyaways:
            #   curve_pal set   → one shape per CURVE, curve's colour
            #   per-ply tagged  → one shape per PLY, ply's colour
            #   default         → one cream shape for everything
            sample = all_hairs[0]
            per_ply_tagged = (args.color_plies and len(sample) == 3)
            if curve_pal is not None:
                for c, hairs in enumerate(flyaways_by_curve):
                    if not hairs:
                        continue
                    col = list(curve_pal[c % len(curve_pal)])
                    bsdf_c = mk_bsdf(col)
                    pts_list = [h[0] for h in hairs]
                    hpath = os.path.join(tmp, f'fly_c{c}.txt')
                    if write_curve_file(hpath, pts_list, fly_r):
                        shapes[f'fly_c{c}'] = {
                            'type': 'linearcurve',
                            'filename': hpath,
                            'bsdf': bsdf_c,
                        }
            elif per_ply_tagged:
                from collections import defaultdict
                buckets: dict[int, list] = defaultdict(list)
                for item in all_hairs:
                    pts, _rf, pi = item
                    buckets[int(pi)].append(pts)
                for pi, pts_list in buckets.items():
                    col = list(pal[pi % len(pal)])
                    bsdf = mk_bsdf(col)
                    hpath = os.path.join(tmp, f'fly_p{pi}.txt')
                    if write_curve_file(hpath, pts_list, fly_r):
                        shapes[f'fly_p{pi}'] = {
                            'type': 'linearcurve',
                            'filename': hpath,
                            'bsdf': bsdf,
                        }
            else:
                hpath = os.path.join(tmp, 'flyaways.txt')
                if write_curve_file(hpath,
                                    [it[0] for it in all_hairs], fly_r):
                    shapes['flyaways'] = {
                        'type': 'linearcurve', 'filename': hpath,
                        'bsdf': mk_bsdf(base_col),   # fuzz follows the yarn colour
                    }

    scene = {
        'type': 'scene',
        'integrator': {'type': 'path', 'max_depth': args.max_depth},
        'sensor': {
            'type': 'perspective', 'fov': args.fov,
            'to_world': mi.ScalarTransform4f().look_at(
                origin=list(origin), target=list(center), up=up),
            'film': {'type': 'hdrfilm', 'width': args.width,
                     'height': args.height,
                     'rfilter': {'type': 'gaussian'}},
            'sampler': {'type': 'independent', 'sample_count': args.spp},
        },
    }
    # Lighting: HDR environment map (custom file, else a procedural studio
    # HDR) gives directional light + graded background + reflections; or a
    # flat constant emitter with --no_env.
    if args.no_env:
        scene['env'] = {'type': 'constant', 'radiance': args.bg}
    else:
        epath = args.envmap or synth_envmap(os.path.join(tmp, 'env.exr'),
                                            intensity=args.bg)
        scene['env'] = {'type': 'envmap', 'filename': epath,
                        'scale': 1.0 if args.envmap else 1.0}
    if args.key > 0:
        # key light: a point emitter high & front; intensity ~ r^2 so the
        # irradiance it delivers is roughly `key` regardless of scene scale.
        kaz = np.radians(args.azim + 25.0); kel = np.radians(65.0)
        kd = np.array([np.cos(kel) * np.cos(kaz), np.cos(kel) * np.sin(kaz),
                       np.sin(kel)])
        kr = diag * 1.3
        kpos = center + kd * kr
        scene['key'] = {'type': 'point', 'position': list(kpos),
                        'intensity': {'type': 'rgb',
                                      'value': [args.key * kr * kr] * 3}}
    if args.backdrop:
        ftex = getattr(args, 'floor_texture', None)
        if ftex:
            tile = float(getattr(args, 'floor_tile', 8.0))
            floor_bsdf = {
                'type': 'diffuse',
                'reflectance': {
                    'type': 'bitmap',
                    'filename': str(ftex),
                    'wrap_mode': 'repeat',
                    'filter_type': 'bilinear',
                    'to_uv': mi.ScalarTransform4f().scale(
                        [tile, tile, 1.0]),
                },
            }
        else:
            floor_bsdf = {
                'type': 'diffuse',
                'reflectance': {'type': 'rgb', 'value': [0.85, 0.85, 0.85]},
            }
        scene['floor'] = {
            'type': 'rectangle',
            'to_world': mi.ScalarTransform4f().translate(
                [center[0], center[1], lo[2] - 0.05 * diag]).scale(diag * 1.5),
            'bsdf': floor_bsdf,
        }
    scene.update(shapes)
    if extra_shapes:
        scene.update(extra_shapes)

    n_sh = len(shapes)
    pp = int(getattr(args, 'spp_per_pass', 0) or 0)
    print(f'[yarn_render] {n_sh} shape(s), '
          f'{"fibers" if have_fibers else "ply tubes"}, '
          f'{args.width}x{args.height} @ {args.spp} spp'
          + (f' ({(args.spp + pp - 1)//pp} passes × {pp} spp)' if pp > 0
             else ''))
    t0 = time.time()
    sc = mi.load_dict(scene)
    if pp > 0 and pp < args.spp:
        import drjit as dr
        n_passes = (args.spp + pp - 1) // pp
        accum, done = None, 0
        for i in range(n_passes):
            s = min(pp, args.spp - done)
            img_i = mi.render(sc, spp=s, seed=i)
            contrib = mi.TensorXf(img_i) * (s / args.spp)
            accum = contrib if accum is None else accum + contrib
            dr.eval(accum)
            dr.flush_malloc_cache()
            done += s
        img = accum
    else:
        img = mi.render(sc)
    out = args.out or f'yarn_render_{time.strftime("%Y%m%d_%H%M%S")}.png'
    mi.util.write_bitmap(out, img)
    print(f'[yarn_render] wrote {out} in {time.time() - t0:.1f}s')
    return out


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('npz', help='centerline npz or yarn_plies_v1 npz')
    # geometry (used only when generating from a centerline)
    ap.add_argument('--n_plies', type=int, default=8)
    ap.add_argument('--pitch', type=float, default=300.0)
    ap.add_argument('--radius', type=float, default=16.0)
    ap.add_argument('--tube_radius', type=float, default=9.0)
    ap.add_argument('--core', action='store_true', default=True,
                    help='add a center ply filling the hollow core')
    ap.add_argument('--no_core', dest='core', action='store_false')
    ap.add_argument('--nseg', type=int, default=12)
    ap.add_argument('--sign', type=float, default=1.0)
    ap.add_argument('--phase0', type=float, default=0.0)
    ap.add_argument('--fibers', action='store_true')
    ap.add_argument('--fibers_per_ply', type=int, default=20)
    ap.add_argument('--fiber_pitch', type=float, default=80.0)
    ap.add_argument('--bundle_radius', type=float, default=6.0)
    ap.add_argument('--migration', type=float, default=0.3)
    ap.add_argument('--perlin_amt', type=float, default=2.0)
    ap.add_argument('--perlin_scale', type=float, default=0.03)
    ap.add_argument('--plies_only', action='store_true',
                    help='render ply tubes even if fibers are available')
    # flyaway hairs (fuzz)
    ap.add_argument('--fly', action='store_true', default=True,
                    help='add flyaway hairs (fuzz)')
    ap.add_argument('--no_fly', dest='fly', action='store_false')
    ap.add_argument('--fly_n', type=int, default=2500)
    ap.add_argument('--fly_len', type=float, default=45.0)
    ap.add_argument('--fly_step', type=float, default=5.0)
    ap.add_argument('--fly_amt', type=float, default=1.2)
    ap.add_argument('--fly_outward', type=float, default=0.7)
    ap.add_argument('--fly_thickness', type=float, default=0.28,
                    help='radius of each rendered flyaway hair (voxels)')
    # render
    ap.add_argument('--fiber_thickness', type=float, default=0.7,
                    help='radius of each rendered fiber (voxels)')
    ap.add_argument('--spp', type=int, default=128)
    ap.add_argument('--spp_per_pass', type=int, default=0,
                    help='Split spp into multiple passes (CUDA OOM '
                         'workaround for high-res renders). 0 = single pass.')
    ap.add_argument('--width', type=int, default=1000)
    ap.add_argument('--height', type=int, default=750)
    ap.add_argument('--max_depth', type=int, default=12)
    ap.add_argument('--fov', type=float, default=32.0)
    ap.add_argument('--elev', type=float, default=35.0, help='camera elevation deg')
    ap.add_argument('--azim', type=float, default=-60.0, help='camera azimuth deg')
    ap.add_argument('--dist_scale', type=float, default=1.4)
    ap.add_argument('--bg', type=float, default=0.8,
                    help='environment brightness')
    ap.add_argument('--envmap', type=str, default='',
                    help='HDR (.exr/.hdr) environment map; default = built-in '
                         'procedural studio HDR')
    ap.add_argument('--no_env', action='store_true',
                    help='use a flat constant emitter instead of an HDR env')
    ap.add_argument('--key', type=float, default=0.0,
                    help='extra key point-light strength (env usually enough)')
    ap.add_argument('--bsdf',
                    choices=['hair', 'dielectric', 'fiber', 'principled',
                             'roughplastic'],
                    default='hair',
                    help='surface model. "hair" = Mitsuba 3 real Chiang fiber '
                         'BCSDF; "dielectric" = translucent cylinder; "fiber" '
                         '= principledthin stand-in')
    ap.add_argument('--roughness', type=float, default=0.35)
    ap.add_argument('--diff_trans', type=float, default=0.5,
                    help='diffuse transmission for the fiber BSDF (0..2)')
    # hair BCSDF params
    ap.add_argument('--long_rough', type=float, default=0.6,
                    help='hair longitudinal roughness (high = matte, woolly; '
                         'low = silky/glossy like hair)')
    ap.add_argument('--azim_rough', type=float, default=0.4,
                    help='hair azimuthal roughness (smaller = more saturated)')
    ap.add_argument('--scale_tilt', type=float, default=2.0,
                    help='hair cuticle scale tilt (deg)')
    ap.add_argument('--hair_sat', type=float, default=2.2,
                    help='hair colour deepening (>1 = richer/more contrast)')
    ap.add_argument('--eta', type=float, default=1.55, help='fiber IOR')
    ap.add_argument('--backdrop', action='store_true', default=True)
    ap.add_argument('--no_backdrop', dest='backdrop', action='store_false')
    ap.add_argument('--floor_texture', type=str, default='',
                    help='Path to a wood/floor texture image (jpg/png/exr); '
                         'overrides the default plain grey floor.')
    ap.add_argument('--floor_tile', type=float, default=8.0,
                    help='UV repeat count across the floor rectangle '
                         '(default 8: texture tiles 8x8 over the floor '
                         'so it stays sharp).')
    ap.add_argument('--color', type=str, default='',
                    help='single yarn colour "r,g,b" (0..1)')
    ap.add_argument('--pastel', action='store_true',
                    help='Lighten the --color_plies palette (50%% white mix) '
                         'for a softer pastel look.')
    ap.add_argument('--curve_colors', nargs='+', default=None,
                    metavar='HEX_OR_RGB',
                    help='Per-curve colours: hex (#RRGGBB) or "r,g,b" 0..1. '
                         'One entry per centerline in the input npz; cycles '
                         'if fewer entries than curves. Overrides '
                         '--color_plies for the curve coloring.')
    ap.add_argument('--color_plies', action='store_true',
                    help='colour each ply distinctly')
    ap.add_argument('--rainbow', action='store_true',
                    help='rainbow colour following the yarn path')
    ap.add_argument('--rainbow_bands', type=int, default=48,
                    help='hue bands along the path (more = smoother gradient)')
    ap.add_argument('--variant', type=str, default=None)
    ap.add_argument('--out', type=str, default='')
    args = ap.parse_args()

    plies_by_curve, fibers_by_curve, flyaways_by_curve = build_geometry(args)
    render(plies_by_curve, fibers_by_curve, flyaways_by_curve, args)


if __name__ == '__main__':
    main()
