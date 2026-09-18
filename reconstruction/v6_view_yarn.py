#!/usr/bin/env python3
"""Standalone viewer for saved crochet yarn.

USAGE — pass an input STEM (the npz file's basename without extension)
and the viewer auto-finds the yarn under output/<stem>/curves/:

    python v6_view_yarn.py G1_pointcloud_full_energy_64000_linearity_0.03_ds3.5
    python v6_view_yarn.py G1_..._ds3.5 --possibilities      # all-combos viewer
    python v6_view_yarn.py /absolute/path/to/yarn.npz        # back-compat

Auto-find priority (single yarn):
    1. output/<stem>/curves/yarn_latest.npz
    2. newest output/<stem>/curves/yarn_*.npz

With --possibilities:
    newest output/<stem>/curves/possibilities_*.npz

If the arg ends with `.npz` or contains `/`, it is taken as a direct path.

Two input formats (auto-detected by file content):

  • SINGLE yarn  (`yarn_*.npz` from [save-yarn], key "yarn" = an ordered
    (N,3) polyline): rendered as ONE tube coloured in a rainbow by
    progression (blue = start → red = end = travel direction), with
    green/red start/end balls and a toggleable heading-arrow field.

  • POSSIBILITIES (`possibilities_*.npz` from "Export all possibilities",
    `fmt='yarn_possibilities_v1'`): every loop-free connection combination,
    each a set of curves.  Rendered one at a time, each curve a distinct
    colour; use ← / → to cycle possibilities (fewer curves = better joined).
"""
import argparse
import colorsys
import glob
import os
import sys

import numpy as np
import polyscope as ps
import polyscope.imgui as psim


def _resolve_yarn_path(stem_or_path: str, want_possibilities: bool) -> str:
    """Map an input STEM (e.g. 'G1_..._ds3.5') to its npz path under
    output/<stem>/curves/.  If the input already looks like a file
    path (contains '/' or ends with '.npz'), pass it through.  Raises
    SystemExit with a helpful message if nothing matches."""
    # ── direct-path back-compat ──────────────────────────────────────
    if stem_or_path.endswith('.npz') or os.sep in stem_or_path:
        if os.path.exists(stem_or_path):
            return stem_or_path
        raise SystemExit(f'  [view] file not found: {stem_or_path}')
    # ── stem-name resolution ─────────────────────────────────────────
    folder = os.path.join('output', stem_or_path, 'curves')
    if not os.path.isdir(folder):
        raise SystemExit(
            f'  [view] no folder {folder}/ — did v5 run on this input?\n'
            f'  Expected layout: output/<stem>/curves/yarn_latest.npz')
    if want_possibilities:
        cand = sorted(glob.glob(os.path.join(
            folder, 'possibilities_*.npz')))
        if not cand:
            raise SystemExit(
                f'  [view] no possibilities_*.npz in {folder}/ — '
                f'run "Export all possibilities" in v5 first')
        return cand[-1]
    # single yarn: latest, then newest timestamped
    latest = os.path.join(folder, 'yarn_latest.npz')
    if os.path.exists(latest):
        return latest
    cand = sorted(glob.glob(os.path.join(folder, 'yarn_*.npz')))
    if not cand:
        raise SystemExit(
            f'  [view] no yarn_latest.npz / yarn_*.npz in {folder}/ — '
            f'run "Save yarn" or "Fit curves" in v5 first')
    return cand[-1]


def rainbow_rgb(t: np.ndarray) -> np.ndarray:
    """t in [0,1] -> (N,3) rainbow RGB: blue (start) → … → red (end)."""
    h = (1.0 - np.clip(t, 0.0, 1.0)) * 0.70      # hue 0.70(blue)..0.0(red)
    return np.array([colorsys.hsv_to_rgb(float(x), 1.0, 1.0) for x in h],
                    np.float64)


def _auto_radius(pts_list, override: float):
    """(radius, bbox_diag) — radius = override if >0, else ~0.3% of the diag
    over all given point arrays."""
    allp = np.vstack([np.asarray(p, float) for p in pts_list if len(p) >= 1])
    diag = float(np.linalg.norm(allp.max(0) - allp.min(0)))
    return (override if override > 0 else max(diag * 0.003, 1e-6)), diag


# ───────────────────────── knot determinant ────────────────────────────────
def _xsect2(a, b, c, d):
    """2D segment ab × cd proper interior crossing → (s, t) params, else None."""
    r = b - a; s = d - c
    den = r[0] * s[1] - r[1] * s[0]
    if abs(den) < 1e-12:
        return None
    diff = c - a
    u = (diff[0] * s[1] - diff[1] * s[0]) / den
    v = (diff[0] * r[1] - diff[1] * r[0]) / den
    if 1e-9 < u < 1 - 1e-9 and 1e-9 < v < 1 - 1e-9:
        return u, v
    return None


def knot_determinant(P3, n_max: int = 10_000000):
    """Knot determinant of the closed single strand `P3` — a true knot
    invariant (unknot=1, trefoil=3, fig-8=5; det≠1 ⟹ definitely knotted).
    Closes the strand, projects to its best-fit plane (PCA), finds 2D
    self-crossings (over/under from depth along the plane normal), builds the
    Fox-colouring matrix and returns (det | None, log10|det|, n_crossings):
    `det` is the exact integer when small enough (else None for a huge weave);
    `log10|det|` (via the numerically-STABLE slogdet) is what to RANK by —
    the smallest across the possibilities is the least-knotted = likely the
    correct weave.  Resampled to ≤ n_max points (arc length) to bound cost."""
    import bisect
    P3 = np.asarray(P3, float)
    if len(P3) < 4:
        return 1, 0.0, 0
    if np.linalg.norm(P3[0] - P3[-1]) > 1e-9:
        P3 = np.vstack([P3, P3[0]])                 # close the loop
    seg = np.linalg.norm(np.diff(P3, axis=0), axis=1)
    al = np.concatenate([[0.0], np.cumsum(seg)]); L = float(al[-1])
    if L <= 0:
        return 1, 0.0, 0
    N = int(min(len(P3) - 1, n_max))
    ts = np.linspace(0.0, L, N, endpoint=False)     # N distinct, treat closed
    j = np.clip(np.searchsorted(al, ts, 'right') - 1, 0, len(P3) - 2)
    f = (ts - al[j]) / np.maximum(al[j + 1] - al[j], 1e-12)
    P = P3[j] + f[:, None] * (P3[j + 1] - P3[j])
    c0 = P.mean(0)
    _, _, Vt = np.linalg.svd(P - c0, full_matrices=False)
    P2 = np.c_[(P - c0) @ Vt[0], (P - c0) @ Vt[1]]
    depth = (P - c0) @ Vt[2]
    nxt = np.r_[1:N, 0]
    mids = 0.5 * (P2 + P2[nxt])
    sl = np.linalg.norm(P2[nxt] - P2, axis=1)
    r = 2.0 * float(sl.max()) if N else 0.0
    try:
        from scipy.spatial import cKDTree
        cand = cKDTree(mids).query_pairs(r)
    except Exception:
        cand = ((k, l) for k in range(N) for l in range(k + 1, N))
    cross = []
    for k, l in cand:
        if (l - k) % N <= 1 or (k - l) % N <= 1:    # adjacent segments
            continue
        hit = _xsect2(P2[k], P2[nxt[k]], P2[l], P2[nxt[l]])
        if hit:
            s, t = hit
            dk = depth[k] + s * (depth[nxt[k]] - depth[k])
            dl = depth[l] + t * (depth[nxt[l]] - depth[l])
            uk, ul = k + s, l + t
            cross.append((uk, ul) if dk >= dl else (ul, uk))  # (over, under)
    nc = len(cross)
    if nc == 0:
        return 1, 0.0, 0
    U = sorted(c[1] for c in cross)
    A = np.zeros((nc, nc))
    for row, (ov, un) in enumerate(cross):
        oa = (bisect.bisect_right(U, ov) - 1) % nc
        m = bisect.bisect_left(U, un)
        A[row, oa] += 2; A[row, (m - 1) % nc] -= 1; A[row, m % nc] -= 1
    _sign, lad = np.linalg.slogdet(A[:-1, :-1])      # stable log|det|
    if not np.isfinite(lad):
        return 0, 0.0, nc                            # singular → |det|=0
    log10 = float(lad / np.log(10.0))
    det = int(round(float(np.exp(lad)))) if lad < 25.0 else None
    return det, log10, nc


# ───────────────────────── single yarn (unchanged behaviour) ───────────────
def _yarn_polyline(d, key: str) -> np.ndarray:
    if key in d.files:
        P = np.asarray(d[key], np.float64)
    else:
        # Only `curve_<int>` are per-curve arrays.  `curve_K`,
        # `curve_smooth`, `curve_seg_labels` etc. are scalar metadata
        # — `len()` on a 0-d ndarray throws TypeError, so filter them
        # out before picking the longest.
        cand = [k for k in d.files
                if k.startswith('curve_')
                and k[len('curve_'):].isdigit()]
        if not cand:
            sys.exit(f"'{key}' not in npz; keys = {list(d.files)}")
        key2 = max(cand, key=lambda k: len(d[k]))
        print(f"  ['{key}' absent — not a single strand; showing the "
              f"longest curve '{key2}']")
        P = np.asarray(d[key2], np.float64)
    if P.ndim != 2 or P.shape[1] != 3 or len(P) < 2:
        sys.exit(f"expected an (N>=2, 3) array, got shape {P.shape}")
    return P


def _palette_rgb(n: int, sat: float = 0.78,
                 val: float = 0.95) -> np.ndarray:
    """`n` visually-distinct colours via golden-ratio HSV stepping.
    Adjacent indices land far apart on the colour wheel — same
    technique v5 uses for curve segments."""
    phi = 0.61803398875
    out = np.empty((n, 3), dtype=np.float64)
    for i in range(n):
        out[i] = colorsys.hsv_to_rgb((i * phi) % 1.0, sat, val)
    return out


def run_all_curves(d, args) -> None:
    """Render every `curve_<sid>` array as its own polyscope
    curve_network, each with a distinct golden-ratio HSV colour.
    Use this for multi-curve outputs (baseline pipelines pre-merge,
    fragmented MSTs, etc.).  Sorted longest → shortest so the warmest
    palette entries land on the longest strands."""
    cand = [k for k in d.files
            if k.startswith('curve_') and k[len('curve_'):].isdigit()]
    if not cand:
        sys.exit(f"no `curve_<sid>` arrays in npz; "
                 f"keys = {list(d.files)}")
    curves: list = []
    for k in cand:
        sp = np.asarray(d[k], np.float64)
        if sp.ndim == 2 and sp.shape[1] == 3 and len(sp) >= 2:
            curves.append((k, sp))
    curves.sort(key=lambda kv: -float(np.linalg.norm(
        np.diff(kv[1], axis=0), axis=1).sum()))

    radius, diag = _auto_radius([sp for _, sp in curves], args.radius)
    total = sum(float(np.linalg.norm(np.diff(sp, axis=0), axis=1).sum())
                for _, sp in curves)
    max_L = float(np.linalg.norm(np.diff(curves[0][1], axis=0),
                                  axis=1).sum())
    print(f"  {len(curves)} curves  total {total:.0f} vox  "
          f"(longest {max_L:.0f}; bbox diag {diag:.0f})")
    print(f"  tube radius {radius:.2f} vox  "
          f"(palette: golden-ratio HSV — adjacent curves never share a hue)")

    pal = _palette_rgb(len(curves))
    for i, (k, sp) in enumerate(curves):
        N = len(sp)
        edges = np.stack([np.arange(N - 1), np.arange(1, N)], axis=1)
        cn = ps.register_curve_network(k, sp, edges)
        cn.set_radius(radius, relative=False)
        cn.set_color(tuple(float(x) for x in pal[i]))


def run_single_yarn(d, args) -> None:
    P = _yarn_polyline(d, args.key)
    N = len(P)
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    length = float(seg.sum())
    radius, diag = _auto_radius([P], args.radius)

    al = np.concatenate([[0.0], np.cumsum(seg)])
    t = al / (al[-1] if al[-1] > 0 else 1.0)
    cols = (np.tile(np.asarray(args.color), (N, 1))
            if args.color is not None else rainbow_rgb(t))
    tang = np.empty_like(P)
    tang[:-1] = np.diff(P, axis=0)
    tang[-1] = tang[-2]
    tang /= (np.linalg.norm(tang, axis=1, keepdims=True) + 1e-12)

    tag = ('solid colour ' + ','.join(f'{x:.3f}' for x in args.color)
           if args.color is not None else 'rainbow blue→red = start→end')
    print(f"  yarn: {N} pts, length {length:.0f} vox, bbox diag {diag:.0f} vox")
    print(f"  tube radius {radius:.2f} vox  ({tag})")

    edges = np.stack([np.arange(N - 1), np.arange(1, N)], axis=1)
    cn = ps.register_curve_network('yarn', P, edges)
    cn.set_radius(radius, relative=False)
    cn.add_color_quantity('progress (blue=start → red=end)', cols,
                          enabled=True)
    try:
        cn.add_node_vector_quantity('heading', tang * (radius * 4.0),
                                    enabled=False, vectortype='ambient',
                                    color=(0.95, 0.95, 0.95))
    except Exception:
        try:
            cn.add_vector_quantity('heading', tang * (radius * 4.0),
                                   enabled=False, vectortype='ambient',
                                   color=(0.95, 0.95, 0.95))
        except Exception as e:
            print(f"  [heading arrows skipped: {e}]")
    ends = ps.register_point_cloud('start (green) / end (red)',
                                   np.vstack([P[0], P[-1]]))
    ends.set_radius(radius * 3.0, relative=False)
    ends.add_color_quantity('start/end',
                            np.array([[0.1, 0.9, 0.2], [0.9, 0.1, 0.1]]),
                            enabled=True)


# ───────────────────────── possibilities (new) ─────────────────────────────
def load_possibilities(d):
    """If `d` is a `yarn_possibilities_v1` npz → (possibilities, combos):
    possibilities[p] = list of (M,3) curve arrays, combos[p] = way-per-region
    indices.  Else None (so the single-yarn path is used)."""
    files = set(d.files)
    fmt = d['fmt'].item() if 'fmt' in files else ''
    if 'n_poss' not in files and fmt != 'yarn_possibilities_v1':
        return None
    n = int(d['n_poss'])
    poss, combos = [], []
    for p in range(n):
        nc = int(d[f'p{p}_n'])
        poss.append([np.asarray(d[f'p{p}_c{c}'], np.float64)
                     for c in range(nc) if f'p{p}_c{c}' in files])
        combos.append([int(x) for x in d[f'p{p}_combo']]
                      if f'p{p}_combo' in files else [])
    rc = (np.asarray(d['region_center'], np.float64)
          if 'region_center' in files else None)
    return poss, combos, rc


def save_yarn_npz(curves, path):
    """Write `curves` (list of (M,3)) as a yarn npz in toy_chain_spring.py's
    [save-yarn] format: `curve_<i>` per curve + `sub_ids` + `n_curves` +
    `fmt='yarn_state_v1'`, and (when it's ONE strand) the `yarn` key.  Loadable
    by both this viewer and the toy's `_load_yarn`."""
    out = {f'curve_{i}': np.asarray(sp, np.float64)
           for i, sp in enumerate(curves)}
    out['sub_ids'] = np.asarray(list(range(len(curves))), np.int64)
    out['n_curves'] = np.asarray(len(curves))
    out['fmt'] = np.asarray('yarn_state_v1')
    if len(curves) == 1:
        out['yarn'] = np.asarray(curves[0], np.float64)
    np.savez(path, **out)


_PVIEW = {'p': 0}                       # current possibility index (for ←/→)


def show_possibility(poss, combos, p: int, radius: float) -> None:
    p %= len(poss)
    _PVIEW['p'] = p
    curves = poss[p]
    # RAINBOW by arc-length progression along EACH curve (blue start → red
    # end = travel direction); single-strand possibilities → one clean rainbow.
    nodes, edges, cols, ends, endcols = [], [], [], [], []
    for sp in curves:
        sp = np.asarray(sp, np.float64)
        if len(sp) < 2:
            continue
        i0 = len(nodes)
        nodes.extend(sp.tolist())
        edges.extend([[i0 + t, i0 + t + 1] for t in range(len(sp) - 1)])
        seg = np.linalg.norm(np.diff(sp, axis=0), axis=1)
        al = np.concatenate([[0.0], np.cumsum(seg)])
        t = al / (al[-1] if al[-1] > 0 else 1.0)
        solid = _PVIEW.get('color')
        cols.extend((np.tile(np.asarray(solid), (len(sp), 1)).tolist()
                     if solid is not None else rainbow_rgb(t).tolist()))
        ends.append(sp[0].tolist());  endcols.append([0.1, 0.9, 0.2])  # green
        ends.append(sp[-1].tolist()); endcols.append([0.9, 0.1, 0.1])  # red
    if ps.has_curve_network('possibility'):
        ps.remove_curve_network('possibility')
    if ps.has_point_cloud('possibility_ends'):
        ps.remove_point_cloud('possibility_ends')
    if nodes:
        cn = ps.register_curve_network('possibility',
                                       np.asarray(nodes, np.float64),
                                       np.asarray(edges, np.int64))
        cn.set_radius(radius, relative=False)
        cn.add_color_quantity('progress (blue=start → red=end)',
                              np.asarray(cols, np.float64), enabled=True)
    if ends:
        pe = ps.register_point_cloud('possibility_ends',
                                     np.asarray(ends, np.float64))
        pe.set_radius(radius * 3.0, relative=False)
        pe.add_color_quantity('start (green) / end (red)',
                              np.asarray(endcols, np.float64), enabled=True)
    combo = combos[p] if p < len(combos) else []
    # mark WHERE this possibility differs from the min-complexity one: the
    # regions whose way-choice differs are the crossings that ADD knotting.
    if ps.has_point_cloud('diff_vs_min'):
        ps.remove_point_cloud('diff_vs_min')
    rc = _PVIEW.get('region_center')
    mp = _PVIEW.get('min_p')
    diff = []
    if rc is not None and mp is not None and mp != p:
        mc = combos[mp]
        diff = [r for r in range(min(len(combo), len(mc), len(rc)))
                if combo[r] != mc[r]]
        if diff:
            dc = ps.register_point_cloud(
                'diff_vs_min', np.asarray([rc[r] for r in diff], np.float64))
            dc.set_radius(radius * 4.0, relative=False)
            dc.set_color((1.0, 0.1, 0.1))        # red = differs from best here
            dc.set_enabled(True)
    if mp is None:
        dtag = ''
    elif rc is None:
        dtag = '  (re-export possibilities npz to mark diff regions)'
    elif mp == p:
        dtag = '  [= the min itself]'
    else:
        dtag = (f'  differs from min(P{mp + 1}) at region(s) '
                f'{[r + 1 for r in diff]} → RED')
    print(f"  possibility {p + 1}/{len(poss)}: {len(curves)} curve(s), "
          f"combo (way per region) = {list(combo)}{dtag}   "
          f"(slider / ←→ to switch)")


def run_possibilities(poss, combos, region_center, args) -> None:
    _PVIEW['region_center'] = region_center
    _PVIEW['color'] = args.color
    radius, diag = _auto_radius([sp for cs in poss for sp in cs], args.radius)
    counts = sorted({len(cs) for cs in poss})
    print(f"  {len(poss)} loop-free possibilities; bbox diag {diag:.0f} vox, "
          f"tube radius {radius:.2f} vox")
    print(f"  curve-counts across possibilities: {counts}  (fewest = best "
          f"joined).  Use the 'possibility' slider or ← / → to cycle.")
    n = len(poss)

    cache = {}                                  # p → (det, log10, nc)

    def _det(p):
        if p not in cache:
            cache[p] = knot_determinant(poss[p][0])
        return cache[p]

    def _ensure_min():
        """Index of the min-complexity (likely-correct) single-strand
        possibility; computes/caches dets if not done yet."""
        if _PVIEW.get('min_p') is None:
            best, best_log = None, float('inf')
            for p in range(n):
                if len(poss[p]) != 1:
                    continue
                _d, log10, _nc = _det(p)
                if log10 < best_log:
                    best_log, best = log10, p
            _PVIEW['min_p'] = best if best is not None else 0
        return _PVIEW['min_p']

    def _label(det, log10, nc):
        if nc == 0:
            return 'UNKNOT (0 crossings)'
        dval = str(det) if det is not None else f'~10^{log10:.1f}'
        return (f'{nc} crossings, det={dval}  → '
                f'{"UNKNOT (untangles)" if det == 1 else "KNOTTED (cannot)"}')

    def _cb():
        cur = _PVIEW['p']
        new = cur
        changed, val = psim.SliderInt('possibility', cur, 0, max(n - 1, 0))
        if changed:
            new = val
        if psim.IsKeyPressed(psim.ImGuiKey_LeftArrow, repeat=False):
            new = cur - 1
        elif psim.IsKeyPressed(psim.ImGuiKey_RightArrow, repeat=False):
            new = cur + 1
        if new % n != cur:
            show_possibility(poss, combos, new, radius)
            _PVIEW['det_text'] = ''                 # stale after switching
        # determinant of the CURRENT possibility (closes the loop)
        if psim.Button('Knot determinant (close loop)'):
            p = _PVIEW['p']
            if len(poss[p]) != 1:
                _PVIEW['det_text'] = (f'possibility {p + 1}: {len(poss[p])} '
                                      f'components (link) — single-strand only')
            else:
                _PVIEW['det_text'] = (f'possibility {p + 1}: '
                                      f'{_label(*_det(p))}')
            print('  [knot] ' + _PVIEW['det_text'])
        # rank ALL by knot complexity → the SMALLEST is the least-knotted,
        # i.e. the likely-correct weave (wrong over/under adds extra tangling).
        if psim.Button('Rank all by knot complexity (min = likely correct)'):
            rows = []
            for p in range(n):
                if len(poss[p]) != 1:
                    rows.append((float('inf'), p, None))   # link → push last
                else:
                    det, log10, nc = _det(p)
                    rows.append((log10, p, (det, log10, nc)))
            rows.sort(key=lambda r: r[0])
            # Δlog = log10(det) − log10(det_min): cancels the huge COMMON
            # fabric background, leaving "orders of magnitude of EXTRA knotting
            # vs the simplest weave" (best = 0.0).
            base = next((r[0] for r in rows if np.isfinite(r[0])), 0.0)
            print('  [knot] possibilities ranked by complexity '
                  '(Δlog = extra knotting vs the simplest; 0 = best):')
            for rank, (log10, p, info) in enumerate(rows):
                desc = (f'{len(poss[p])} comps (link)' if info is None
                        else _label(*info))
                dl = ('Δlog=+inf' if not np.isfinite(log10)
                      else f'Δlog=+{log10 - base:.1f}')
                tag = '   ← LIKELY CORRECT' if rank == 0 else ''
                print(f'      #{rank + 1}  possibility {p + 1}: {dl}  '
                      f'[{desc}]{tag}')
            best = rows[0][1]
            _PVIEW['min_p'] = best              # enables diff-vs-min markers
            _PVIEW['det_text'] = (
                f'likely-correct = possibility {best + 1} (min complexity); '
                f'other possibilities now show RED balls at the regions where '
                f'they differ from it; ←→/slider to inspect')
            show_possibility(poss, combos, best, radius)
        # SAVE the likely-correct (min-complexity) possibility as a yarn npz
        # in toy_chain_spring.py's [save-yarn] format.
        if psim.Button('Save likely-correct yarn (min → toy yarn npz)'):
            import time
            mp = _ensure_min()
            fn = f'yarn_likely_{time.strftime("%Y%m%d_%H%M%S")}.npz'
            save_yarn_npz(poss[mp], fn)
            kind = ('SINGLE strand' if len(poss[mp]) == 1
                    else f'{len(poss[mp])} curves')
            _PVIEW['det_text'] = (f'saved likely-correct possibility {mp + 1} '
                                  f'({kind}) → {fn}  [toy yarn format]')
            print('  [save] ' + _PVIEW['det_text'])
        if _PVIEW.get('det_text'):
            psim.TextUnformatted(_PVIEW['det_text'])

    show_possibility(poss, combos, 0, radius)
    ps.set_user_callback(_cb)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stem',
                    help='input STEM (the npz basename without extension, '
                         'e.g. "G1_pointcloud_full_energy_64000_linearity_'
                         '0.03_ds3.5") — auto-resolves to output/<stem>/'
                         'curves/yarn_latest.npz.  Also accepts a direct '
                         '.npz path (back-compat).')
    ap.add_argument('--possibilities', action='store_true',
                    help='Look for the newest possibilities_*.npz under '
                         'output/<stem>/curves/ instead of yarn_latest.npz.')
    ap.add_argument('--key', default='yarn', help='single-yarn array key')
    ap.add_argument('--radius', type=float, default=0.0,
                    help='tube radius in vox (0 = auto, ~0.3%% of bbox diag)')
    ap.add_argument('--color', type=str, default='',
                    help='solid colour "r,g,b" (0..1 or 0..255); overrides '
                         'the default blue→red rainbow.')
    ap.add_argument('--longest-only', action='store_true',
                    help='When the npz has multiple curves but no '
                         'single `yarn` key, show ONLY the longest '
                         '(old behaviour).  Default shows ALL curves '
                         'with a rainbow palette.')
    args = ap.parse_args()
    if args.color:
        c = [float(x) for x in args.color.split(',')]
        if max(c) > 1.5:
            c = [x / 255.0 for x in c]
        args.color = tuple(float(x) for x in c)
    else:
        args.color = None

    args.npz = _resolve_yarn_path(args.stem, args.possibilities)
    print(f'  [view] loading {args.npz}')
    d = np.load(args.npz, allow_pickle=False)
    ps.init()
    ps.set_up_dir('z_up')                       # match toy_chain_spring.py
    ps.set_background_color((0.08, 0.08, 0.10))
    ps.set_ground_plane_mode('none')

    pd = load_possibilities(d)
    if pd is not None:
        run_possibilities(pd[0], pd[1], pd[2], args)
    elif args.key in d.files or args.longest_only:
        # single `yarn` key present OR user explicitly forced single
        # mode → use the classic blue→red rainbow tube.
        run_single_yarn(d, args)
    else:
        # multi-curve baseline output → show ALL with a per-curve palette.
        run_all_curves(d, args)
    ps.show()


if __name__ == '__main__':
    main()
