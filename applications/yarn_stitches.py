"""
Curve autocorrelation in curvature local space (yarn-npz edition).

Reads fitted curves and, for each curve, computes arclength-parameterised
curvature κ(s) and its normalised autocorrelation. Peaks in the
autocorrelation correspond to candidate periods of repeated stitches
(independent of 3D position/orientation).

This variant of curve_autocorr.py accepts BOTH input formats:
  • the original mean-shift export (`curve_pts` + `curve_offsets` +
    `curve_seg_ids`), and
  • the yarn npz written by toy_chain_spring.py's `_save_yarn` /
    view_yarn.py's `save_yarn_npz` (`fmt='yarn_state_v1'`: per-curve
    `curve_<sid>` arrays ordered by `sub_ids`, plus a single `yarn` key
    when there is only one strand), and
  • a `yarn_possibilities_v1` npz (loads one possibility, default 0,
    selected with --poss).
The format is auto-detected from the npz keys.
"""
from __future__ import annotations
import argparse
import colorsys
import os
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import polyscope as ps
import polyscope.imgui as psim


# ── Camera save/load + transparent screenshot ────────────────────────────────
# Same camera.txt path as meanshift_centers.py so the two viewers share views.
_HERE          = os.path.dirname(os.path.abspath(__file__))
_CAMERA_PATH   = os.path.join(_HERE, 'camera.txt')
_RENDER_DIR    = os.path.join(_HERE, 'renders')
_camera_msg    = ['']
_obj_export_msg = ['']
_pending_camera_load = [False]   # one-shot apply on first callback frame
_pending_stitch_load = [None]    # one-shot: '__latest__'/path to load 1st frame


def _save_camera_view():
    try:
        with open(_CAMERA_PATH, 'w') as f:
            f.write(ps.get_view_as_json())
        _camera_msg[0] = f'saved → {os.path.basename(_CAMERA_PATH)}'
        print(f'  Camera saved → {_CAMERA_PATH}')
    except Exception as e:
        _camera_msg[0] = f'save failed: {e}'
        print(f'  [warn] camera save failed: {e}')


def _load_camera_view():
    if not os.path.isfile(_CAMERA_PATH):
        _camera_msg[0] = f'not found: {os.path.basename(_CAMERA_PATH)}'
        print(f'  [warn] camera file not found: {_CAMERA_PATH}')
        return
    try:
        with open(_CAMERA_PATH, 'r') as f:
            ps.set_view_from_json(f.read())
        _camera_msg[0] = f'loaded ← {os.path.basename(_CAMERA_PATH)}'
        print(f'  Camera loaded ← {_CAMERA_PATH}')
    except Exception as e:
        _camera_msg[0] = f'load failed: {e}'
        print(f'  [warn] camera load failed: {e}')


def _postprocess_transparent(path, tol=4):
    """Polyscope 2.6.1's transparent_bg=True does not actually produce alpha-0
    pixels. Workaround: chroma-key out the background by sampling the (0,0)
    corner pixel and setting any near-match pixel's alpha to 0."""
    try:
        from PIL import Image
        img = Image.open(path).convert('RGBA')
        arr = np.array(img)
        bg  = arr[0, 0, :3].astype(np.int16)
        diff = np.abs(arr[..., :3].astype(np.int16) - bg).max(axis=-1)
        arr[..., 3] = np.where(diff <= tol, 0, 255).astype(np.uint8)
        Image.fromarray(arr, 'RGBA').save(path)
    except Exception as e:
        print(f'  [warn] transparent post-process failed: {e}')


def _save_transparent_screenshot():
    stamp = time.strftime('%Y%m%d_%H%M%S')
    path  = os.path.join(_RENDER_DIR, f'shot_{stamp}.png')
    try:
        os.makedirs(_RENDER_DIR, exist_ok=True)
        ps.screenshot(path, transparent_bg=True)
        _postprocess_transparent(path)
        print(f'  PNG saved → {path}')
    except Exception as e:
        print(f'  [warn] screenshot failed: {e}')


def load_curves_and_ids(npz_path: Path,
                        poss: int = 0) -> tuple[list[np.ndarray], np.ndarray]:
    """Load polyline curves from any supported npz, returning
    ``(curves, seg_ids)`` where curves is a list of (M,3) float arrays and
    seg_ids is a parallel int array (one id per curve).

    Auto-detects three layouts:
      1. mean-shift export: ``curve_pts`` + ``curve_offsets`` (+ optional
         ``curve_seg_ids``);
      2. yarn npz (``fmt='yarn_state_v1'``): per-curve ``curve_<sid>``
         arrays ordered by ``sub_ids``, or a bare single ``yarn`` strand;
      3. ``yarn_possibilities_v1``: loads possibility ``poss`` (default 0),
         whose curves are stored as ``p<poss>_c<c>``.
    """
    d = np.load(npz_path, allow_pickle=False)
    files = set(d.files)
    fmt = str(d['fmt'].item()) if 'fmt' in files else ''

    # 1. original mean-shift export ─────────────────────────────────────────
    if 'curve_pts' in files and 'curve_offsets' in files:
        pts  = np.asarray(d['curve_pts'], np.float64)
        offs = d['curve_offsets']
        curves = [pts[offs[i]:offs[i + 1]] for i in range(len(offs) - 1)]
        seg_ids = (np.asarray(d['curve_seg_ids'], np.int64)
                   if 'curve_seg_ids' in files
                   else np.arange(len(curves), dtype=np.int64))
        return curves, seg_ids

    # 3. possibilities npz → pick one possibility ───────────────────────────
    if fmt == 'yarn_possibilities_v1' or 'n_poss' in files:
        n_poss = int(d['n_poss'])
        p = int(np.clip(poss, 0, max(0, n_poss - 1)))
        nc = int(d[f'p{p}_n'])
        curves = [np.asarray(d[f'p{p}_c{c}'], np.float64)
                  for c in range(nc) if f'p{p}_c{c}' in files]
        print(f'  [possibilities npz: {n_poss} possibilities; '
              f'showing #{p + 1} with {len(curves)} curve(s) — pick with --poss]')
        return curves, np.arange(len(curves), dtype=np.int64)

    # 2. yarn npz (yarn_state_v1) or old single-strand export ───────────────
    if 'sub_ids' in files:
        pairs = [(int(sid), np.asarray(d[f'curve_{int(sid)}'], np.float64))
                 for sid in d['sub_ids'] if f'curve_{int(sid)}' in files]
    else:
        pairs = [(int(k.split('_', 1)[1]), np.asarray(d[k], np.float64))
                 for k in sorted(files)
                 if k.startswith('curve_') and k.split('_', 1)[1].isdigit()]
    if not pairs and 'yarn' in files:                 # bare single strand
        pairs = [(0, np.asarray(d['yarn'], np.float64))]
    if not pairs:
        raise SystemExit(
            f'{npz_path}: no recognisable curve arrays; keys = {sorted(files)}')
    curves  = [p for _, p in pairs]
    seg_ids = np.asarray([s for s, _ in pairs], np.int64)
    return curves, seg_ids


def load_curves(npz_path: Path) -> list[np.ndarray]:
    """Back-compat wrapper: return only the curve list (see
    ``load_curves_and_ids`` for the seg-id-aware loader)."""
    return load_curves_and_ids(npz_path)[0]


def export_curves_obj(curves: list[np.ndarray],
                      out_path: Path,
                      seg_ids: np.ndarray | None = None,
                      split: bool = False) -> list[Path]:
    """Write polyline curves as Wavefront OBJ.

    With split=False, all curves go into one file as separate groups.
    With split=True, each curve is written to `<stem>_seg<sid>.obj` next to
    out_path. Returns the list of files written.
    """
    out_path = Path(out_path)
    written: list[Path] = []

    def _sid(i: int) -> int:
        if seg_ids is not None and i < len(seg_ids):
            return int(seg_ids[i])
        return i

    if split:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        for i, c in enumerate(curves):
            sid = _sid(i)
            p = out_path.with_name(f'{out_path.stem}_seg{sid:04d}.obj')
            with open(p, 'w') as f:
                f.write(f'o curve_seg{sid:04d}\n')
                for x, y, z in np.asarray(c, dtype=np.float64):
                    f.write(f'v {x:.6f} {y:.6f} {z:.6f}\n')
                for k in range(len(c) - 1):
                    f.write(f'l {k + 1} {k + 2}\n')
            written.append(p)
        return written

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        base = 0
        for i, c in enumerate(curves):
            sid = _sid(i)
            f.write(f'o curve_seg{sid:04d}\n')
            f.write(f'g curve_seg{sid:04d}\n')
            for x, y, z in np.asarray(c, dtype=np.float64):
                f.write(f'v {x:.6f} {y:.6f} {z:.6f}\n')
            for k in range(len(c) - 1):
                f.write(f'l {base + k + 1} {base + k + 2}\n')
            base += len(c)
    written.append(out_path)
    return written


def resample_uniform(pts: np.ndarray, ds: float) -> tuple[np.ndarray, np.ndarray]:
    """Resample a polyline at uniform arclength spacing ≈ ds using cubic
    Hermite interpolation with non-uniform Catmull-Rom tangents.
    """
    pts = np.asarray(pts, dtype=np.float64)
    seg = np.diff(pts, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg_len)])
    L = float(s[-1])
    n = max(4, int(round(L / ds)) + 1)
    s_new = np.linspace(0.0, L, n)

    T = np.zeros_like(pts)
    T[1:-1] = (pts[2:] - pts[:-2]) / (s[2:] - s[:-2])[:, None]
    T[0]    = (pts[1]  - pts[0])  / (s[1]  - s[0])
    T[-1]   = (pts[-1] - pts[-2]) / (s[-1] - s[-2])

    idx = np.clip(np.searchsorted(s, s_new, side='right') - 1, 0, len(s) - 2)
    h = s[idx + 1] - s[idx]
    u = (s_new - s[idx]) / h
    u2 = u * u
    u3 = u2 * u
    h00 =  2.0 * u3 - 3.0 * u2 + 1.0
    h10 =        u3 - 2.0 * u2 + u
    h01 = -2.0 * u3 + 3.0 * u2
    h11 =        u3 -       u2

    out = (h00[:, None] * pts[idx]
           + (h10 * h)[:, None] * T[idx]
           + h01[:, None] * pts[idx + 1]
           + (h11 * h)[:, None] * T[idx + 1])
    return out, s_new


def curvature(pts_uniform: np.ndarray, ds_eff: float, sigma: float) -> np.ndarray:
    """κ(s) ≈ |r''(s)| for arclength-parameterised r."""
    sm = np.column_stack([gaussian_filter1d(pts_uniform[:, k], sigma)
                          for k in range(3)])
    d2 = np.zeros_like(sm)
    d2[1:-1] = (sm[2:] - 2.0 * sm[1:-1] + sm[:-2]) / (ds_eff * ds_eff)
    return np.linalg.norm(d2, axis=1)


def autocorr_normalised(x: np.ndarray) -> np.ndarray:
    """Zero-mean, lag-0 normalised autocorrelation (positive lags only)."""
    x = x - x.mean()
    n = len(x)
    full = np.correlate(x, x, mode='full')[n - 1:]
    if full[0] > 0:
        full = full / full[0]
    return full


def equispaced_resample(points_3d: np.ndarray, n_samples: int) -> np.ndarray:
    """Return ``n_samples`` 3D points equispaced along the polyline arclength
    of ``points_3d``. Linear interpolation between original vertices.

    Step: cumulative chord-length → linspace(0, L, n) → np.interp each axis.
    """
    pts = np.asarray(points_3d, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 2 or n_samples < 2:
        return np.repeat(pts[:1] if len(pts) else np.zeros((1, 3)),
                         max(1, n_samples), axis=0)
    seg_len = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg_len)])
    L = float(s[-1])
    if L <= 0.0:
        return np.repeat(pts[:1], n_samples, axis=0)
    s_new = np.linspace(0.0, L, n_samples)
    return np.column_stack([np.interp(s_new, s, pts[:, k]) for k in range(3)])


def procrustes_distance(A: np.ndarray, B: np.ndarray) -> float:
    """Sum of squared point-wise distances between A and B (both (N,3))
    after the optimal Kabsch rigid alignment (rotation + translation, no
    reflection, no scaling) of A onto B.

    Step: center both, SVD of A.T @ B, fix reflection via det sign, apply
    rotation, return Σ‖A_aligned − B‖².
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if A.shape != B.shape or A.ndim != 2 or A.shape[1] != 3 or len(A) < 2:
        return float('inf')
    Ac = A - A.mean(axis=0)
    Bc = B - B.mean(axis=0)
    H = Ac.T @ Bc
    try:
        U, _, Vt = np.linalg.svd(H)
    except np.linalg.LinAlgError:
        return float('inf')
    D = np.eye(3)
    if np.linalg.det(Vt.T @ U.T) < 0.0:
        D[2, 2] = -1.0
    R = Vt.T @ D @ U.T
    diff = Ac @ R.T - Bc
    return float(np.sum(diff * diff))


def local_writhe(pts_u: np.ndarray, ds_eff: float, window_voxels: float,
                 sigma: float) -> np.ndarray:
    """Local self-writhe density w(s) via Gauss linking integral within a
    centered window of arclength half-width ≈ ``window_voxels``.

    Positions are smoothed with σ (same as κ) before central-difference
    tangents. End samples where the full window doesn't fit return NaN.
    Output is dimensionless and signed (open-curve writhe density, not an
    integer crossing number).
    """
    sm = np.column_stack([gaussian_filter1d(pts_u[:, k], sigma)
                          for k in range(3)])
    n = len(sm)
    T = np.zeros_like(sm)
    T[1:-1] = (sm[2:] - sm[:-2]) / (2.0 * ds_eff)
    T[0]  = T[1]
    T[-1] = T[-2]

    half_w = max(2, int(round(window_voxels / ds_eff)))
    out = np.full(n, np.nan, dtype=np.float64)
    if 2 * half_w + 1 >= n:
        return out
    coef = (ds_eff * ds_eff) / (4.0 * np.pi)
    for c in range(half_w, n - half_w):
        a, b = c - half_w, c + half_w + 1
        r = sm[a:b]
        t = T[a:b]
        diff  = r[:, None, :] - r[None, :, :]
        dist2 = np.einsum('ijk,ijk->ij', diff, diff)
        np.fill_diagonal(dist2, 1.0)
        inv_d3 = dist2 ** (-1.5)
        np.fill_diagonal(inv_d3, 0.0)
        cross = np.cross(t[:, None, :], t[None, :, :])
        val   = np.einsum('ijk,ijk->ij', cross, diff) * inv_d3
        out[c] = coef * val.sum()
    return out


def analyse_curve(pts: np.ndarray, ds: float, sigma: float, top: int,
                  min_period: float, writhe_window: float | None = None):
    pts_u, s = resample_uniform(pts, ds)
    if len(pts_u) < 8:
        return None
    ds_eff = float(s[1] - s[0])
    kap = curvature(pts_u, ds_eff, sigma)
    ac  = autocorr_normalised(kap)

    lags = np.arange(len(ac)) * ds_eff
    min_lag = max(1, int(round(min_period / ds_eff)))
    if min_lag >= len(ac):
        peaks = np.array([], dtype=int)
        props = {'peak_heights': np.array([])}
    else:
        peaks, props = find_peaks(ac[min_lag:], height=0.05,
                                  distance=max(1, min_lag // 2))
        peaks = peaks + min_lag

    if len(peaks):
        order = np.argsort(props['peak_heights'])[::-1][:top]
        peaks_top = peaks[order]
        heights_top = props['peak_heights'][order]
        rank = np.argsort(peaks_top)
        peaks_top = peaks_top[rank]
        heights_top = heights_top[rank]
    else:
        peaks_top = np.array([], dtype=int)
        heights_top = np.array([])

    writhe = None
    writhe_ac = None
    if writhe_window is not None and writhe_window > 0:
        writhe = local_writhe(pts_u, ds_eff, writhe_window, sigma)
        valid = np.isfinite(writhe)
        if valid.sum() >= 8:
            writhe_ac = autocorr_normalised(writhe[valid])

    return {
        's': s, 'kappa': kap, 'lags': lags, 'ac': ac,
        'peaks': peaks_top, 'heights': heights_top,
        'ds_eff': ds_eff, 'L': float(s[-1]),
        'pts_u': pts_u,
        'writhe': writhe, 'writhe_ac': writhe_ac,
        'writhe_window': writhe_window,
    }


def plot_curve(res: dict, sid: int, out_path: Path) -> None:
    has_w = res.get('writhe') is not None
    nrows = 4 if has_w else 2
    fig, axes = plt.subplots(nrows, 1, figsize=(11, 2.6 * nrows + 0.3))
    axes = list(axes)

    ax_k, ax_kac = axes[0], axes[1]

    ax_k.plot(res['s'], res['kappa'], lw=0.8, color='#1f77b4')
    ax_k.set_xlabel('arclength s [voxels]')
    ax_k.set_ylabel('curvature κ(s)')
    ax_k.set_title(f'curve seg={sid}  L={res["L"]:.1f}  ds_eff={res["ds_eff"]:.2f}')
    ax_k.grid(alpha=0.3)

    ax_kac.plot(res['lags'], res['ac'], lw=0.8, color='#d62728')
    ax_kac.axhline(0, color='gray', lw=0.5)
    ax_kac.set_xlabel('lag [voxels]')
    ax_kac.set_ylabel('κ autocorr')
    ax_kac.set_title('κ(s) autocorrelation — peaks = candidate stitch periods')
    ax_kac.grid(alpha=0.3)
    for p, h in zip(res['peaks'], res['heights']):
        lag = p * res['ds_eff']
        ax_kac.axvline(lag, color='black', lw=0.5, ls='--', alpha=0.5)
        ax_kac.annotate(f'{lag:.0f}', xy=(lag, h),
                        xytext=(2, 2), textcoords='offset points', fontsize=8)

    if has_w:
        ax_w  = axes[2]
        ax_wac = axes[3]
        w = res['writhe']
        ax_w.plot(res['s'], w, lw=0.8, color='#2ca02c')
        ax_w.axhline(0, color='gray', lw=0.5)
        ax_w.set_xlabel('arclength s [voxels]')
        ax_w.set_ylabel('local writhe w(s)')
        ax_w.set_title(f'self-writhe density  (window=±{res["writhe_window"]:.0f} vox, σ={res["writhe_window"] and "shared"})')
        ax_w.grid(alpha=0.3)

        wac = res.get('writhe_ac')
        if wac is not None:
            wac_lags = np.arange(len(wac)) * res['ds_eff']
            ax_wac.plot(wac_lags, wac, lw=0.8, color='#9467bd')
            ax_wac.axhline(0, color='gray', lw=0.5)
            ax_wac.set_xlabel('lag [voxels]')
            ax_wac.set_ylabel('w autocorr')
            ax_wac.set_title('w(s) autocorrelation')
            ax_wac.grid(alpha=0.3)
        else:
            ax_wac.text(0.5, 0.5, 'w(s) too short for autocorr',
                        ha='center', va='center', transform=ax_wac.transAxes)
            ax_wac.set_axis_off()

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# Polyscope state (template-matching only; user picks span and presses Match)
_view_state: dict = {
    'curves': [],          # list of dicts: {sid, res}
    'curve_idx': 0,
    'pick_mode': None,     # None | 'start' | 'end'
    'template_idx0': 1391, # default span (start)
    'template_idx1': 1876, # default span (end)
    'match_threshold': 0.7,
    'match_top': 10,
    'match_scores': None,
    'match_template_len': 0,
    'match_w_weight': 0.5, # α in (1−α)·Pearson_κ + α·Pearson_w
    'match_mode': '',      # last computed mode label
    'match_kind': '',      # 'pearson' | 'hmm'
    'hmm_max_stretch': 1.5,
    'hmm_segments': None,  # HMM/Viterbi: list of (start_idx, end_idx) inclusive
    'hmm_gap_cost': 0.8,   # β: per-sample cost of "not in a stitch"
    'hmm_total_cost': None,
    'hmm_gap_frac': None,
    # Procrustes auto-regressive detector (3D rigid-aligned template match)
    'procrustes_segments':         None, # list of (start_idx, end_idx) inclusive
    'procrustes_n_samples':        30,
    'procrustes_threshold':        100.0,
    'procrustes_length_variation': 0.3,
    'procrustes_total_cost':       None,
    'procrustes_n_detected':       0,
    'procrustes_n_fwd':            0,
    'procrustes_n_bwd':            0,
    'procrustes_median_cost':      None,
    'procrustes_s_fwd':            None,  # current forward frontier (seed for next step)
    'procrustes_s_bwd':            None,  # current backward frontier
    'procrustes_L_fwd':            None,  # EMA length estimate (forward)
    'procrustes_L_bwd':            None,
    'procrustes_costs':            None,  # list of per-step rms costs
    # LIFO undo stacks: one record per appended Step Fwd / Step Bwd, so the
    # last step in either direction can be individually rolled back.
    'procrustes_undo_fwd':         [],
    'procrustes_undo_bwd':         [],
    # Names of polyscope curve networks created for individual segments,
    # so each Step Fwd/Bwd output is its own toggle-able layer.
    'procrustes_seg_names':        [],
    # Per-segment uniformly-sampled point clouds (one per segment, same
    # color), and a toggle to show/hide them all at once.
    'procrustes_seg_sample_names': [],
    'procrustes_show_samples':     True,
    # One curve-network connecting adjacent segments' 10 sample points by
    # within-segment index (sample i of seg k → sample i of seg k+1).
    'procrustes_seg_corr_name':    None,
    'procrustes_show_corr':        True,
    # ── Auto stitch-type match: pick ONE start, classify via N templates ────
    'match_start_idx':   -1,     # picked start node on the current curve
    'stitch_templates':  [],     # [{name, pts(Nx3), color[3], arclen}]
    'stitch_softmax_T':  15.0,   # softmax temperature on rms (voxels)
    'automatch':         None,   # last result {name,color,s,e,rms,prob,all}
    'match_rates':       None,   # softmax rates of the SELECTED span vs templates
}


def _curve_net_name(sid: int) -> str:
    return f'curve_seg{sid:04d}'


def _samples_pc_name(sid: int) -> str:
    return f'samples_seg{sid:04d}'


def _template_net_name(sid: int) -> str:
    return f'template_seg{sid:04d}'


def _matchhi_net_name(sid: int) -> str:
    return f'match_hi_seg{sid:04d}'


def _procrustes_hi_net_name(sid: int) -> str:
    return f'procrustes_hi_seg{sid:04d}'


def _start_marker_name(sid: int) -> str:
    return f'tpl_start_seg{sid:04d}'


def _end_marker_name(sid: int) -> str:
    return f'tpl_end_seg{sid:04d}'


def _refresh_endpoint_markers() -> None:
    """Show a green dot at template start and red dot at template end."""
    vs = _view_state
    if not vs['curves']:
        return
    cur = vs['curves'][vs['curve_idx']]
    sid = cur['sid']
    pts_u = cur['res']['pts_u']
    n = len(pts_u)
    nm_s, nm_e = _start_marker_name(sid), _end_marker_name(sid)
    for nm in (nm_s, nm_e):
        if ps.has_point_cloud(nm):
            ps.remove_point_cloud(nm)
    i0, i1 = int(vs['template_idx0']), int(vs['template_idx1'])
    if 0 <= i0 < n:
        pc = ps.register_point_cloud(nm_s, pts_u[i0:i0 + 1], radius=0.005)
        pc.set_color((0.1, 0.85, 0.2))
    if 0 <= i1 < n:
        pc = ps.register_point_cloud(nm_e, pts_u[i1:i1 + 1], radius=0.005)
        pc.set_color((0.95, 0.1, 0.1))


def _refresh_template_highlight() -> None:
    """Orange highlight for template span — only drawn when both ends picked."""
    vs = _view_state
    if not vs['curves']:
        return
    cur = vs['curves'][vs['curve_idx']]
    sid = cur['sid']
    pts_u = cur['res']['pts_u']
    n = len(pts_u)
    nm = _template_net_name(sid)
    if ps.has_curve_network(nm):
        ps.remove_curve_network(nm)
    i0, i1 = int(vs['template_idx0']), int(vs['template_idx1'])
    if i0 < 0 or i1 < 0 or i1 <= i0 or i1 >= n:
        return
    seg_pts = pts_u[i0:i1 + 1]
    seg_edges = np.stack([np.arange(len(seg_pts) - 1),
                          np.arange(1, len(seg_pts))], axis=1).astype(np.int32)
    cn = ps.register_curve_network(nm, seg_pts, seg_edges, radius=0.0022)
    cn.set_color((1.0, 0.45, 0.0))


def _pearson_slide(signal: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Per-position Pearson r of ``template`` (length m) against sliding
    windows of ``signal`` (length n). Returns length n − m + 1.
    NaN-containing windows or templates yield NaN."""
    n, m = len(signal), len(template)
    out = np.full(n - m + 1, np.nan, dtype=np.float64)
    if not np.all(np.isfinite(template)):
        return out
    t  = template - template.mean()
    tn = float(np.linalg.norm(t))
    if tn < 1e-9:
        return out
    for i in range(n - m + 1):
        w = signal[i:i + m]
        if not np.all(np.isfinite(w)):
            continue
        wn  = w - w.mean()
        wnn = float(np.linalg.norm(wn))
        out[i] = 0.0 if wnn < 1e-9 else float(np.dot(wn, t) / (wnn * tn))
    return out


def _zscore(x: np.ndarray) -> np.ndarray:
    """Z-normalise; NaN replaced with 0, constant input returns zeros."""
    x = np.asarray(x, dtype=np.float64)
    finite = np.isfinite(x)
    if finite.sum() < 2:
        return np.zeros_like(x)
    mu = float(x[finite].mean())
    sd = float(x[finite].std())
    if sd < 1e-12:
        return np.where(finite, 0.0, 0.0)
    z = (x - mu) / sd
    return np.where(np.isfinite(z), z, 0.0)


def _subseq_dtw(template_chs: list[np.ndarray],
                signal_chs:   list[np.ndarray],
                weights:      list[float]) -> tuple[np.ndarray, np.ndarray]:
    """Subsequence DTW with multi-channel weighted abs cost.

    template can begin at any signal index (free start), must end at the
    queried index (we return per-end cost). Step set: (−1,−1), (−1, 0), (0, −1).
    Returns (cost_per_end, start_per_end), both length n. NaN if not reachable.
    """
    m = len(template_chs[0])
    n = len(signal_chs[0])
    nch = len(template_chs)
    if m < 2 or n < m:
        return np.full(n, np.inf), np.full(n, -1, dtype=np.int64)

    def crow(i: int) -> np.ndarray:
        c = np.zeros(n, dtype=np.float64)
        for k in range(nch):
            c += weights[k] * np.abs(template_chs[k][i] - signal_chs[k])
        return c

    D_prev = crow(0)
    S_prev = np.arange(n, dtype=np.int64)
    for i in range(1, m):
        ci = crow(i)
        d_left  = D_prev[:-1]
        d_above = D_prev[1:]
        left_win = d_left <= d_above
        d_pred = np.where(left_win, d_left, d_above)
        s_pred = np.where(left_win, S_prev[:-1], S_prev[1:])
        D_curr = np.empty(n, dtype=np.float64)
        S_curr = np.empty(n, dtype=np.int64)
        D_curr[0] = ci[0] + D_prev[0]
        S_curr[0] = S_prev[0]
        for j in range(1, n):
            pred_d = D_curr[j - 1]
            pred_s = S_curr[j - 1]
            cand_d = d_pred[j - 1]
            cand_s = s_pred[j - 1]
            if cand_d <= pred_d:
                D_curr[j] = ci[j] + cand_d
                S_curr[j] = cand_s
            else:
                D_curr[j] = ci[j] + pred_d
                S_curr[j] = pred_s
        D_prev = D_curr
        S_prev = S_curr
    return D_prev, S_prev


def _compute_template_match_hmm(use_writhe: bool = False) -> None:
    """HSMM / segment-level Viterbi: globally segment the signal into a
    non-overlapping sequence of (stitch | gap) segments that minimises
        Σ DTW_cost(stitch_k)  +  β · (total gap length)
    subject to stitch length ∈ [m/α, m·α].

    Reuses the precomputed sub-sequence DTW: for each end index j we only
    consider the single best start = starts[j] (not arbitrary (i,j) pairs).
    This is the main silent simplification vs. a full HSMM — worth knowing
    if a "correct" stitch is getting beaten by DTW picking a nearby start.
    """
    vs = _view_state
    if not vs['curves']:
        return
    cur = vs['curves'][vs['curve_idx']]
    res = cur['res']
    kap = res['kappa']
    n = len(kap)
    i0, i1 = int(vs['template_idx0']), int(vs['template_idx1'])
    if i0 < 0 or i1 < 0 or i1 <= i0 or i1 >= n:
        vs['hmm_segments'] = None
        vs['match_template_len'] = 0
        vs['match_kind'] = ''
        vs['match_mode'] = ''
        return
    m = i1 - i0 + 1
    if m < 3 or m >= n:
        vs['hmm_segments'] = None
        vs['match_template_len'] = 0
        vs['match_kind'] = ''
        vs['match_mode'] = ''
        return

    kap_z = _zscore(kap)
    t_kap = kap_z[i0:i1 + 1]

    if use_writhe and res.get('writhe') is not None:
        wr_z = _zscore(res['writhe'])
        t_wr = wr_z[i0:i1 + 1]
        a = float(vs['match_w_weight'])
        cost, starts = _subseq_dtw([t_kap, t_wr], [kap_z, wr_z],
                                   [1.0 - a, a])
        label = f'HMM κ+w (α={a:.2f})'
    else:
        cost, starts = _subseq_dtw([t_kap], [kap_z], [1.0])
        label = 'HMM κ'

    max_stretch = float(vs['hmm_max_stretch'])
    L_min = max(2, int(round(m / max_stretch)))
    L_max = int(round(m * max_stretch))
    beta = float(vs['hmm_gap_cost'])

    # Segment DP: f[j] = min cost to explain signal[:j]. j ∈ [0..n].
    # back[j] encodes the last segment: ('gap', j-1) or ('stitch', start, end=j-1)
    INF = np.inf
    f = np.full(n + 1, INF, dtype=np.float64)
    f[0] = 0.0
    back: list = [None] * (n + 1)

    for j in range(1, n + 1):
        best = f[j - 1] + beta
        best_back = ('gap', j - 1)
        end_idx = j - 1
        s_idx = int(starts[end_idx]) if end_idx < n else -1
        if s_idx >= 0 and np.isfinite(cost[end_idx]):
            seg_len = end_idx - s_idx + 1
            if L_min <= seg_len <= L_max:
                cand = f[s_idx] + float(cost[end_idx])
                if cand < best:
                    best = cand
                    best_back = ('stitch', s_idx, end_idx)
        f[j] = best
        back[j] = best_back

    segments: list[tuple[int, int]] = []
    j = n
    gap_samples = 0
    while j > 0:
        bk = back[j]
        if bk is None:
            break
        if bk[0] == 'gap':
            gap_samples += 1
            j = bk[1]
        else:
            segments.append((bk[1], bk[2]))
            j = bk[1]
    segments.reverse()

    vs['hmm_segments']       = segments
    vs['hmm_total_cost']     = float(f[n])
    vs['hmm_gap_frac']       = gap_samples / max(1, n)
    vs['match_kind']         = 'hmm'
    vs['match_mode']         = label
    vs['match_template_len'] = m
    vs['match_scores']       = None


def _compute_template_match(use_writhe: bool = False) -> None:
    """Sliding Pearson cross-correlation of κ (and optionally w) template."""
    vs = _view_state
    if not vs['curves']:
        return
    cur = vs['curves'][vs['curve_idx']]
    kap = cur['res']['kappa']
    n = len(kap)
    i0, i1 = int(vs['template_idx0']), int(vs['template_idx1'])
    if i0 < 0 or i1 < 0 or i1 <= i0 or i1 >= n:
        vs['match_scores'] = None
        vs['match_template_len'] = 0
        vs['match_mode'] = ''
        return
    m = i1 - i0 + 1
    if m < 3 or m >= n:
        vs['match_scores'] = None
        vs['match_template_len'] = 0
        vs['match_mode'] = ''
        return

    s_k = _pearson_slide(kap, kap[i0:i1 + 1])

    if use_writhe and cur['res'].get('writhe') is not None:
        wr = cur['res']['writhe']
        s_w = _pearson_slide(wr, wr[i0:i1 + 1])
        a = float(vs['match_w_weight'])
        scores = (1.0 - a) * np.where(np.isfinite(s_k), s_k, 0.0) \
                 + a * np.where(np.isfinite(s_w), s_w, 0.0)
        scores[~np.isfinite(s_k) | ~np.isfinite(s_w)] = np.nan
        vs['match_mode'] = f'κ+w (α={a:.2f})'
    else:
        scores = s_k
        vs['match_mode'] = 'κ'

    vs['match_scores'] = scores
    vs['match_template_len'] = m
    vs['match_kind'] = 'pearson'


def _procrustes_single_step(pts_u: np.ndarray, tmpl_rs: np.ndarray,
                            s: int, L_exp: float, direction: int,
                            N: int, var: float) -> tuple[int, float]:
    """One AR iteration: grid-search the next segment endpoint.

    direction=+1: search e ∈ [s + (1−v)L, s + (1+v)L], window pts_u[s:e+1].
    direction=−1: search e ∈ [s − (1+v)L, s − (1−v)L], window pts_u[e:s+1].
    Returns (best_e, best_rms_per_point). best_e < 0 if range empty.
    """
    n = len(pts_u)
    L_min = max(3, int(round((1.0 - var) * L_exp)))
    L_max = int(round((1.0 + var) * L_exp))
    best_c, best_e = np.inf, -1
    if direction > 0:
        e_lo = s + L_min
        e_hi = min(n - 1, s + L_max)
        if e_lo > e_hi or e_lo >= n:
            return -1, float('inf')
        rng = range(e_lo, e_hi + 1)
        def _win(e: int) -> np.ndarray:
            return pts_u[s:e + 1]
    else:
        e_hi = s - L_min
        e_lo = max(0, s - L_max)
        if e_lo > e_hi or e_hi < 0:
            return -1, float('inf')
        rng = range(e_hi, e_lo - 1, -1)
        def _win(e: int) -> np.ndarray:
            return pts_u[e:s + 1]
    for e in rng:
        d2 = procrustes_distance(tmpl_rs, equispaced_resample(_win(e), N))
        c = float(np.sqrt(d2 / N)) if np.isfinite(d2) else float('inf')
        if c < best_c:
            best_c, best_e = c, e
    return best_e, best_c


def _load_stitch_templates(path) -> list:
    """Load stitch-type templates from a json list of {name, npy, color[3]}.
    npy paths are relative to the json's directory. Returns
    [{name, pts(Nx3 float), color[3], arclen}]."""
    import json
    path = Path(path)
    if not path.is_file():
        print(f'[stitch-match] no templates json: {path}')
        return []
    base = path.parent
    out = []
    for e in json.loads(path.read_text()):
        npy = Path(e['npy'])
        if not npy.is_absolute():
            npy = base / npy
        if not npy.is_file():
            print(f'[stitch-match] template npy missing: {npy}'); continue
        pts = np.asarray(np.load(npy), np.float64).reshape(-1, 3)
        arclen = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
        out.append({'name': str(e['name']),
                    'color': [float(c) for c in e['color']],
                    'pts': pts, 'arclen': arclen})
        print(f'[stitch-match] template "{e["name"]}": {len(pts)} pts, '
              f'arclen={arclen:.1f}, color={e["color"]}')
    return out


def _automatch_net_name(sid: int) -> str:
    return f'automatch_seg{sid:04d}'


def _match_start_marker_name(sid: int) -> str:
    return f'matchstart_seg{sid:04d}'


def _draw_match_start_marker() -> None:
    """Magenta dot at the picked auto-match start point."""
    vs = _view_state
    if not vs['curves']:
        return
    cur = vs['curves'][vs['curve_idx']]
    sid = cur['sid']
    pts_u = cur['res']['pts_u']
    nm = _match_start_marker_name(sid)
    if ps.has_point_cloud(nm):
        ps.remove_point_cloud(nm)
    i = int(vs['match_start_idx'])
    if 0 <= i < len(pts_u):
        pc = ps.register_point_cloud(nm, pts_u[i:i + 1], radius=0.006)
        pc.set_color((1.0, 0.0, 1.0))


def _clear_automatch() -> None:
    vs = _view_state
    if vs['curves']:
        sid = vs['curves'][vs['curve_idx']]['sid']
        for nm in (_automatch_net_name(sid),):
            if ps.has_curve_network(nm):
                ps.remove_curve_network(nm)
        nm_s = _match_start_marker_name(sid)
        if ps.has_point_cloud(nm_s):
            ps.remove_point_cloud(nm_s)
    vs['automatch'] = None


def _draw_automatch() -> None:
    """Draw the matched span pts_u[s:e+1] as a tube in the winning template
    color."""
    vs = _view_state
    r = vs.get('automatch')
    if not r or not vs['curves']:
        return
    cur = vs['curves'][vs['curve_idx']]
    sid = cur['sid']
    pts_u = cur['res']['pts_u']
    s, e = int(r['s']), int(r['e'])
    nm = _automatch_net_name(sid)
    if ps.has_curve_network(nm):
        ps.remove_curve_network(nm)
    if not (0 <= s < e < len(pts_u)):
        return
    seg = pts_u[s:e + 1]
    edges = np.stack([np.arange(len(seg) - 1), np.arange(1, len(seg))],
                     axis=1).astype(np.int32)
    cn = ps.register_curve_network(nm, seg, edges, radius=0.0026)
    cn.set_color(tuple(r['color']))


def _auto_match_stitch() -> None:
    """From the single picked start point, Kabsch-match each stitch template
    forward (the grid search refines the end by min rms), softmax the per-
    template rms to pick the type, and draw the matched span in that type's
    color.  Only the start is user-chosen; everything else is automatic."""
    vs = _view_state
    if not vs['curves']:
        print('[stitch-match] no curve'); return
    tmpls = vs.get('stitch_templates') or []
    if not tmpls:
        print('[stitch-match] no templates loaded (--stitch_templates)'); return
    s = int(vs['match_start_idx'])
    cur = vs['curves'][vs['curve_idx']]
    pts_u = np.asarray(cur['res']['pts_u'], np.float64)
    n = len(pts_u)
    if not (0 <= s < n - 3):
        print(f'[stitch-match] invalid start {s}'); return
    ds  = float(cur['res']['ds_eff'])
    N   = max(4, int(vs['procrustes_n_samples']))
    var = float(np.clip(vs['procrustes_length_variation'], 0.01, 0.99))

    # 1) Kabsch each template forward from s; grid search refines the end.
    results = []
    for t in tmpls:
        tmpl_rs = equispaced_resample(t['pts'], N)
        L_exp = max(3.0, t['arclen'] / max(ds, 1e-6))   # expected #samples
        best_e, best_rms = _procrustes_single_step(
            pts_u, tmpl_rs, s, L_exp, +1, N, var)
        results.append({'name': t['name'], 'color': t['color'],
                        'e': int(best_e), 'rms': float(best_rms)})
    finite = [r for r in results if np.isfinite(r['rms']) and r['e'] > s]
    if not finite:
        print('[stitch-match] no valid match from this start'); return

    # 2) Softmax over -rms/T -> per-type probability (lower rms => higher p).
    T = max(1e-3, float(vs['stitch_softmax_T']))
    rms = np.array([r['rms'] for r in finite], float)
    z = -(rms - rms.min()) / T
    p = np.exp(z); p = p / p.sum()
    for r, pi in zip(finite, p):
        r['prob'] = float(pi)
    win = max(finite, key=lambda r: r['prob'])          # == argmin rms
    vs['automatch'] = {'name': win['name'], 'color': win['color'],
                       's': s, 'e': win['e'], 'rms': win['rms'],
                       'prob': win['prob'],
                       'all': sorted(finite, key=lambda r: r['rms'])}
    _draw_automatch()
    print(f'[stitch-match] start={s} -> WIN "{win["name"]}" end={win["e"]} '
          f'rms={win["rms"]:.2f} p={win["prob"]:.2f}')
    for r in sorted(finite, key=lambda r: r['rms']):
        print(f'    {r["name"]:>10s}  rms={r["rms"]:.2f}  p={r["prob"]:.2f}  '
              f'end={r["e"]}')


def _stitch_match_rates() -> None:
    """For the SELECTED span [template_idx0, template_idx1] (the orange span
    from Pick start + Pick end), compute the Kabsch rms against each template
    and softmax them into matching rates. Uses the span EXACTLY as picked — no
    end search. Stores/prints per-template rms + rate (%)."""
    vs = _view_state
    if not vs['curves']:
        print('[match-rate] no curve'); return
    tmpls = vs.get('stitch_templates') or []
    if not tmpls:
        print('[match-rate] no templates loaded'); return
    cur = vs['curves'][vs['curve_idx']]
    pts_u = np.asarray(cur['res']['pts_u'], np.float64)
    n = len(pts_u)
    i0, i1 = int(vs['template_idx0']), int(vs['template_idx1'])
    if not (0 <= i0 < i1 < n):
        print('[match-rate] select a stitch span first (Pick start + Pick end)')
        return
    N = max(4, int(vs['procrustes_n_samples']))
    span_rs = equispaced_resample(pts_u[i0:i1 + 1], N)
    res = []
    for t in tmpls:
        tmpl_rs = equispaced_resample(t['pts'], N)
        d2 = procrustes_distance(tmpl_rs, span_rs)
        rms = float(np.sqrt(d2 / N)) if np.isfinite(d2) else float('inf')
        res.append({'name': t['name'], 'color': t['color'], 'rms': rms})
    finite = [r for r in res if np.isfinite(r['rms'])]
    if not finite:
        print('[match-rate] no valid comparison'); return
    T = max(1e-3, float(vs['stitch_softmax_T']))
    rms = np.array([r['rms'] for r in finite], float)
    z = -(rms - rms.min()) / T
    p = np.exp(z); p = p / p.sum()
    for r, pi in zip(finite, p):
        r['prob'] = float(pi)
    vs['match_rates'] = sorted(finite, key=lambda r: r['rms'])
    print(f'[match-rate] selected span [{i0},{i1}] ({i1 - i0 + 1} samples):')
    for r in vs['match_rates']:
        print(f'    {r["name"]:>10s}  rms={r["rms"]:.2f}  rate={r["prob"]*100:.1f}%')


def _procrustes_reset_state() -> None:
    """Clear all AR state so the next Step/Match starts fresh from template."""
    vs = _view_state
    vs['procrustes_segments']   = None
    vs['procrustes_s_fwd']      = None
    vs['procrustes_s_bwd']      = None
    vs['procrustes_L_fwd']      = None
    vs['procrustes_L_bwd']      = None
    vs['procrustes_costs']      = None
    vs['procrustes_n_fwd']      = 0
    vs['procrustes_n_bwd']      = 0
    vs['procrustes_total_cost'] = None
    vs['procrustes_median_cost']= None
    vs['procrustes_undo_fwd']   = []
    vs['procrustes_undo_bwd']   = []
    # Note: per-segment polyscope networks are removed by callers
    # (Clear Procrustes button, curve switch) — this just resets logical state.
    vs['procrustes_seg_names']  = []


def _procrustes_seed_from_template() -> bool:
    """Initialise AR state from the current template span. Returns False
    if the template is invalid."""
    vs = _view_state
    if not vs['curves']:
        return False
    cur = vs['curves'][vs['curve_idx']]
    pts_u = cur['res']['pts_u']
    n = len(pts_u)
    i0, i1 = int(vs['template_idx0']), int(vs['template_idx1'])
    if i0 < 0 or i1 < 0 or i1 <= i0 or i1 >= n:
        return False
    L_nominal = i1 - i0 + 1
    if L_nominal < 3:
        return False
    vs['procrustes_segments']   = [(i0, i1)]
    vs['procrustes_s_fwd']      = i1
    vs['procrustes_s_bwd']      = i0
    vs['procrustes_L_fwd']      = float(L_nominal)
    vs['procrustes_L_bwd']      = float(L_nominal)
    vs['procrustes_costs']      = []
    vs['procrustes_n_fwd']      = 0
    vs['procrustes_n_bwd']      = 0
    vs['procrustes_total_cost'] = 0.0
    vs['procrustes_median_cost']= 0.0
    vs['procrustes_undo_fwd']   = []
    vs['procrustes_undo_bwd']   = []
    vs['match_template_len']    = L_nominal
    vs['match_kind']            = 'procrustes'
    vs['match_scores']          = None
    return True


def _procrustes_push_undo(direction: int, seg, cost: float,
                          prev_s: int, prev_L: float) -> None:
    """Record everything needed to roll back one appended step in
    ``direction`` (so Undo Fwd/Bwd can cancel the last Step in that
    direction): the appended segment + its rms cost, plus the frontier
    index and EMA length that were in effect *before* the step."""
    key = 'procrustes_undo_fwd' if direction > 0 else 'procrustes_undo_bwd'
    _view_state.setdefault(key, []).append(
        {'seg': (int(seg[0]), int(seg[1])), 'cost': float(cost),
         'prev_s': int(prev_s), 'prev_L': float(prev_L)})


def _procrustes_undo_step(direction: int) -> bool:
    """Roll back the most recent Step in ``direction`` (+1 fwd / -1 bwd):
    drop its segment + cost, restore the prior frontier / EMA length, and
    decrement the count. The template seed segment is never removed.
    Returns True if something was undone."""
    vs = _view_state
    tag = 'fwd' if direction > 0 else 'bwd'
    hist = vs.get('procrustes_undo_fwd' if direction > 0
                  else 'procrustes_undo_bwd') or []
    if not hist:
        print(f'Procrustes undo {tag}: nothing to undo')
        return False
    rec  = hist.pop()
    segs = vs.get('procrustes_segments') or []
    for k in range(len(segs) - 1, -1, -1):          # last exact match
        if tuple(segs[k]) == rec['seg']:
            del segs[k]
            break
    costs = vs.get('procrustes_costs') or []
    for k in range(len(costs) - 1, -1, -1):          # its rms cost
        if costs[k] == rec['cost']:
            del costs[k]
            break
    if direction > 0:
        vs['procrustes_s_fwd'] = rec['prev_s']
        vs['procrustes_L_fwd'] = rec['prev_L']
        vs['procrustes_n_fwd'] = max(0, int(vs['procrustes_n_fwd']) - 1)
    else:
        vs['procrustes_s_bwd'] = rec['prev_s']
        vs['procrustes_L_bwd'] = rec['prev_L']
        vs['procrustes_n_bwd'] = max(0, int(vs['procrustes_n_bwd']) - 1)
    vs['procrustes_total_cost']  = float(sum(costs)) if costs else 0.0
    vs['procrustes_median_cost'] = float(np.median(costs)) if costs else 0.0
    vs['match_kind'] = 'procrustes'
    print(f'Procrustes undo {tag}: removed seg {rec["seg"]} '
          f'(rms={rec["cost"]:.2f})  → {len(segs)} segment(s) left')
    return True


def _procrustes_step(direction: int) -> None:
    """Do one AR iteration in the given direction and append the result
    if it passes τ. Prints outcome either way."""
    vs = _view_state
    if vs.get('procrustes_segments') is None:
        if not _procrustes_seed_from_template():
            print('Procrustes step: no valid template')
            return
    cur = vs['curves'][vs['curve_idx']]
    pts_u = cur['res']['pts_u']
    i0 = int(vs['template_idx0'])
    i1 = int(vs['template_idx1'])

    N   = max(4, int(vs['procrustes_n_samples']))
    thr = float(vs['procrustes_threshold'])
    var = float(np.clip(vs['procrustes_length_variation'], 0.01, 0.99))
    ema_alpha = 0.2
    tmpl_rs = equispaced_resample(pts_u[i0:i1 + 1], N)

    if direction > 0:
        s     = int(vs['procrustes_s_fwd'])
        L_exp = float(vs['procrustes_L_fwd'])
    else:
        s     = int(vs['procrustes_s_bwd'])
        L_exp = float(vs['procrustes_L_bwd'])

    n = len(pts_u)
    L_min = max(3, int(round((1.0 - var) * L_exp)))
    L_max = int(round((1.0 + var) * L_exp))
    if direction > 0:
        e_lo = s + L_min
        e_hi = min(n - 1, s + L_max)
        e_guess = s + int(round(L_exp))
    else:
        e_hi = s - L_min
        e_lo = max(0, s - L_max)
        e_guess = s - int(round(L_exp))

    best_e, best_c = _procrustes_single_step(pts_u, tmpl_rs, s, L_exp,
                                              direction, N, var)
    tag = 'fwd' if direction > 0 else 'bwd'

    print(f'Procrustes step {tag}: s={s}  L_exp={L_exp:.1f} '
          f'(nominal len = {int(round(L_exp))})')
    print(f'  search e∈[{e_lo}, {e_hi}]  center guess e={e_guess} '
          f'(len={abs(e_guess - s) + 1})')

    if best_e < 0:
        print('  → no candidates (reached curve end)')
        return

    opt_len = abs(best_e - s) + 1
    delta   = best_e - e_guess
    print(f'  → optimized  e={best_e} (len={opt_len})  '
          f'Δe={delta:+d}  rms={best_c:.2f} vox/pt')

    if best_c > thr:
        print(f'  ✗ rms > τ={thr:.2f} — not appended')
        return

    if direction > 0:
        _procrustes_push_undo(+1, (s, best_e), best_c, s, L_exp)
        vs['procrustes_segments'].append((s, best_e))
        new_L = (1.0 - ema_alpha) * L_exp + ema_alpha * opt_len
        vs['procrustes_L_fwd'] = new_L
        vs['procrustes_s_fwd'] = best_e
        vs['procrustes_n_fwd'] = int(vs['procrustes_n_fwd']) + 1
    else:
        _procrustes_push_undo(-1, (best_e, s), best_c, s, L_exp)
        vs['procrustes_segments'].append((best_e, s))
        new_L = (1.0 - ema_alpha) * L_exp + ema_alpha * opt_len
        vs['procrustes_L_bwd'] = new_L
        vs['procrustes_s_bwd'] = best_e
        vs['procrustes_n_bwd'] = int(vs['procrustes_n_bwd']) + 1

    vs['procrustes_segments'].sort(key=lambda ab: ab[0])
    vs['procrustes_costs'].append(best_c)
    vs['procrustes_total_cost']  = float(sum(vs['procrustes_costs']))
    vs['procrustes_median_cost'] = float(np.median(vs['procrustes_costs']))
    vs['match_kind']             = 'procrustes'
    vs['match_mode']             = (f'Procrustes AR step '
                                    f'(N={N}, τ={thr:.2f}, v={var:.2f})')
    print(f'  ✓ appended  L_exp: {L_exp:.1f} → {new_L:.1f}')


def _compute_stitch_match_procrustes() -> None:
    """Auto-regressive stitch detector using Procrustes distance on 3D
    curve segments. Steps:
      1. Resample template (pts_u[i0:i1+1]) to N equispaced points.
      2. Forward pass from s=i1: grid-search e ∈ [s+(1-v)L, s+(1+v)L];
         for each candidate resample the window pts_u[s:e+1] to N points
         and score with procrustes_distance(template_rs, window_rs).
      3. Pick e* = argmin; stop if cost > threshold. Otherwise record
         (s, e*), update L via running average, and repeat from s=e*.
      4. Backward pass from s=i0 in reverse, symmetric.
    The template span itself is included as the seed stitch. Results
    written to vs['procrustes_segments'] sorted by start index.
    """
    vs = _view_state
    if not vs['curves']:
        return
    cur = vs['curves'][vs['curve_idx']]
    pts_u = cur['res']['pts_u']
    n = len(pts_u)
    i0, i1 = int(vs['template_idx0']), int(vs['template_idx1'])
    if i0 < 0 or i1 < 0 or i1 <= i0 or i1 >= n:
        vs['procrustes_segments'] = None
        return
    L_nominal = i1 - i0 + 1
    if L_nominal < 3:
        vs['procrustes_segments'] = None
        return

    N   = max(4, int(vs['procrustes_n_samples']))
    thr = float(vs['procrustes_threshold'])   # per-point RMS residual [voxels]
    var = float(np.clip(vs['procrustes_length_variation'], 0.01, 0.99))
    ema_alpha = 0.2

    tmpl_rs = equispaced_resample(pts_u[i0:i1 + 1], N)

    # Seed AR state from template (resets any prior step state).
    _procrustes_seed_from_template()
    reject_fwd_c = None
    reject_bwd_c = None

    # ---- forward pass ----
    s = int(vs['procrustes_s_fwd'])
    L_exp = float(vs['procrustes_L_fwd'])
    while True:
        best_e, best_c = _procrustes_single_step(pts_u, tmpl_rs, s, L_exp,
                                                  +1, N, var)
        if best_e < 0:
            break
        if best_c > thr:
            reject_fwd_c = best_c
            break
        vs['procrustes_segments'].append((s, best_e))
        vs['procrustes_costs'].append(best_c)
        _procrustes_push_undo(+1, (s, best_e), best_c, s, L_exp)
        L_exp = (1.0 - ema_alpha) * L_exp + ema_alpha * (best_e - s + 1)
        s = best_e
        vs['procrustes_n_fwd'] = int(vs['procrustes_n_fwd']) + 1
    vs['procrustes_s_fwd'] = s
    vs['procrustes_L_fwd'] = L_exp

    # ---- backward pass ----
    s = int(vs['procrustes_s_bwd'])
    L_exp = float(vs['procrustes_L_bwd'])
    while True:
        best_e, best_c = _procrustes_single_step(pts_u, tmpl_rs, s, L_exp,
                                                  -1, N, var)
        if best_e < 0:
            break
        if best_c > thr:
            reject_bwd_c = best_c
            break
        vs['procrustes_segments'].append((best_e, s))
        vs['procrustes_costs'].append(best_c)
        _procrustes_push_undo(-1, (best_e, s), best_c, s, L_exp)
        L_exp = (1.0 - ema_alpha) * L_exp + ema_alpha * (s - best_e + 1)
        s = best_e
        vs['procrustes_n_bwd'] = int(vs['procrustes_n_bwd']) + 1
    vs['procrustes_s_bwd'] = s
    vs['procrustes_L_bwd'] = L_exp

    vs['procrustes_segments'].sort(key=lambda ab: ab[0])
    costs = vs['procrustes_costs']
    median_c = float(np.median(costs)) if costs else 0.0
    n_fwd = int(vs['procrustes_n_fwd'])
    n_bwd = int(vs['procrustes_n_bwd'])

    vs['procrustes_n_detected'] = n_fwd + n_bwd
    vs['procrustes_total_cost'] = float(sum(costs)) if costs else 0.0
    vs['procrustes_median_cost']= median_c
    vs['match_mode']            = f'Procrustes AR (N={N}, τ={thr:.2f}, v={var:.2f})'

    print(f'Procrustes: detected {n_fwd} stitches forward + {n_bwd} backward, '
          f'median_cost={median_c:.3f} vox/pt  (τ={thr:.2f})')
    if n_fwd == 0 and reject_fwd_c is not None:
        print(f'  forward stopped at first step: best rms={reject_fwd_c:.2f} vox/pt '
              f'> τ — raise τ above that to proceed')
    if n_bwd == 0 and reject_bwd_c is not None:
        print(f'  backward stopped at first step: best rms={reject_bwd_c:.2f} vox/pt '
              f'> τ — raise τ above that to proceed')


def _draw_segment_highlights(pts_u: np.ndarray,
                             segments: list[tuple[int, int]],
                             nm_hi: str) -> None:
    """Register a curve-network with per-edge HSV-cycled colours for
    ``segments`` (list of (start_idx, end_idx) inclusive)."""
    hi_pts, hi_edges, hi_colors = [], [], []
    K = len(segments)
    for k, (a, b) in enumerate(segments):
        if b <= a:
            continue
        h = (k / max(1, K)) * 0.85 + 0.55
        h = h % 1.0
        rgb = colorsys.hsv_to_rgb(h, 0.85, 0.95)
        for j in range(a, b):
            idx = len(hi_pts)
            hi_pts.append(pts_u[j])
            hi_pts.append(pts_u[j + 1])
            hi_edges.append([idx, idx + 1])
            hi_colors.append(rgb)
    if not hi_edges:
        return
    cn_hi = ps.register_curve_network(
        nm_hi,
        np.asarray(hi_pts, dtype=np.float64),
        np.asarray(hi_edges, dtype=np.int32),
        radius=0.0018)
    cn_hi.add_color_quantity('match_color',
                             np.asarray(hi_colors, dtype=np.float64),
                             defined_on='edges', enabled=True)


def _procrustes_seg_net_name(sid: int, idx: int) -> str:
    """One polyscope curve-network per Procrustes segment so each Step
    output is its own toggle-able / removable layer."""
    return f'procrustes_hi_seg{sid:04d}_{idx:03d}'


def _procrustes_seg_sample_pc_name(sid: int, idx: int) -> str:
    """Companion point cloud name (uniform samples along the segment)."""
    return f'procrustes_hi_seg{sid:04d}_{idx:03d}_samples'


def _stitch_save_path(timestamped: bool = False) -> Path:
    """Default save location: next to the input yarn npz, named by curve sid."""
    vs = _view_state
    base = Path(vs.get('npz_path', 'yarn.npz')).resolve().parent
    sid = int(vs['curves'][vs['curve_idx']]['sid']) if vs.get('curves') else 0
    if timestamped:
        import time
        return base / f'stitches_sid{sid:04d}_{time.strftime("%Y%m%d_%H%M%S")}.npz'
    return base / f'stitches_sid{sid:04d}_latest.npz'


def _save_stitch_extraction() -> None:
    """Save the CURRENT Procrustes stitch extraction — the detected segments,
    the template span, the params, and the resampled curve they index into —
    to an npz (stitch_extraction_v1) next to the input yarn.  Always refreshes
    stitches_sid<sid>_latest.npz and also writes a timestamped backup, so it
    can be reloaded with 'Load stitches'."""
    vs = _view_state
    if not vs.get('curves'):
        print('[stitch-save] no curve loaded'); return
    cur = vs['curves'][vs['curve_idx']]
    pts_u = np.asarray(cur['res']['pts_u'], np.float32)
    segs = np.asarray(vs.get('procrustes_segments') or [],
                      np.int64).reshape(-1, 2).astype(np.int32)
    costs = np.asarray(vs.get('procrustes_costs') or [], np.float32)
    payload = dict(
        fmt=np.asarray('stitch_extraction_v1'),
        sid=np.int64(int(cur['sid'])),
        curve_idx=np.int64(int(vs['curve_idx'])),
        n_pts=np.int64(len(pts_u)),
        pts_u=pts_u,
        segments=segs,
        costs=costs,
        template_idx0=np.int64(int(vs['template_idx0'])),
        template_idx1=np.int64(int(vs['template_idx1'])),
        procrustes_threshold=np.float64(float(vs['procrustes_threshold'])),
        procrustes_n_samples=np.int64(int(vs['procrustes_n_samples'])),
        procrustes_length_variation=np.float64(
            float(vs['procrustes_length_variation'])),
    )
    p_latest = _stitch_save_path(False)
    for p in (p_latest, _stitch_save_path(True)):
        np.savez(p, **payload)
    print(f'[stitch-save] wrote {p_latest} (+timestamped): {len(segs)} stitches, '
          f'template=[{int(vs["template_idx0"])},{int(vs["template_idx1"])}], '
          f'tau={float(vs["procrustes_threshold"]):.1f}, {len(pts_u)} curve pts')


def _export_template_segment() -> None:
    """Export ONLY the user-selected orange template span (the current curve
    between the picked start/end) as a standalone polyline: a Wavefront OBJ
    for rendering plus a raw Nx3 float32 .npy, written next to the input yarn
    npz.  This is the template used to seed the template-fitting figure."""
    vs = _view_state
    if not vs.get('curves'):
        print('[tpl-export] no curve loaded'); return
    cur = vs['curves'][vs['curve_idx']]
    sid = int(cur['sid'])
    pts_u = np.asarray(cur['res']['pts_u'], np.float64)
    n = len(pts_u)
    i0, i1 = int(vs['template_idx0']), int(vs['template_idx1'])
    if not (0 <= i0 < i1 < n):
        print(f'[tpl-export] no valid span (start={i0}, end={i1}); '
              f'pick start and end first'); return
    seg = pts_u[i0:i1 + 1]
    arclen = float(np.sum(np.linalg.norm(np.diff(seg, axis=0), axis=1)))
    base = Path(vs.get('npz_path', 'yarn.npz')).resolve().parent
    stem = f'template_sid{sid:04d}_{i0}_{i1}'
    obj_path = base / f'{stem}.obj'
    npy_path = base / f'{stem}.npy'
    export_curves_obj([seg], obj_path, seg_ids=np.asarray([sid]), split=False)
    np.save(npy_path, seg.astype(np.float32))
    # Pair the CURRENT camera with this segment 1:1 — the render reuses exactly
    # the angle you exported it from (only look/up matter; framing auto-fits).
    cam_path = base / f'{stem}.camera.txt'
    try:
        with open(cam_path, 'w') as fh:
            fh.write(ps.get_view_as_json())
        cam_line = f'\n  {cam_path}'
    except Exception as e:
        cam_line = f'\n  [warn] camera save failed: {e}'
    print(f'[tpl-export] wrote:\n  {obj_path}\n  {npy_path}{cam_line}\n'
          f'  {len(seg)} pts, span [{i0},{i1}], arclen={arclen:.1f} voxels')


def _load_stitch_extraction(path=None) -> bool:
    """Load a saved stitch extraction (stitch_extraction_v1) and restore the
    segments + template + params, then redraw.  Defaults to the _latest file
    next to the input yarn.  Returns True on success."""
    vs = _view_state
    path = Path(path) if path is not None else _stitch_save_path(False)
    if not path.exists():
        print(f'[stitch-load] not found: {path}'); return False
    d = np.load(path, allow_pickle=False)
    if str(d['fmt']) != 'stitch_extraction_v1' if 'fmt' in d.files else True:
        print(f"[stitch-load] not a stitch_extraction_v1 npz: {path}"); return False
    segs = [(int(a), int(b))
            for a, b in np.asarray(d['segments']).reshape(-1, 2)]
    vs['template_idx0'] = int(d['template_idx0'])
    vs['template_idx1'] = int(d['template_idx1'])
    vs['procrustes_threshold'] = float(d['procrustes_threshold'])
    vs['procrustes_n_samples'] = int(d['procrustes_n_samples'])
    vs['procrustes_length_variation'] = float(d['procrustes_length_variation'])
    vs['procrustes_segments'] = segs
    vs['procrustes_costs'] = (np.asarray(d['costs']).astype(float).tolist()
                              if 'costs' in d.files else [])
    vs['procrustes_n_detected'] = len(segs)
    vs['procrustes_n_fwd'] = len(segs)
    vs['procrustes_n_bwd'] = 0
    vs['match_kind'] = 'procrustes'
    vs['match_mode'] = (f'Procrustes AR (loaded {len(segs)} stitches, '
                        f'tau={float(d["procrustes_threshold"]):.1f})')
    if segs:                       # set AR frontiers so Step Fwd/Bwd can continue
        vs['procrustes_s_fwd'] = max(b for _, b in segs)
        vs['procrustes_s_bwd'] = min(a for a, _ in segs)
        med = float(np.median([b - a + 1 for a, b in segs]))
        vs['procrustes_L_fwd'] = med
        vs['procrustes_L_bwd'] = med
    vs['procrustes_undo_fwd'] = []
    vs['procrustes_undo_bwd'] = []
    if vs.get('curves'):           # sanity: do the saved indices fit this curve?
        n_now = len(vs['curves'][vs['curve_idx']]['res']['pts_u'])
        n_saved = int(d['n_pts']) if 'n_pts' in d.files else n_now
        if n_now != n_saved:
            print(f'[stitch-load] WARNING: current curve has {n_now} pts but the '
                  f'saved extraction was on {n_saved} pts — indices may not align '
                  f'(reload the same --npz with the same --ds).')
    print(f'[stitch-load] restored {len(segs)} stitches from {path}')
    _refresh_match_view()
    return True


def _remove_all_procrustes_segments() -> None:
    """Remove every per-segment Procrustes curve network AND its companion
    sample point cloud + the inter-segment correspondence lines, then
    clear the tracked-name state."""
    vs = _view_state
    for nm in vs.get('procrustes_seg_names', []) or []:
        if ps.has_curve_network(nm):
            ps.remove_curve_network(nm)
    for nm in vs.get('procrustes_seg_sample_names', []) or []:
        if ps.has_point_cloud(nm):
            ps.remove_point_cloud(nm)
    corr_nm = vs.get('procrustes_seg_corr_name')
    if corr_nm and ps.has_curve_network(corr_nm):
        ps.remove_curve_network(corr_nm)
    vs['procrustes_seg_names']        = []
    vs['procrustes_seg_sample_names'] = []
    vs['procrustes_seg_corr_name']    = None


def _draw_procrustes_segments_individual(pts_u: np.ndarray,
                                         segments: list[tuple[int, int]],
                                         sid: int) -> None:
    """Register each (a, b) as its own curve network plus a companion
    point cloud of N uniformly-arclength-sampled points (same color).
    Color rule mirrors the original `_draw_segment_highlights` rainbow:
    hue = k/K * 0.85 + 0.55 so the K segments span the rainbow in
    registration order. The sample clouds respect the
    'procrustes_show_samples' toggle for initial visibility."""
    vs = _view_state
    _remove_all_procrustes_segments()
    new_names: list[str] = []
    new_sample_names: list[str] = []
    K = len(segments)
    N_VIZ = 10                                          # fixed sample count
    show_samples = bool(vs.get('procrustes_show_samples', True))
    show_corr    = bool(vs.get('procrustes_show_corr', True))
    order = np.arange(N_VIZ, dtype=np.float64)          # within-segment index
    sample_clouds: list[np.ndarray] = []                # for corr-line wiring
    for k, (a, b) in enumerate(segments):
        if b <= a:
            sample_clouds.append(np.empty((0, 3)))
            continue
        h = ((k / max(1, K)) * 0.85 + 0.55) % 1.0
        rgb = colorsys.hsv_to_rgb(h, 0.85, 0.95)
        seg_pts   = pts_u[a:b + 1]
        seg_edges = np.stack([np.arange(len(seg_pts) - 1),
                              np.arange(1, len(seg_pts))],
                             axis=1).astype(np.int32)
        nm = _procrustes_seg_net_name(sid, k)
        cn = ps.register_curve_network(nm, seg_pts, seg_edges, radius=0.0018)
        cn.set_color(rgb)
        new_names.append(nm)

        # Companion: 10 uniformly-arclength samples along this segment,
        # colored by within-segment order using a shared cmap so all
        # segments use the same color scale (start → end = same gradient).
        sample_pts = equispaced_resample(seg_pts, N_VIZ)
        nm_pc = _procrustes_seg_sample_pc_name(sid, k)
        pc = ps.register_point_cloud(nm_pc, sample_pts, radius=0.0050)
        pc.add_scalar_quantity('order', order, enabled=True,
                               cmap='viridis',
                               vminmax=(0.0, float(N_VIZ - 1)))
        pc.set_enabled(show_samples)
        new_sample_names.append(nm_pc)
        sample_clouds.append(sample_pts)
    vs['procrustes_seg_names']        = new_names
    vs['procrustes_seg_sample_names'] = new_sample_names

    # ── Correspondence lines: sample i of seg k ↔ sample i of seg k+1 ──
    # Single curve network so the whole bundle toggles as one unit. Edges
    # carry a within-pair 'order' scalar so each correspondence shares the
    # viridis color of its two endpoints (matching the sample point colors).
    corr_nm = f'procrustes_hi_seg{sid:04d}_corr'
    if ps.has_curve_network(corr_nm):
        ps.remove_curve_network(corr_nm)

    nodes_chunks: list[np.ndarray] = []
    edges_list:   list[tuple[int, int]] = []
    edge_order:   list[float] = []
    base = 0
    for k in range(len(sample_clouds) - 1):
        a_pts = sample_clouds[k]
        b_pts = sample_clouds[k + 1]
        if len(a_pts) != N_VIZ or len(b_pts) != N_VIZ:
            continue
        nodes_chunks.append(a_pts)
        nodes_chunks.append(b_pts)
        for i in range(N_VIZ):
            edges_list.append((base + i, base + N_VIZ + i))
            edge_order.append(float(i))
        base += 2 * N_VIZ

    if edges_list:
        nodes = np.concatenate(nodes_chunks, axis=0)
        edges = np.array(edges_list, dtype=np.int32)
        cn_corr = ps.register_curve_network(corr_nm, nodes, edges,
                                            radius=0.0008)
        cn_corr.add_scalar_quantity('order',
                                    np.asarray(edge_order, dtype=np.float64),
                                    defined_on='edges', enabled=True,
                                    cmap='viridis',
                                    vminmax=(0.0, float(N_VIZ - 1)))
        cn_corr.set_enabled(show_corr)
        vs['procrustes_seg_corr_name'] = corr_nm
    else:
        vs['procrustes_seg_corr_name'] = None


def _set_procrustes_samples_visible(visible: bool) -> None:
    """Toggle visibility of every per-segment sample point cloud at once
    without re-registering."""
    vs = _view_state
    vs['procrustes_show_samples'] = visible
    for nm in vs.get('procrustes_seg_sample_names', []) or []:
        if ps.has_point_cloud(nm):
            ps.get_point_cloud(nm).set_enabled(visible)


def _set_procrustes_corr_visible(visible: bool) -> None:
    """Toggle visibility of the inter-segment correspondence-line bundle."""
    vs = _view_state
    vs['procrustes_show_corr'] = visible
    nm = vs.get('procrustes_seg_corr_name')
    if nm and ps.has_curve_network(nm):
        ps.get_curve_network(nm).set_enabled(visible)


# ── Kabsch algorithm visualization (independent of procrustes match) ─────────
# Animates the alignment of two segments' 10-sample point clouds.
# Phases: corr_draw (correspondences appear one by one) → align (src eases
# from original to Kabsch-aligned position) → idle (residuals = remaining
# corr-line lengths). Uses its own polyscope entities; does not touch the
# procrustes_segments / sample / corr layers.
_kabsch_viz: dict = {
    'src_seg_idx':       0,
    'tgt_seg_idx':       1,
    'src_pts_orig':      None,    # (N, 3)
    'src_pts_current':   None,    # (N, 3) — currently displayed positions
    'src_pts_aligned':   None,    # (N, 3) after Kabsch
    'tgt_pts':           None,    # (N, 3)
    'R':                 None,    # (3, 3)
    't':                 None,    # (3,)
    'ssd':               None,    # float
    'rms':               None,    # float
    'phase':             'idle',  # 'idle' | 'corr_draw' | 'align'
    'phase_t_start':     0.0,
    'phase_dur':         1.5,     # seconds per animated phase
    'sequence':          None,    # optional list[(phase, dur)] to chain phases
    'sequence_idx':      0,
    'names': {
        'src_pc':    'kabsch_src_samples',
        'tgt_pc':    'kabsch_tgt_samples',
        'src_curve': 'kabsch_src_curve',
        'tgt_curve': 'kabsch_tgt_curve',
        'corr':      'kabsch_corr_lines',
    },
}


# ── Kabsch GIF export state (defer-by-one-frame screenshot capture) ──────────
_kabsch_export = {
    'active':              False,
    'run_id':              '',          # 'YYYYMMDD_HHMMSS' stamp
    'pending_screenshot':  None,        # path to drain on next callback
    'frames':              [],          # list of png paths captured this run
    'frame_counter':       0,
    'combine_after_drain': False,       # True when sequence ends; combine on drain
    'fps':                 15.0,        # GIF playback fps
}


def _kabsch_compute(A: np.ndarray, B: np.ndarray):
    """Pure Kabsch: align A onto B (rigid: rotation + translation, no scale,
    no reflection). Returns (R, t, A_aligned, ssd, rms_per_pt)."""
    A_mean = A.mean(axis=0)
    B_mean = B.mean(axis=0)
    Ac = A - A_mean
    Bc = B - B_mean
    H = Ac.T @ Bc
    U, _, Vt = np.linalg.svd(H)
    D = np.eye(3)
    if np.linalg.det(Vt.T @ U.T) < 0.0:
        D[2, 2] = -1.0
    R = Vt.T @ D @ U.T
    t = B_mean - R @ A_mean
    A_aligned = A @ R.T + t
    diff = A_aligned - B
    ssd = float((diff * diff).sum())
    rms = float(np.sqrt(ssd / max(1, len(A))))
    return R, t, A_aligned, ssd, rms


def _kabsch_clear_visuals() -> None:
    for nm in _kabsch_viz['names'].values():
        if ps.has_point_cloud(nm):
            ps.remove_point_cloud(nm)
        if ps.has_curve_network(nm):
            ps.remove_curve_network(nm)


def _kabsch_register_corr_lines(n_visible: int, src_pts: np.ndarray) -> None:
    """Register the first ``n_visible`` correspondence edges (sample i of
    src ↔ sample i of tgt). Called repeatedly during corr_draw and align."""
    kab = _kabsch_viz
    nm = kab['names']['corr']
    if ps.has_curve_network(nm):
        ps.remove_curve_network(nm)
    if n_visible <= 0 or kab['tgt_pts'] is None:
        return
    N_total = len(src_pts)
    n_visible = min(n_visible, N_total)
    nodes = np.vstack([src_pts[:n_visible], kab['tgt_pts'][:n_visible]])
    edges = np.column_stack(
        [np.arange(n_visible), n_visible + np.arange(n_visible)]
    ).astype(np.int32)
    cn = ps.register_curve_network(nm, nodes, edges, radius=0.0010)
    cn.add_scalar_quantity('order',
                           np.arange(n_visible, dtype=np.float64),
                           defined_on='edges', enabled=True,
                           cmap='viridis',
                           vminmax=(0.0, float(N_total - 1)))


def _kabsch_register_initial(src_pts: np.ndarray, tgt_pts: np.ndarray,
                             src_full: np.ndarray, tgt_full: np.ndarray) -> None:
    n = _kabsch_viz['names']
    N = len(src_pts)
    order = np.arange(N, dtype=np.float64)

    pc_s = ps.register_point_cloud(n['src_pc'], src_pts, radius=0.0070)
    pc_s.add_scalar_quantity('order', order, enabled=True,
                             cmap='viridis', vminmax=(0.0, float(N - 1)))

    pc_t = ps.register_point_cloud(n['tgt_pc'], tgt_pts, radius=0.0070)
    pc_t.add_scalar_quantity('order', order, enabled=True,
                             cmap='viridis', vminmax=(0.0, float(N - 1)))

    if len(src_full) >= 2:
        e = np.stack([np.arange(len(src_full) - 1),
                      np.arange(1, len(src_full))], axis=1).astype(np.int32)
        cn = ps.register_curve_network(n['src_curve'], src_full, e,
                                       radius=0.0008)
        cn.set_color((0.95, 0.55, 0.20))   # orange
    if len(tgt_full) >= 2:
        e = np.stack([np.arange(len(tgt_full) - 1),
                      np.arange(1, len(tgt_full))], axis=1).astype(np.int32)
        cn = ps.register_curve_network(n['tgt_curve'], tgt_full, e,
                                       radius=0.0008)
        cn.set_color((0.20, 0.55, 0.95))   # blue


def _kabsch_setup(src_idx: int, tgt_idx: int) -> None:
    vs  = _view_state
    kab = _kabsch_viz
    if not vs['curves']:
        return
    cur = vs['curves'][vs['curve_idx']]
    pts_u = cur['res']['pts_u']
    segments = vs.get('procrustes_segments') or []
    if len(segments) < 2:
        print('Kabsch viz: need at least 2 procrustes segments '
              '(run Step Fwd/Bwd or Match Procrustes first).')
        return
    src_idx = max(0, min(src_idx, len(segments) - 1))
    tgt_idx = max(0, min(tgt_idx, len(segments) - 1))
    if src_idx == tgt_idx:
        print('Kabsch viz: src and tgt must be different segments.')
        return

    a, b = segments[src_idx]
    c, d = segments[tgt_idx]
    src_full = pts_u[a:b + 1]
    tgt_full = pts_u[c:d + 1]
    if len(src_full) < 2 or len(tgt_full) < 2:
        print('Kabsch viz: degenerate segment.')
        return
    src_pts = equispaced_resample(src_full, 10)
    tgt_pts = equispaced_resample(tgt_full, 10)
    R, t_, A_aligned, ssd, rms = _kabsch_compute(src_pts, tgt_pts)

    kab['src_seg_idx']     = src_idx
    kab['tgt_seg_idx']     = tgt_idx
    kab['src_pts_orig']    = src_pts
    kab['src_pts_current'] = src_pts.copy()
    kab['src_pts_aligned'] = A_aligned
    kab['tgt_pts']         = tgt_pts
    kab['R']               = R
    kab['t']               = t_
    kab['ssd']             = ssd
    kab['rms']             = rms
    kab['phase']           = 'idle'
    kab['sequence']        = None

    _kabsch_clear_visuals()
    _kabsch_register_initial(src_pts, tgt_pts, src_full, tgt_full)
    print(f'Kabsch viz setup: src=seg{src_idx} ({a}-{b})  '
          f'tgt=seg{tgt_idx} ({c}-{d})  '
          f'predicted SSD={ssd:.3f}  rms={rms:.3f}')


def _kabsch_start_phase(phase: str, dur: float = None) -> None:
    kab = _kabsch_viz
    if dur is not None:
        kab['phase_dur'] = float(dur)
    kab['phase'] = phase
    kab['phase_t_start'] = time.perf_counter()


def _kabsch_run_sequence(seq: list) -> None:
    """Chain phases: e.g. [('corr_draw', 1.5), ('align', 1.5)]."""
    kab = _kabsch_viz
    if kab['src_pts_orig'] is None:
        print('Kabsch viz: click Setup first.')
        return
    kab['sequence']     = list(seq)
    kab['sequence_idx'] = 0
    if seq:
        ph, dur = seq[0]
        _kabsch_start_phase(ph, dur)


def _kabsch_reset_pos() -> None:
    """Return src samples to original position; clear corr lines."""
    kab = _kabsch_viz
    if kab['src_pts_orig'] is None:
        return
    kab['phase']           = 'idle'
    kab['sequence']        = None
    kab['src_pts_current'] = kab['src_pts_orig'].copy()
    nm = kab['names']['src_pc']
    if ps.has_point_cloud(nm):
        ps.get_point_cloud(nm).update_point_positions(kab['src_pts_orig'])
    nm_corr = kab['names']['corr']
    if ps.has_curve_network(nm_corr):
        ps.remove_curve_network(nm_corr)


def _kabsch_tick() -> None:
    """Advance the active phase. Called every UI callback frame.
    If a GIF export is active, queues a screenshot for the next frame
    after each phase update (deferred-by-one-frame so the capture
    reflects the just-rendered state)."""
    kab = _kabsch_viz
    ke  = _kabsch_export
    if kab['phase'] == 'idle' or kab['src_pts_orig'] is None:
        return
    elapsed = time.perf_counter() - kab['phase_t_start']
    finished = False

    if kab['phase'] == 'corr_draw':
        N = len(kab['src_pts_orig'])
        per = max(1e-3, kab['phase_dur'] / N)
        n_show = min(N, int(elapsed / per) + 1)
        _kabsch_register_corr_lines(n_show, kab['src_pts_current'])
        if elapsed >= kab['phase_dur']:
            finished = True

    elif kab['phase'] == 'align':
        t = min(elapsed / max(1e-3, kab['phase_dur']), 1.0)
        e = t * t * (3.0 - 2.0 * t)        # smoothstep
        interp = (1.0 - e) * kab['src_pts_orig'] + e * kab['src_pts_aligned']
        kab['src_pts_current'] = interp
        nm_s = kab['names']['src_pc']
        if ps.has_point_cloud(nm_s):
            ps.get_point_cloud(nm_s).update_point_positions(interp)
        nm_c = kab['names']['corr']
        if ps.has_curve_network(nm_c):
            N = len(interp)
            nodes = np.vstack([interp, kab['tgt_pts']])
            ps.get_curve_network(nm_c).update_node_positions(nodes)
        if elapsed >= kab['phase_dur']:
            finished = True

    # Queue per-frame screenshot for export (drained on next callback).
    if ke['active']:
        ke['pending_screenshot'] = os.path.join(
            _RENDER_DIR, f'kabsch_{ke["run_id"]}',
            f'frame_{ke["frame_counter"]:04d}.png')
        ke['frame_counter'] += 1

    if finished:
        kab['phase'] = 'idle'
        if kab['sequence'] is not None:
            kab['sequence_idx'] += 1
            if kab['sequence_idx'] < len(kab['sequence']):
                ph, dur = kab['sequence'][kab['sequence_idx']]
                _kabsch_start_phase(ph, dur)
            else:
                kab['sequence']     = None
                kab['sequence_idx'] = 0
                # Final residual report
                if kab['src_pts_current'] is not None and kab['tgt_pts'] is not None:
                    diff = kab['src_pts_current'] - kab['tgt_pts']
                    final_ssd = float((diff * diff).sum())
                    final_rms = float(np.sqrt(final_ssd / max(1, len(diff))))
                    print(f'Kabsch viz done: final SSD={final_ssd:.3f}  '
                          f'rms={final_rms:.3f}')
                # Sequence finished — combine the export GIF on next drain.
                if ke['active']:
                    ke['combine_after_drain'] = True


def _kabsch_export_start() -> None:
    """Begin a Kabsch GIF capture: stamps run_id, captures the iter-0
    frame synchronously (current view), then chains corr_draw → align,
    queueing a screenshot per frame. The GIF is built when the chain
    ends (see drain block at top of _ui_callback)."""
    kab = _kabsch_viz
    if kab['src_pts_orig'] is None:
        print('Kabsch export: click Setup first.')
        return
    ke = _kabsch_export
    ke['active']              = True
    ke['run_id']              = time.strftime('%Y%m%d_%H%M%S')
    ke['frames']              = []
    ke['frame_counter']       = 0
    ke['combine_after_drain'] = False

    # Make sure we start from the unaligned state so the GIF shows the
    # full motion (Setup leaves things at orig anyway, but be safe).
    _kabsch_reset_pos()

    # Capture iter-0 (current rendered frame: src at orig, no corr lines).
    iter0_path = os.path.join(
        _RENDER_DIR, f'kabsch_{ke["run_id"]}',
        f'frame_{ke["frame_counter"]:04d}.png')
    try:
        os.makedirs(os.path.dirname(iter0_path), exist_ok=True)
        ps.screenshot(iter0_path, transparent_bg=True)
        _postprocess_transparent(iter0_path)
        ke['frames'].append(iter0_path)
        ke['frame_counter'] += 1
        print(f'Kabsch export start → {iter0_path}')
    except Exception as e:
        print(f'  [warn] kabsch start screenshot failed: {e}')

    # Chain the standard sequence so per-frame queueing kicks in.
    _kabsch_run_sequence([
        ('corr_draw', float(kab['phase_dur'])),
        ('align',     float(kab['phase_dur'])),
    ])


def _kabsch_export_drain_and_maybe_combine() -> None:
    """Drain a queued screenshot (captures the just-rendered frame).
    When the sequence has finished AND there's nothing left pending,
    combine all captured frames into a GIF."""
    ke = _kabsch_export
    if ke['pending_screenshot'] is not None:
        path = ke['pending_screenshot']
        ke['pending_screenshot'] = None
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            ps.screenshot(path, transparent_bg=True)
            _postprocess_transparent(path)
            ke['frames'].append(path)
        except Exception as e:
            print(f'  [warn] kabsch screenshot failed: {e}')

    if ke['combine_after_drain'] and ke['pending_screenshot'] is None:
        ke['combine_after_drain'] = False
        if ke['frames']:
            try:
                from make_gif_from_frames import build_gif as _build_gif
                gif_path = os.path.join(
                    _RENDER_DIR, f'kabsch_{ke["run_id"]}.gif')
                n = _build_gif(ke['frames'], gif_path,
                               fps=float(ke.get('fps', 15.0)))
                print(f'Kabsch GIF saved → {gif_path}  '
                      f'({n} frames, disposal=2)')
            except Exception as e:
                print(f'  [warn] kabsch GIF combine failed: {e}')
        ke['frames']        = []
        ke['active']        = False
        ke['frame_counter'] = 0


def _refresh_match_view() -> None:
    """Show match scores as scalar on curve and highlight top-K windows.

    Pearson and HMM share the ``match_hi`` highlight layer (toggling between
    them overwrites the previous). Procrustes uses its own ``procrustes_hi``
    layer so it coexists visually with whichever Pearson/HMM result is shown.
    """
    vs = _view_state
    if not vs['curves']:
        return
    cur = vs['curves'][vs['curve_idx']]
    sid = cur['sid']
    res = cur['res']
    n = len(res['pts_u'])
    nm_curve = _curve_net_name(sid)

    # Procrustes path: separate layer; don't touch pearson/hmm highlights.
    # Each segment becomes its own curve network so Step Fwd/Bwd outputs
    # can be independently toggled/inspected in polyscope.
    if vs['match_kind'] == 'procrustes':
        # Remove the legacy single-network layer if it still exists from an
        # earlier session.
        nm_proc_legacy = _procrustes_hi_net_name(sid)
        if ps.has_curve_network(nm_proc_legacy):
            ps.remove_curve_network(nm_proc_legacy)
        segments = vs.get('procrustes_segments') or []
        if not segments or not ps.has_curve_network(nm_curve):
            _remove_all_procrustes_segments()
            return
        _draw_procrustes_segments_individual(res['pts_u'], segments, sid)
        return

    nm_hi = _matchhi_net_name(sid)
    if ps.has_curve_network(nm_hi):
        ps.remove_curve_network(nm_hi)

    # HMM path: draw stored segments directly, no thresholding / top-K.
    if vs['match_kind'] == 'hmm':
        segments = vs.get('hmm_segments') or []
        if not segments or not ps.has_curve_network(nm_curve):
            return
        flag = np.zeros(n, dtype=np.float64)
        for a, b in segments:
            flag[a:b + 1] = 1.0
        ps.get_curve_network(nm_curve).add_scalar_quantity(
            'match_score', flag, enabled=True,
            cmap='coolwarm', vminmax=(0.0, 1.0))
        _draw_segment_highlights(res['pts_u'], segments, nm_hi)
        return

    scores = vs['match_scores']
    m = vs['match_template_len']
    if scores is None or m == 0 or not ps.has_curve_network(nm_curve):
        return

    padded = np.full(n, -1.0, dtype=np.float64)
    padded[:len(scores)] = np.where(np.isfinite(scores), scores, -1.0)
    cn = ps.get_curve_network(nm_curve)
    cn.add_scalar_quantity('match_score', padded, enabled=True,
                           cmap='coolwarm', vminmax=(-1.0, 1.0))

    scores_for_peaks = np.where(np.isfinite(scores), scores, -np.inf)
    peaks, _ = find_peaks(scores_for_peaks, height=float(vs['match_threshold']),
                          distance=max(1, m // 2))
    if len(peaks) == 0:
        return

    order = np.argsort(scores[peaks])[::-1][:int(vs['match_top'])]
    peaks = peaks[order]

    segments = [(int(p), min(int(p) + m - 1, n - 1)) for p in peaks]
    _draw_segment_highlights(res['pts_u'], segments, nm_hi)


def _curve_index_for_struct(struct_name: str):
    """Index into vs['curves'] whose curve-network OR sample point cloud is
    named ``struct_name``, else None."""
    for j, c in enumerate(_view_state['curves']):
        sid = c['sid']
        if struct_name in (_curve_net_name(sid), _samples_pc_name(sid)):
            return j
    return None


def _switch_active_curve(j: int) -> None:
    """Make curve j active, clearing the previous curve's template markers /
    match overlays (mirrors the 'Curve' combo handler)."""
    vs = _view_state
    if j == vs['curve_idx']:
        return
    sid_old = vs['curves'][vs['curve_idx']]['sid']
    for nm in (_matchhi_net_name(sid_old), _procrustes_hi_net_name(sid_old),
               _template_net_name(sid_old)):
        if ps.has_curve_network(nm):
            ps.remove_curve_network(nm)
    _remove_all_procrustes_segments()
    for nm in (_start_marker_name(sid_old), _end_marker_name(sid_old)):
        if ps.has_point_cloud(nm):
            ps.remove_point_cloud(nm)
    vs['curve_idx']          = j
    vs['template_idx0']      = -1
    vs['template_idx1']      = -1
    vs['match_scores']       = None
    vs['match_template_len'] = 0
    vs['match_kind']         = ''
    vs['match_mode']         = ''
    vs['hmm_segments']       = None
    _procrustes_reset_state()


def _consume_pick() -> bool:
    """If pick_mode is active and the user has clicked a curve — either its
    rainbow/κ tube (curve-network) OR its sample point cloud — capture the
    node index and store it as the template start/end. Clicking a different
    shown curve switches to it first."""
    vs = _view_state
    if vs['pick_mode'] is None or not vs['curves']:
        return False
    if not ps.have_selection():
        return False
    try:
        sel = ps.get_selection()
    except Exception:
        return False
    # Modern polyscope → PickResult (.is_hit/.structure_name/.local_index/
    # .structure_data); older → (name, index) tuple.
    if isinstance(sel, tuple):
        struct_name, local_idx, sdata = sel[0], sel[1], {}
    else:
        if hasattr(sel, 'is_hit') and not sel.is_hit:
            return False
        struct_name = getattr(sel, 'structure_name', None)
        local_idx   = getattr(sel, 'local_index', None)
        sdata       = getattr(sel, 'structure_data', {}) or {}
    if struct_name is None or local_idx is None:
        return False

    j = _curve_index_for_struct(struct_name)
    if j is None:                      # clicked something that isn't a curve
        return False
    _switch_active_curve(j)
    cur = vs['curves'][vs['curve_idx']]
    n = len(cur['res']['pts_u'])

    # Curve-network nodes index directly into pts_u; an edge pick (edge e
    # joins nodes e, e+1) snaps to its start node; point-cloud picks are
    # already node indices.
    et  = str(sdata.get('element_type', '')).lower()
    idx = int(sdata.get('index', local_idx))
    if et == 'edge':
        idx = min(idx, n - 2)
    idx = int(np.clip(idx, 0, n - 1))

    mode = vs['pick_mode']
    if mode == 'start':
        vs['template_idx0'] = idx
    elif mode == 'match_start':
        vs['match_start_idx'] = idx
    else:
        vs['template_idx1'] = idx
    vs['pick_mode'] = None
    print(f'  [pick {mode}] curve seg{cur["sid"]:04d}  node {idx}/{n - 1}')
    return True


def _ui_callback() -> None:
    # One-shot: re-apply saved camera view on the first callback frame.
    # ps.show() resets to the home view, so loading earlier does not stick.
    if _pending_camera_load[0]:
        _load_camera_view()
        _pending_camera_load[0] = False

    # One-shot: auto-load a saved stitch extraction (--load_stitches).  Done
    # here (not in main) so the curve networks are already registered.
    if _pending_stitch_load[0] is not None:
        _p = _pending_stitch_load[0]
        _load_stitch_extraction(None if _p == '__latest__' else _p)
        _pending_stitch_load[0] = None

    # Drain any queued Kabsch-export screenshot (captures the previously
    # rendered frame), and combine the GIF if the export sequence finished.
    _kabsch_export_drain_and_maybe_combine()

    # Advance the Kabsch viz animation if a phase is active.
    _kabsch_tick()

    vs = _view_state
    if not vs['curves']:
        return
    cur = vs['curves'][vs['curve_idx']]
    sid = cur['sid']
    res = cur['res']

    psim.TextUnformatted(f'curve seg={sid}  L={res["L"]:.1f}  '
                         f'pts={len(res["s"])}  ds_eff={res["ds_eff"]:.2f}')

    # ── Camera & Screenshot ──────────────────────────────────────────────────
    psim.Separator()
    psim.TextUnformatted(f'camera: {_CAMERA_PATH}')
    if psim.Button('Save PNG (transparent)'):
        _save_transparent_screenshot()
    psim.SameLine()
    if psim.Button('Save Camera'):
        _save_camera_view()
    psim.SameLine()
    if psim.Button('Load Camera'):
        _load_camera_view()
    if _camera_msg[0]:
        psim.SameLine()
        psim.TextUnformatted(_camera_msg[0])

    # ── Export curves as OBJ ─────────────────────────────────────────────────
    raw_curves = vs.get('raw_curves')
    if raw_curves:
        out_dir  = vs.get('out_dir') or Path('.')
        npz_path = vs.get('npz_path')
        stem     = (Path(npz_path).stem if npz_path else 'curves') + '_curves'
        if psim.Button('Export OBJ (combined)'):
            p = out_dir / f'{stem}.obj'
            written = export_curves_obj(raw_curves, p,
                                        seg_ids=vs.get('raw_seg_ids'),
                                        split=False)
            _obj_export_msg[0] = f'saved → {written[0].name}'
            for w in written:
                print(f'  OBJ saved → {w}')
        psim.SameLine()
        if psim.Button('Export OBJ (split per curve)'):
            p = out_dir / f'{stem}.obj'
            written = export_curves_obj(raw_curves, p,
                                        seg_ids=vs.get('raw_seg_ids'),
                                        split=True)
            _obj_export_msg[0] = f'saved {len(written)} files → {out_dir}'
            for w in written:
                print(f'  OBJ saved → {w}')
        if _obj_export_msg[0]:
            psim.SameLine()
            psim.TextUnformatted(_obj_export_msg[0])

    # Curve selector
    if len(vs['curves']) > 1:
        names = [f'seg{c["sid"]:04d}' for c in vs['curves']]
        ch, vs['curve_idx'] = psim.Combo('Curve', vs['curve_idx'], names)
        if ch:
            sid_old = cur['sid']
            for nm in (_matchhi_net_name(sid_old),
                       _procrustes_hi_net_name(sid_old),
                       _template_net_name(sid_old)):
                if ps.has_curve_network(nm):
                    ps.remove_curve_network(nm)
            _remove_all_procrustes_segments()
            for nm in (_start_marker_name(sid_old), _end_marker_name(sid_old)):
                if ps.has_point_cloud(nm):
                    ps.remove_point_cloud(nm)
            vs['template_idx0'] = -1
            vs['template_idx1'] = -1
            vs['match_scores'] = None
            vs['match_template_len'] = 0
            vs['match_kind'] = ''
            vs['match_mode'] = ''
            vs['hmm_segments'] = None
            _procrustes_reset_state()

    # ── Template matching ─────────────────────────────────────────────────────
    psim.Separator()
    psim.TextUnformatted('Template matching:')

    if psim.Button('Pick start'):
        vs['pick_mode'] = 'start'
    psim.SameLine()
    if psim.Button('Pick end'):
        vs['pick_mode'] = 'end'
    psim.SameLine()
    if psim.Button('Clear'):
        sid_cur = cur['sid']
        for nm in (_template_net_name(sid_cur),
                   _matchhi_net_name(sid_cur),
                   _procrustes_hi_net_name(sid_cur)):
            if ps.has_curve_network(nm):
                ps.remove_curve_network(nm)
        for nm in (_start_marker_name(sid_cur), _end_marker_name(sid_cur)):
            if ps.has_point_cloud(nm):
                ps.remove_point_cloud(nm)
        nm_curve = _curve_net_name(sid_cur)
        if ps.has_curve_network(nm_curve):
            try:
                ps.get_curve_network(nm_curve).remove_quantity('match_score')
            except Exception:
                pass
        vs['template_idx0'] = -1
        vs['template_idx1'] = -1
        vs['pick_mode'] = None
        vs['match_scores'] = None
        vs['match_template_len'] = 0
        vs['match_kind'] = ''
        vs['match_mode'] = ''
        vs['hmm_segments'] = None
        _procrustes_reset_state()
    if vs['pick_mode'] is not None:
        psim.SameLine()
        psim.TextUnformatted(f'(click a sample dot for {vs["pick_mode"]})')

    if _consume_pick():
        _refresh_endpoint_markers()
        _refresh_template_highlight()
        _draw_match_start_marker()

    i0 = int(vs['template_idx0'])
    i1 = int(vs['template_idx1'])
    s0 = 'unset' if i0 < 0 else str(i0)
    s1 = 'unset' if i1 < 0 else str(i1)
    if i0 >= 0 and i1 >= 0 and i1 > i0:
        L = (i1 - i0 + 1) * res['ds_eff']
        psim.TextUnformatted(f'  start={s0}  end={s1}  '
                             f'len={i1 - i0 + 1} samples ({L:.1f} voxels)')
    else:
        psim.TextUnformatted(f'  start={s0}  end={s1}  (need both, end > start)')

    can_match = i0 >= 0 and i1 >= 0 and i1 > i0

    # Export just the selected orange span (OBJ + npy) for the template-fitting figure.
    if psim.Button('Export segment'):
        _export_template_segment()
    psim.SameLine()
    psim.TextUnformatted('(orange span -> OBJ + npy next to input npz)')

    # Match the SELECTED span (i0..i1) against the templates -> softmax rates.
    if psim.Button('Match rates (selected span)'):
        _stitch_match_rates()
    mr = vs.get('match_rates')
    if mr:
        for r in mr:
            psim.TextUnformatted(
                f'    {r["name"]:>10s}: rms={r["rms"]:.2f}  '
                f'rate={r["prob"] * 100:.1f}%')

    # ── Auto stitch-type match (pick ONE start; classify via templates) ─────
    psim.Separator()
    ntmpl = len(vs.get('stitch_templates') or [])
    psim.TextUnformatted(f'Auto stitch match ({ntmpl} templates):')
    if ntmpl == 0:
        psim.TextUnformatted('  (none; pass --stitch_templates <json>)')
    else:
        if psim.Button('Pick match start'):
            vs['pick_mode'] = 'match_start'
        psim.SameLine()
        ms = int(vs['match_start_idx'])
        psim.TextUnformatted(f'start={"unset" if ms < 0 else ms}')
        if vs['pick_mode'] == 'match_start':
            psim.SameLine(); psim.TextUnformatted('(click a sample dot)')
        if psim.Button('Auto-match stitch'):
            if ms >= 0:
                _auto_match_stitch()
            else:
                print('[stitch-match] pick a start first')
        psim.SameLine()
        if psim.Button('Clear match'):
            _clear_automatch()
            vs['match_start_idx'] = -1
        _, vs['stitch_softmax_T'] = psim.SliderFloat(
            'softmax T (vox)', float(vs['stitch_softmax_T']), 1.0, 60.0)
        r = vs.get('automatch')
        if r:
            psim.TextUnformatted(
                f'  -> {r["name"]}  rms={r["rms"]:.2f}  p={r["prob"]:.2f}  '
                f'span=[{r["s"]},{r["e"]}]')
            for a in r.get('all', []):
                psim.TextUnformatted(
                    f'      {a["name"]:>10s}: rms={a["rms"]:.2f}  p={a["prob"]:.2f}')

    psim.PushItemWidth(180)

    # ── Procrustes Auto-Reg (3D rigid-aligned AR stitch detection) ──────────
    psim.Separator()
    psim.TextUnformatted('Procrustes Auto-Reg stitch detection:')
    if can_match:
        if psim.Button('Match Procrustes'):
            _compute_stitch_match_procrustes()
            _refresh_match_view()
        psim.SameLine()
        if psim.Button('Step Fwd'):
            _procrustes_step(+1)
            _refresh_match_view()
        psim.SameLine()
        if psim.Button('Step Bwd'):
            _procrustes_step(-1)
            _refresh_match_view()
        psim.SameLine()
        if psim.Button('Clear Procrustes'):
            sid_cur = cur['sid']
            nm_proc = _procrustes_hi_net_name(sid_cur)
            if ps.has_curve_network(nm_proc):
                ps.remove_curve_network(nm_proc)
            _remove_all_procrustes_segments()
            _procrustes_reset_state()
            if vs['match_kind'] == 'procrustes':
                vs['match_kind'] = ''
                vs['match_mode'] = ''
        # Undo the last Step in either direction (the template seed is kept).
        n_uf = len(vs.get('procrustes_undo_fwd') or [])
        n_ub = len(vs.get('procrustes_undo_bwd') or [])
        if psim.Button(f'Undo Fwd ({n_uf})'):
            if _procrustes_undo_step(+1):
                _refresh_match_view()
        psim.SameLine()
        if psim.Button(f'Undo Bwd ({n_ub})'):
            if _procrustes_undo_step(-1):
                _refresh_match_view()
    else:
        psim.TextDisabled('  (pick template to enable)')
    # Save / load the current stitch extraction (segments + template + params)
    # to an npz next to the input yarn.
    if psim.Button('Save stitches (-> npz)'):
        _save_stitch_extraction()
    psim.SameLine()
    if psim.Button('Load stitches'):
        _load_stitch_extraction()
    _, vs['procrustes_n_samples'] = psim.SliderInt(
        'Proc N samples', int(vs['procrustes_n_samples']), 10, 100)
    _, vs['procrustes_threshold'] = psim.SliderFloat(
        'Proc threshold', float(vs['procrustes_threshold']), 0.1, 100.0)
    _, vs['procrustes_length_variation'] = psim.SliderFloat(
        'Proc length var', float(vs['procrustes_length_variation']), 0.1, 1.0)
    show_changed, new_show = psim.Checkbox(
        'Show seg samples', bool(vs.get('procrustes_show_samples', True)))
    if show_changed:
        _set_procrustes_samples_visible(new_show)
    psim.SameLine()
    corr_changed, new_corr = psim.Checkbox(
        'Show seg corr lines', bool(vs.get('procrustes_show_corr', True)))
    if corr_changed:
        _set_procrustes_corr_visible(new_corr)

    psim.PopItemWidth()

    # ── Kabsch Algorithm Viz (independent of procrustes match results) ────
    psim.Separator()
    psim.TextUnformatted('── Kabsch Algorithm Viz ──')
    kab = _kabsch_viz
    segments = vs.get('procrustes_segments') or []
    if len(segments) < 2:
        psim.TextDisabled('  (need ≥2 procrustes segments — '
                          'run Step Fwd/Bwd or Match Procrustes first)')
    else:
        seg_names = [f'{i}: ({a}-{b})' for i, (a, b) in enumerate(segments)]
        cur_src = max(0, min(int(kab['src_seg_idx']), len(segments) - 1))
        cur_tgt = max(0, min(int(kab['tgt_seg_idx']), len(segments) - 1))
        src_changed, cur_src = psim.Combo('Kabsch src', cur_src, seg_names)
        tgt_changed, cur_tgt = psim.Combo('Kabsch tgt', cur_tgt, seg_names)
        if src_changed: kab['src_seg_idx'] = cur_src
        if tgt_changed: kab['tgt_seg_idx'] = cur_tgt

        if psim.Button('Setup'):
            _kabsch_setup(cur_src, cur_tgt)
        psim.SameLine()
        if psim.Button('Animate'):
            # Run corr_draw → align as a chained sequence.
            _kabsch_run_sequence([
                ('corr_draw', float(kab['phase_dur'])),
                ('align',     float(kab['phase_dur'])),
            ])
        psim.SameLine()
        if psim.Button('Draw Corr'):
            if kab['src_pts_orig'] is not None:
                _kabsch_start_phase('corr_draw', kab['phase_dur'])
            else:
                print('Kabsch viz: click Setup first.')
        psim.SameLine()
        if psim.Button('Align Only'):
            if kab['src_pts_orig'] is not None:
                _kabsch_start_phase('align', kab['phase_dur'])
            else:
                print('Kabsch viz: click Setup first.')

        if psim.Button('Reset Pos'):
            _kabsch_reset_pos()
        psim.SameLine()
        if psim.Button('Clear Kabsch'):
            _kabsch_clear_visuals()
            kab['phase']    = 'idle'
            kab['sequence'] = None
        psim.SameLine()
        if psim.Button('Export GIF'):
            _kabsch_export_start()

        _, kab['phase_dur'] = psim.SliderFloat(
            'phase_dur', float(kab['phase_dur']), 0.3, 5.0)
        psim.SameLine()
        _, _kabsch_export['fps'] = psim.SliderFloat(
            'gif_fps', float(_kabsch_export.get('fps', 15.0)), 5.0, 60.0)

        if kab['ssd'] is not None:
            psim.TextUnformatted(
                f'  predicted SSD = {kab["ssd"]:.3f}    '
                f'rms = {kab["rms"]:.3f}')
        if kab['phase'] != 'idle':
            psim.TextUnformatted(f'  [animating: {kab["phase"]}]')
        if _kabsch_export['active']:
            psim.TextUnformatted(
                f'  [export active: {_kabsch_export["frame_counter"]} '
                f'frames captured]')

    psim.Separator()

    scores = vs['match_scores']
    m = vs['match_template_len']
    if vs['match_kind'] == 'hmm':
        segs = vs.get('hmm_segments') or []
        gf = vs.get('hmm_gap_frac')
        tc = vs.get('hmm_total_cost')
        psim.TextUnformatted(
            f'  mode={vs["match_mode"]}  segments={len(segs)}  '
            f'gap_frac={(gf or 0.0):.2f}  total_cost={(tc or 0.0):.1f}')
    elif vs['match_kind'] == 'procrustes':
        segs  = vs.get('procrustes_segments') or []
        n_fwd = int(vs.get('procrustes_n_fwd') or 0)
        n_bwd = int(vs.get('procrustes_n_bwd') or 0)
        med   = vs.get('procrustes_median_cost') or 0.0
        psim.TextUnformatted(
            f'  mode={vs["match_mode"]}  segments={len(segs)}  '
            f'(+{n_fwd} fwd / +{n_bwd} bwd)  median_cost={med:.3f}')
    elif scores is not None and m:
        finite = scores[np.isfinite(scores)]
        if finite.size:
            psim.TextUnformatted(
                f'  mode={vs["match_mode"]}  '
                f'score range [{float(finite.min()):+.3f}, {float(finite.max()):+.3f}]')
        else:
            psim.TextUnformatted(f'  mode={vs["match_mode"]}  (all NaN)')


def visualise(curves_data: list[tuple[int, dict]]) -> None:
    """Open a polyscope window: pick template span, then press Match."""
    ps.set_program_name('crochet curve template matching')
    ps.init()
    ps.set_up_dir('z_up')

    vs_curves = []
    for sid, res in curves_data:
        nm = _curve_net_name(sid)
        n = len(res['pts_u'])
        edges = np.stack([np.arange(n - 1), np.arange(1, n)], axis=1).astype(np.int32)
        cn = ps.register_curve_network(nm, res['pts_u'], edges, radius=0.0010)
        cn.add_scalar_quantity('kappa', res['kappa'], enabled=True, cmap='viridis')
        if res.get('writhe') is not None:
            w = res['writhe']
            w_finite = w[np.isfinite(w)]
            if w_finite.size:
                vmax = float(np.nanmax(np.abs(w_finite)))
                if vmax < 1e-12:
                    vmax = 1e-12
                w_plot = np.where(np.isfinite(w), w, 0.0)
                cn.add_scalar_quantity('writhe', w_plot, enabled=False,
                                       cmap='coolwarm', vminmax=(-vmax, vmax))

        # Dense, clickable sample point cloud (snap target for template picking)
        pc = ps.register_point_cloud(_samples_pc_name(sid), res['pts_u'],
                                     radius=0.0008)
        pc.set_color((0.85, 0.85, 0.85))
        pc.set_transparency(0.4)

        vs_curves.append({'sid': sid, 'res': res})

    _view_state['curves'] = vs_curves
    _view_state['curve_idx'] = 0
    # No default template selection: start unset (-1) so nothing is
    # highlighted until the user picks a start and an end.
    n_first = len(vs_curves[0]['res']['pts_u']) if vs_curves else 0
    if not (0 <= _view_state['template_idx0'] < n_first
            and 0 <= _view_state['template_idx1'] < n_first
            and _view_state['template_idx1'] > _view_state['template_idx0']):
        _view_state['template_idx0'] = -1
        _view_state['template_idx1'] = -1
    _view_state['match_scores'] = None
    _view_state['match_template_len'] = 0
    # Refresh markers/highlight (draws nothing while the span is unset).
    _refresh_endpoint_markers()
    _refresh_template_highlight()
    ps.set_user_callback(_ui_callback)

    # Auto-restore saved camera view on the first callback frame, if present.
    if os.path.isfile(_CAMERA_PATH):
        _pending_camera_load[0] = True
        print(f'  Will restore camera from {_CAMERA_PATH} on first frame.')
    else:
        print(f'  (no camera file at {_CAMERA_PATH} — '
              'use "Save Camera" to create one)')

    ps.show()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', type=Path,
                    default=Path('./'
                                 'c_denoised_pointcloud_full_energy_10000_'
                                 'linearity_0.03_r_10_sim_0.85_united_gt_0.85_ms.npz'))
    ap.add_argument('--ds', type=float, default=2.0,
                    help='resample spacing along arclength [voxels]')
    ap.add_argument('--sigma', type=float, default=3.0,
                    help='Gaussian smoothing σ for κ [voxels]')
    ap.add_argument('--min-period', type=float, default=20.0,
                    help='ignore autocorr peaks below this lag [voxels]')
    ap.add_argument('--top', type=int, default=5)
    ap.add_argument('--out-dir', type=Path, default=Path('output/curve_autocorr'))
    ap.add_argument('--no-view', action='store_true',
                    help='skip polyscope viewer (only write PNG)')
    ap.add_argument('--load_stitches', nargs='?', const='__latest__', default=None,
                    help='auto-load a saved stitch extraction on startup; bare '
                         'flag = stitches_sid<sid>_latest.npz next to the input '
                         'yarn, or pass an explicit path.')
    ap.add_argument('--writhe-window', type=float, default=30.0,
                    help='half-window for local self-writhe density [voxels]; '
                         '0 disables')
    ap.add_argument('--poss', type=int, default=0,
                    help='for a yarn_possibilities_v1 npz: which possibility '
                         'to analyse (0-based)')
    ap.add_argument('--stitch_templates', type=str,
                    default=os.path.join(_HERE, 'assets', 'stitch_templates',
                                         'stitch_templates.json'),
                    help='json listing stitch-type templates {name,npy,color} '
                         'for the "Auto stitch match" mode.')
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    curves, seg_ids = load_curves_and_ids(args.npz, poss=args.poss)
    print(f'Loaded {len(curves)} curve(s) from {args.npz.name}')

    _view_state['raw_curves']  = curves
    _view_state['raw_seg_ids'] = seg_ids
    _view_state['npz_path']    = args.npz
    _view_state['out_dir']     = args.out_dir
    _view_state['stitch_templates'] = _load_stitch_templates(args.stitch_templates)
    if args.load_stitches is not None:
        _pending_stitch_load[0] = args.load_stitches   # applied on 1st frame

    view_payload = []
    for i, c in enumerate(curves):
        sid = int(seg_ids[i]) if i < len(seg_ids) else i
        ww = args.writhe_window if args.writhe_window > 0 else None
        res = analyse_curve(c, args.ds, args.sigma, args.top, args.min_period,
                            writhe_window=ww)
        if res is None:
            print(f'  curve[{i}] sid={sid}: too short ({len(c)} pts) — skipped')
            continue
        out_path = args.out_dir / f'curve_seg{sid:04d}.png'
        plot_curve(res, sid, out_path)
        print(f'  curve[{i}] sid={sid}: '
              f'L={res["L"]:.1f}  pts={len(res["s"])}  '
              f'top peaks (lag, height):')
        for p, h in zip(res['peaks'], res['heights']):
            print(f'    lag={p * res["ds_eff"]:7.2f}   ac={h:.3f}')
        print(f'    → saved {out_path}')
        view_payload.append((sid, res))

    if not args.no_view and view_payload:
        visualise(view_payload)


if __name__ == '__main__':
    main()
