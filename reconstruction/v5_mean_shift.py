"""Blurring Mean-Shift on a 3D point cloud with directions.

Loads a real npz with (points, directions). Runs MS with anisotropic
Gaussian kernel and tangent-plane projection in Polyscope; history
scrubber, click-inspect (bw ellipsoid), and CPU/GPU shift backends."""

from __future__ import annotations
import argparse
import colorsys
import time

import numpy as np
import polyscope as ps
import polyscope.imgui as psim
from scipy.spatial import cKDTree, KDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import (
    connected_components, minimum_spanning_tree, shortest_path)
from scipy.interpolate import splprep, splev


# ── CLI ────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--npz', default=None,
                    help='Real-data npz with (points, directions).  Optional '
                         'IF --load_seg <path> is given: the matching source '
                         'cloud is auto-found under '
                         'CT_Dataset/BinnedPcds128/<stem>.npz, where <stem> is '
                         'the output/<stem>/ folder the seg_state lives in.')
parser.add_argument('--load_seg', nargs='?', default=None, const='',
                    help='Restore the topo-MST segments/components saved '
                         'by "Save seg state".  Pass NO path → auto-find: '
                         'prefers output/<stem>/segs/seg_state_latest.npz, '
                         'else newest seg_state_*.npz in that folder, else '
                         'CWD (legacy).  Pass a path to load that file. '
                         'MS is replayed deterministically so gidx line up.')
parser.add_argument('--load_yarn', nargs='?', default=None, const='',
                    help='Path to a yarn npz saved by "Save yarn": restores '
                         'the fitted curves on startup.  Bare --load_yarn '
                         '(no path) uses the canonical yarn_latest.npz / '
                         'newest yarn_*.npz.')
parser.add_argument('--batch', action='store_true',
                    help='Headless: run the default Run MS (n_iter iterations) '
                         'then Save ms_points and exit. No GUI, no other steps.')
parser.add_argument('--batch_topo_save', action='store_true',
                    help='Headless: Run MS + Save ms_points + Topo MST + '
                         'Save seg state, then exit.  Implies --batch '
                         'plus the topo-MST + seg-save steps.  Use '
                         '--save_postfix _unprocessed to land in '
                         'segs_unprocessed/ instead of segs_before/.')
parser.add_argument('--batch_fit', action='store_true',
                    help='Headless: after --load_seg restores the cached '
                         'topo adj (incl. stitches), run Fit curves (topo '
                         'MST) + Save yarn into curves<save_postfix>/, '
                         'then exit. No GUI.')
parser.add_argument('--save_postfix', default='_before',
                    help='Suffix appended to the segs/ and curves/ output '
                         'subfolders (e.g. "_before" -> segs_before/, '
                         'curves_before/).  Does NOT affect ms_points/. '
                         'Editable live in the GUI.')
parser.add_argument('--ablation', default=None,
                    help='Override MS state for an ablation variant before a '
                         '--batch run. One of: full, wo_aniso, wo_dir_update, '
                         'wo_dir_asym, wo_topo, wo_anneal_bw30, wo_anneal_bw10.')
args = parser.parse_args()


# ── Load real data ────────────────────────────────────────────────────────
import os
import glob

# Anchor the binned-cloud dataset dir at the script's own location so a
# --load_seg path resolves to its source cloud regardless of the CWD.
_BINNED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'CT_Dataset', 'BinnedPcds128')


def _resolve_npz_from_seg(seg_path: str) -> str | None:
    """Map a seg_state path (output/<stem>/segs*/seg_state_*.npz) back to its
    source cloud CT_Dataset/BinnedPcds128/<stem>.npz.  <stem> is the name of
    the output/<stem>/ folder the seg_state lives under (== the npz basename
    the session was started with).  Returns None if nothing matches."""
    sp = os.path.abspath(seg_path)
    stem = os.path.basename(os.path.dirname(os.path.dirname(sp)))
    cand = os.path.join(_BINNED_DIR, stem + '.npz')
    if os.path.exists(cand):
        return cand
    hits = sorted(glob.glob(os.path.join(_BINNED_DIR, stem + '*.npz')))
    return hits[0] if hits else None


# --npz is optional when --load_seg gives an explicit path: derive the source
# cloud from the seg_state's output/<stem>/ folder so the user need only pass
# --load_seg.  A bare --load_seg (auto-find, '') still needs --npz to know the
# stem, so it falls through to the error below.
if args.npz is None:
    if args.load_seg:                       # explicit (non-empty) path given
        _npz = _resolve_npz_from_seg(args.load_seg)
        if _npz is None:
            _stem = os.path.basename(os.path.dirname(
                os.path.dirname(os.path.abspath(args.load_seg))))
            raise SystemExit(
                f'--load_seg given but no source cloud for stem "{_stem}" '
                f'under {_BINNED_DIR}/.  Pass --npz explicitly.')
        args.npz = _npz
        print(f'[load_seg] resolved source npz: {args.npz}')
    else:
        raise SystemExit(
            '--npz is required (or pass --load_seg <path> to auto-resolve the '
            'source cloud from CT_Dataset/BinnedPcds128/).')

print(f'Loading real data: {os.path.basename(args.npz)}')

# Per-input output directory: output/<npz-stem>/{segs,curves}/.  All seg-
# state pickles land in segs/; all yarn-curves npz in curves/.  Created
# once at startup so the rest of the session can just write into them.
_NPZ_STEM   = os.path.splitext(os.path.basename(args.npz))[0]
_OUT_DIR    = os.path.join('output', _NPZ_STEM)
_SAVE_POSTFIX = args.save_postfix          # suffix for segs/ & curves/ (not ms_points/)
_OUT_SEGS   = os.path.join(_OUT_DIR, 'segs'   + _SAVE_POSTFIX)
_OUT_CURVES = os.path.join(_OUT_DIR, 'curves' + _SAVE_POSTFIX)
_OUT_MS     = os.path.join(_OUT_DIR, 'ms_points')


def _set_save_postfix(pfx: str) -> None:
    """Re-point the segs/ & curves/ output dirs to use suffix `pfx` (live)."""
    global _SAVE_POSTFIX, _OUT_SEGS, _OUT_CURVES
    _SAVE_POSTFIX = pfx
    _OUT_SEGS   = os.path.join(_OUT_DIR, 'segs'   + pfx)
    _OUT_CURVES = os.path.join(_OUT_DIR, 'curves' + pfx)
    os.makedirs(_OUT_SEGS, exist_ok=True); os.makedirs(_OUT_CURVES, exist_ok=True)


os.makedirs(_OUT_SEGS,   exist_ok=True)
os.makedirs(_OUT_CURVES, exist_ok=True)
os.makedirs(_OUT_MS,     exist_ok=True)
print(f'  output dir: {_OUT_DIR}/  (segs{_SAVE_POSTFIX}/, curves{_SAVE_POSTFIX}/, ms_points/)')

# ── repair-session stats: timer + Connect-B (sketch-bridge) counter ────────
# Counts COMPLETED sketch bridges only (Connect B / key B). One JSONL line
# per event is appended to output/<stem>/repair_stats.jsonl, so the record
# survives crashes; the last line of a session is its summary.
_REPAIR_STATS = {'t0': time.time(), 't_first_b': None, 't_last_b': None,
                 'connect_b': 0,
                 'sid': time.strftime('%Y%m%d_%H%M%S')}


def _repair_elapsed() -> float:
    """Seconds since the repair session started."""
    return time.time() - _REPAIR_STATS['t0']


def _repair_stats_bump(kind: str = 'B', g1: int = -1, g2: int = -1,
                       synth_start: int = -1, n_synth: int = 0) -> None:
    """Record one manual connection. kind 'B' (sketch bridge) increments the
    Connect-B counter; kind 'A' (straight connect) is logged but not counted.
    g1/g2 = bridged endpoint gidx; synth_start/n_synth = the synthetic chain
    appended for this bridge (for the later red/green rendering)."""
    import json as _json, os as _os
    now = time.time()
    st = _REPAIR_STATS
    if st['t_first_b'] is None:
        st['t_first_b'] = now
    st['t_last_b'] = now
    if kind == 'B':
        st['connect_b'] += 1
    rec = {
        'ts': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now)),
        'sample': _os.path.basename(str(_OUT_DIR)),
        'sid': st['sid'],
        'kind': kind,
        'g1': int(g1), 'g2': int(g2),
        'synth_start': int(synth_start), 'n_synth': int(n_synth),
        'connect_b': st['connect_b'],
        'active_s': round(_repair_elapsed(), 1),
        'session_s': round(now - st['t0'], 1),
        'repair_span_s': round(st['t_last_b'] - st['t_first_b'], 1),
        'save_postfix': _SAVE_POSTFIX,
    }
    try:
        with open(_os.path.join(str(_OUT_DIR), 'repair_stats.jsonl'),
                  'a') as fh:
            fh.write(_json.dumps(rec) + chr(10))
    except Exception as exn:
        print(f'  [repair-stats] write failed: {exn}')
    _mm, _ss = divmod(int(rec['active_s']), 60)
    print(f"  [repair-stats] Connect-{kind} "
          f"(B count {st['connect_b']})  gidx {g1}<->{g2}  "
          f"synth {synth_start}+{n_synth}  active {_mm}m{_ss:02d}s  "
          f"→ repair_stats.jsonl")

_d = np.load(args.npz, allow_pickle=False)
pts_raw  = _d['points'].astype(np.float32)
if 'directions' in _d.files:
    _DIR_KEY = 'directions'
elif 'dirs' in _d.files:
    _DIR_KEY = 'dirs'
else:
    _keys = list(_d.files)
    _d.close()
    raise SystemExit(f"npz has no `directions`/`dirs`; keys={_keys}")
dirs_raw = _d[_DIR_KEY].astype(np.float32)
# Optional per-point scalar fields — carried through the same edge-trim so a
# saved cloud ("Save current points") can match the INPUT npz format
# (points, directions/dirs, energy, linearity).  None if the key is absent.
_energy_raw    = _d['energy'].astype(np.float32)    if 'energy'    in _d.files else None
_linearity_raw = _d['linearity'].astype(np.float32) if 'linearity' in _d.files else None
_d.close()
# ── Edge-noise trim ───────────────────────────────────────────────
# Voxel-boundary noise sits right on the bbox shell.  Keep only points
# within the central EDGE_KEEP_FRAC of the bbox range PER AXIS (drops
# the outer (1-frac)/2 on each side of every face).  Done at LOAD so
# pts_orig / state['pts'] are born clean and never mutated mid-session.
_EDGE_KEEP_FRAC = 0.995
_bb_lo  = pts_raw.min(axis=0)
_bb_hi  = pts_raw.max(axis=0)
_bb_ctr = (_bb_lo + _bb_hi) * 0.5
_bb_hlf = (_bb_hi - _bb_lo) * 0.5
_edge_keep = np.all(
    np.abs(pts_raw - _bb_ctr) <= _EDGE_KEEP_FRAC * _bb_hlf, axis=1)
_n_drop = int((~_edge_keep).sum())
print(f'  edge-noise trim (central {_EDGE_KEEP_FRAC:.2f} of bbox per '
      f'axis): dropped {_n_drop:,} / {len(_edge_keep):,} edge point(s)')
pts_raw  = pts_raw[_edge_keep]
dirs_raw = dirs_raw[_edge_keep]
if _energy_raw    is not None: _energy_raw    = _energy_raw[_edge_keep]
if _linearity_raw is not None: _linearity_raw = _linearity_raw[_edge_keep]
pts_orig  = np.ascontiguousarray(pts_raw)
# Normalize dirs
_n = np.linalg.norm(dirs_raw, axis=1, keepdims=True)
dirs_orig = (dirs_raw / np.where(_n > 1e-8, _n, 1.0)).astype(np.float32)
# Aligned scalar fields kept for "Save current points" (None if absent).
energy_orig    = None if _energy_raw    is None else np.ascontiguousarray(_energy_raw)
linearity_orig = None if _linearity_raw is None else np.ascontiguousarray(_linearity_raw)
N = len(pts_orig)
print(f'  N = {N:,}    points range: '
      f'{pts_orig.min(axis=0).tolist()} → {pts_orig.max(axis=0).tolist()}')


# ── Mean shift step ───────────────────────────────────────────────────────
def _scheduled_bw(it: int, n_iter: int,
                  bw_start: float, bw_end: float) -> float:
    """Exponential bw schedule from bw_start at iter 0 to bw_end at iter n_iter-1."""
    if n_iter <= 1:
        return float(bw_start)
    t = it / (n_iter - 1)
    return float(bw_start) * (float(bw_end) / max(float(bw_start), 1e-6)) ** t


def _ms_step(pts: np.ndarray, dirs: np.ndarray,
             bw: float, dir_alpha: float, k_search: int,
             update_dirs: bool = False,
             dir_asym_gamma: float = 0.1,
             gauss_penalty_par: float = 1.0,
             gauss_penalty_perp: float = 1.0,
             bw_aniso_ratio: float = 1.0,
             ) -> tuple[np.ndarray, np.ndarray | None]:
    """Standard MS with tangent-plane projection. Returns delta_perp (N,3).

    Anisotropic Gaussian kernel (mirror of meanshift_centers):
      σ_∥ = ratio · bw,  σ_⊥ = bw      (prolate ellipsoid along d_i)
      gauss = exp(-(α_∥·Δ_∥²/2σ_∥² + α_⊥·Δ_⊥²/2σ_⊥²))
      hard cutoff: (Δ_∥/σ_∥)² + (Δ_⊥/σ_⊥)² ≤ 1
    par=perp=ratio=1 → reduces to the isotropic sphere (legacy default)."""
    Nloc = len(pts)
    k    = min(int(k_search), Nloc)
    tree = cKDTree(pts)
    dists, idx = tree.query(pts, k=k + 1, workers=-1)
    dists = dists[:, 1:].astype(np.float32)
    idx   = idx[:, 1:].astype(np.int64)

    nb_pos  = pts[idx]
    nb_dirs = dirs[idx]
    # dir_w still uses the dir field — that's the "are we on the same yarn"
    # semantic, independent of the projection axis.
    cos_sim = np.abs(np.einsum('nki,ni->nk', nb_dirs, dirs)
                     ).astype(np.float32)
    dir_w   = (cos_sim ** np.float32(dir_alpha)).astype(np.float32)

    # ── Anisotropic kernel ───────────────────────────────────────────────
    # axial = (p_j - p_i) · d_i   (sign cancels under squaring)
    axial = np.einsum('nki,ni->nk',
                      nb_pos - pts[:, None, :], dirs).astype(np.float32)
    axial_sq   = (axial ** 2).astype(np.float32)
    total_sq   = (dists ** 2).astype(np.float32)
    lateral_sq = np.maximum(total_sq - axial_sq,
                             np.float32(0.0)).astype(np.float32)
    sigma_par_sq  = np.float32((bw * bw_aniso_ratio) ** 2)
    sigma_perp_sq = np.float32(bw ** 2)
    inv2_par  = np.float32(1.0 / (2.0 * sigma_par_sq  + 1e-20))
    inv2_perp = np.float32(1.0 / (2.0 * sigma_perp_sq + 1e-20))
    gauss  = np.exp(
        -(np.float32(gauss_penalty_par)  * axial_sq   * inv2_par
          + np.float32(gauss_penalty_perp) * lateral_sq * inv2_perp)
        ).astype(np.float32)
    r_mask = ((axial_sq / sigma_par_sq
               + lateral_sq / sigma_perp_sq) <= np.float32(1.0)
              ).astype(np.float32)
    w = (gauss * r_mask * dir_w).astype(np.float32)

    w_sum  = w.sum(axis=1, keepdims=True)
    has_nb = (w_sum.squeeze(1) > 1e-12)
    new_pts = pts.copy()
    new_pts[has_nb] = (
        (w[has_nb, :, None] * nb_pos[has_nb]).sum(axis=1)
        / w_sum[has_nb])

    delta = new_pts - pts
    along = (delta * dirs).sum(axis=1, keepdims=True)
    delta -= along * dirs                           # ⊥ d_i

    # Optional: per-iter direction update with asymmetry damping
    # (mirror of meanshift_centers._do_step update_dirs branch). The
    # asym factor freezes endpoints whose neighbors are all on one side.
    new_dirs = None
    if update_dirs:
        sgn      = np.sign(np.einsum('nki,ni->nk', nb_dirs, dirs))
        sgn[sgn == 0] = 1.0
        aligned  = sgn[:, :, None] * nb_dirs
        d_new    = (w[:, :, None] * aligned).sum(axis=1)
        dn_new   = np.linalg.norm(d_new, axis=1, keepdims=True)
        valid_d  = (dn_new > 1e-8).squeeze(1)
        d_norm_  = dirs.copy()
        d_norm_[valid_d] = (
            d_new[valid_d] / dn_new[valid_d]).astype(np.float32)

        rel_off = nb_pos - pts[:, None, :]
        t_ij    = np.einsum('nki,ni->nk', rel_off, dirs)
        sgn_t   = np.sign(t_ij).astype(np.float32)
        w_sum_a = w.sum(axis=1) + np.float32(1e-12)
        asym    = (np.abs((w * sgn_t).sum(axis=1)) / w_sum_a
                   ).astype(np.float32)
        damp    = (1.0 - asym ** np.float32(dir_asym_gamma)
                   ).astype(np.float32)
        d_blend = (damp[:, None] * d_norm_
                   + (1.0 - damp[:, None]) * dirs).astype(np.float32)
        nn_     = np.linalg.norm(d_blend, axis=1, keepdims=True)
        new_dirs = (d_blend / np.where(nn_ > 1e-8, nn_, 1.0)
                    ).astype(np.float32)
    return delta.astype(np.float32), new_dirs


# Optional GPU backend (cupy). MS shift and PBD link projection both have
# tight inner loops that benefit from GPU when N is large.
try:
    import cupy as _cp
    import cupyx as _cpx
    _cupy_available = True
except ImportError:
    _cp = None
    _cpx = None
    _cupy_available = False


def _ms_step_gpu(pts: np.ndarray, dirs: np.ndarray,
                 bw: float, dir_alpha: float, k_search: int,
                 update_dirs: bool = False,
                 dir_asym_gamma: float = 0.1,
                 gauss_penalty_par: float = 1.0,
                 gauss_penalty_perp: float = 1.0,
                 bw_aniso_ratio: float = 1.0,
                 ) -> tuple[np.ndarray, np.ndarray | None]:
    """GPU port of _ms_step with anisotropic Gaussian kernel."""
    if not _cupy_available:
        raise RuntimeError('cupy not available; use _ms_step (CPU)')
    Nloc = len(pts)
    k    = min(int(k_search), Nloc)
    tree = cKDTree(pts)
    dists_cpu, idx_cpu = tree.query(pts, k=k + 1, workers=-1)
    dists_cpu = dists_cpu[:, 1:].astype(np.float32)
    idx_cpu   = idx_cpu[:, 1:].astype(np.int64)

    alpha32 = _cp.float32(dir_alpha)
    gamma32 = _cp.float32(dir_asym_gamma)
    eps     = _cp.float32(1e-12)
    eps_u   = _cp.float32(1e-8)
    sigma_par_sq  = _cp.float32((bw * bw_aniso_ratio) ** 2)
    sigma_perp_sq = _cp.float32(bw ** 2)
    inv2_par  = _cp.float32(1.0 / (2.0 * sigma_par_sq + 1e-20))
    inv2_perp = _cp.float32(1.0 / (2.0 * sigma_perp_sq + 1e-20))
    gp_par_g  = _cp.float32(gauss_penalty_par)
    gp_perp_g = _cp.float32(gauss_penalty_perp)

    pts_g  = _cp.asarray(pts,  dtype=_cp.float32)
    dirs_g = _cp.asarray(dirs, dtype=_cp.float32)
    idx_g  = _cp.asarray(idx_cpu)
    dists_g = _cp.asarray(dists_cpu)

    nb_pos  = pts_g[idx_g]                              # (N, k, 3)
    nb_dirs = dirs_g[idx_g]

    cos_sim = _cp.abs(_cp.einsum(
        'nki,ni->nk', nb_dirs, dirs_g)).astype(_cp.float32)
    dir_w  = (cos_sim ** alpha32).astype(_cp.float32)

    # ── Anisotropic kernel ───────────────────────────────────────────────
    axial = _cp.einsum('nki,ni->nk',
                       nb_pos - pts_g[:, None, :], dirs_g).astype(_cp.float32)
    axial_sq   = (axial ** 2).astype(_cp.float32)
    total_sq   = (dists_g ** 2).astype(_cp.float32)
    lateral_sq = _cp.maximum(total_sq - axial_sq,
                              _cp.float32(0.0)).astype(_cp.float32)
    gauss  = _cp.exp(-(gp_par_g  * axial_sq   * inv2_par
                       + gp_perp_g * lateral_sq * inv2_perp)
                     ).astype(_cp.float32)
    r_mask = ((axial_sq / sigma_par_sq
               + lateral_sq / sigma_perp_sq) <= _cp.float32(1.0)
              ).astype(_cp.float32)
    w = (gauss * r_mask * dir_w).astype(_cp.float32)

    w_sum  = w.sum(axis=1, keepdims=True)
    has_nb = (w_sum.squeeze(1) > eps)
    new_pts = pts_g.copy()
    weighted = (w[:, :, None] * nb_pos).sum(axis=1)
    if bool(has_nb.any()):
        new_pts[has_nb] = (
            weighted[has_nb] / w_sum[has_nb]).astype(_cp.float32)

    delta = (new_pts - pts_g).astype(_cp.float32)
    along = (delta * dirs_g).sum(axis=1, keepdims=True)
    delta = (delta - along * dirs_g).astype(_cp.float32)

    new_dirs_out = None
    if update_dirs:
        sgn = _cp.sign(_cp.einsum(
            'nki,ni->nk', nb_dirs, dirs_g)).astype(_cp.float32)
        sgn = _cp.where(sgn == 0, _cp.float32(1.0), sgn)
        aligned = sgn[:, :, None] * nb_dirs
        d_new = (w[:, :, None] * aligned).sum(axis=1)
        dn_new = _cp.linalg.norm(d_new, axis=1, keepdims=True)
        valid_d = (dn_new > eps_u).squeeze(1)
        d_norm_ = dirs_g.copy()
        if bool(valid_d.any()):
            d_norm_[valid_d] = (
                d_new[valid_d] / dn_new[valid_d]).astype(_cp.float32)

        rel_off = nb_pos - pts_g[:, None, :]
        t_ij = _cp.einsum(
            'nki,ni->nk', rel_off, dirs_g).astype(_cp.float32)
        sgn_t = _cp.sign(t_ij).astype(_cp.float32)
        w_sum_a = w.sum(axis=1) + eps
        asym = (_cp.abs((w * sgn_t).sum(axis=1)) / w_sum_a
                ).astype(_cp.float32)
        damp = (_cp.float32(1.0) - asym ** gamma32).astype(_cp.float32)
        d_blend = (damp[:, None] * d_norm_
                   + (_cp.float32(1.0) - damp[:, None]) * dirs_g
                   ).astype(_cp.float32)
        nn = _cp.linalg.norm(d_blend, axis=1, keepdims=True)
        new_dirs_g = (d_blend / _cp.where(
            nn > eps_u, nn, _cp.float32(1.0))
            ).astype(_cp.float32)
        new_dirs_out = _cp.asnumpy(new_dirs_g)

    return _cp.asnumpy(delta), new_dirs_out


# ── State ──────────────────────────────────────────────────────────────────
state = {
    'pts':            pts_orig.copy(),
    'dirs':           dirs_orig.copy(),
    'iter':           0,    # per-run counter (resets on Soft reset)
    'global_iter':    0,    # cumulative across all soft-resets — only
                            # cleared by hard Reset
    'history':        [pts_orig.copy()],
    # ── Bandwidth schedule (exp from bw_start to bw_end over n_iter) ──
    # Defaults are the coarse MS setting the curve pipeline is built on:
    # bw 30→10 exp schedule, k=128, dir_alpha=2.
    'bw_start':       30.0,
    'bw_end':         10.0,
    'dir_alpha':      2.0,
    'k_search':       128,
    # ── Direction update with asymmetry damping ──
    'update_dirs':    True,    # default ON for real data
    'dir_asym_gamma': 0.1,
    # ── Anisotropic Gaussian kernel (mirror of meanshift_centers) ──
    # σ_axial = ratio·bw,  σ_lateral = bw  (prolate ellipsoid along d_i)
    # gauss = exp(-(α_∥·Δ_∥²/2σ_∥² + α_⊥·Δ_⊥²/2σ_⊥²))
    # hard cutoff: (Δ_∥/σ_∥)² + (Δ_⊥/σ_⊥)² ≤ 1   (ellipsoid)
    # Defaults: par=1, perp=2, ratio=2
    # (σ_∥ = 2·bw axial, σ_⊥ = bw lateral, penalised 2× — coarse).
    'gauss_penalty_par':  1.0,
    'gauss_penalty_perp': 2.0,
    'bw_aniso_ratio':     2.0,
    # ── Click-to-inspect: click a point → render bw ellipsoid + kNN ──
    # `inspect_picked_idx` ≥ 0 means a point is being inspected. The
    # viz follows view_iter so scrubbing history shows the historical
    # neighbour-set / bw at that iter.
    'inspect_picked_idx':  -1,
    'running':        False,
    'n_iter':         20,
    # ── Topology lock: at iter == topo_lock_iter, freeze a STREAMLINE
    # spring lattice — for each point, find every point inside its
    # forward/backward dir cone (half-angle `topo_link_cone_deg`, length
    # `_TOPO_LINK_REACH`) and link all of them.
    # For every iter AFTER, a two-sided clamped spring to the frozen rest
    # lengths holds it (resists MS collapse) without strutting the
    # cross-section.
    'topo_lock_on':       True,  # default ON
    'topo_lock_iter':     5,     # iter to freeze topology + start preserve
    'topo_link_cone_deg': 30.0,  # fwd/bwd cone half-angle (sharper=lower)
    # ── Topo MST viz / click-junction system ──
    # click on the `topo_mst` curve_network → SELECT one OR two nodes:
    #   slot #1 (yellow ball) — Delete-A / Delete-B act on this one
    #   slot #2 (cyan ball)   — used with slot #1 for "Connect 1↔2"
    # Click order:  empty → fill #1 → fill #2 → rotate (oldest evicted).
    'topo_mst_pick_node':    -1,    # slot #1 curve_network node #
    'topo_mst_pick_gidx':    -1,    # slot #1 global cloud idx
    'topo_mst_pick_deg':     -1,    # slot #1 cached degree
    'topo_mst_pick2_node':   -1,    # slot #2 curve_network node #
    'topo_mst_pick2_gidx':   -1,    # slot #2 global cloud idx
    'topo_mst_pick2_deg':    -1,    # slot #2 cached degree
    'topo_mst_viz_radius':   0.90,  # polyscope DISPLAY radius (vox) of the
    #                                 topo_mst curve network (tube thickness).
    # After Connect / Sketch-bridge: auto-run the junction CLICK pipeline
    # (_topo_branch_explore) on each NEW junction and apply what it
    # classifies (cut/loop/search), one undo snapshot each.  A new collapse
    # (>-----<) skips the whole run.  New junctions are numbered and
    # printed regardless; this flag only gates the auto-apply.
    'connect_auto_process_new_jct': True,
    # Midpoint-cut rule (part of the auto-process above): if a NEW
    # junction's nearest pre-Connect (red) junction is within
    # this many vox, break the MST path between them at its arc-length
    # midpoint BEFORE auto-clicking the new one.
    'connect_newjct_midcut_vox': 100.0,
    # ── Delete-region AABBs: one clickable box per D-delete record ──
    'delete_aabb_show':     True,
    'delete_aabb_margin':   12.0,
    'delete_aabb_alpha':    0.18,
    # ── Collapse solving: loop-free solutions + which one is on screen ──
    'collapse_solutions':   None,
    'collapse_base':        None,
    'collapse_shown':       0,
    'collapse_dets':        None,
    'topo_mst_last_clickdel': -1,   # change-gate (curve_network node #)
    'topo_links_edges': None,  # (E,2) int32 frozen at iter 10
    'topo_links_rest':  None,  # (E,) iter-10 rest length per frozen edge
    'topo_links_iter':  -1,    # which iter the links were frozen at
    # ── Full-state history for scrubbing through past iters ──
    'history_full':   [],     # list of dicts (per iter snapshot)
    'view_iter':      0,
    'shown_iter':     -1,     # -1 forces initial render
    'picked':         0,
    'ms_gpu': bool(_cupy_available),     # GPU shift compute
    # ── Curve fitting (CC seg + B-spline per segment) ──
    'curve_radius':         4.0,   # KDTree.query_pairs radius (vox)
    'curve_min_pts':        30,    # drop CCs smaller than this
    'curve_dedup_tau':  6.0,   # vox: redundancy distance for fit-dedup (0 = off)
    'curve_dedup_frac': 0.6,   # drop a fitted curve if >= this fraction of it
    #                            lies within tau of already-accepted longer curves
    'curve_seg_labels':     None,  # (N,) int32, -1 = discarded
    'curve_K':              0,     # number of fitted segments
    'curve_sub_to_cc':      {},    # sub_idx → original CC sid
    'curve_selected_cc':    -1,    # last CC highlighted via curve click
    'curve_selected_idx':   None,  # (M,) int64 point indices of selection
    'curve_last_clicked_sub':  -1, # sub_idx of last curve click
    'curve_last_clicked_node': -1, # spline-node idx of last curve click
    # ── Endpoint cylinders: a semi-transparent, same-colour tube wrapping
    #    the LAST stretch of every curve at both ends (shown by default) ──
    'curve_endcap_show':    False, # render the endpoint cones/horns (off by default)
    'curve_endcap_radius':  28.0,  # vox: LARGE (far/outer) radius — the flare
    'curve_endcap_base_radius': 3.0,  # vox: SMALL radius where it meets the curve
    'curve_endcap_len':     30.0,  # vox: arc length back from the tip where the
                                   # cone's narrow base starts (wrap region)
    'curve_endcap_extend':  45.0,  # vox: project OUTWARD past the tip along the
                                   # curve's heading (tangent) by this much
    'curve_endcap_alpha':   0.45,  # transparency (0=clear .. 1=opaque)
}


# ── Polyscope viz ─────────────────────────────────────────────────────────
ps.init()
ps.set_up_dir('z_up')
ps.set_background_color((0.08, 0.08, 0.10))

pc = ps.register_point_cloud('points', pts_orig, radius=0.0016)
_gabor = np.clip(np.abs(dirs_orig) ** 0.6, 0.0, 1.0).astype(np.float32)
pc.add_color_quantity('direction_RGB', _gabor, enabled=True)

# Direction lines
# Scale viz lengths with the data span (toy: ~60 vox span; real data:
# ~400 vox span). Use bbox diagonal / 60 as the basis.
_bbox  = pts_orig.max(axis=0) - pts_orig.min(axis=0)
_span  = float(np.linalg.norm(_bbox))
_SCALE = max(_span / 60.0, 1.0)
DIR_LEN = 1.0 * _SCALE
# FIXED absolute marker radii (vox) — NOT scaled by _SCALE (which is huge on
# big clouds and made these balls too large).  Tune here.
_ARM_BREAK_RADIUS = 4.0    # loop-closure marker balls
print(f'  bbox diagonal = {_span:.1f}  viz scale = {_SCALE:.2f}')
_dir_nodes = np.vstack([pts_orig, pts_orig + DIR_LEN * dirs_orig]).astype(np.float32)
_dir_edges = np.column_stack([np.arange(N), np.arange(N) + N]).astype(np.int32)
cn_dir = ps.register_curve_network(
    'direction_lines', _dir_nodes, _dir_edges, radius=0.0005)
cn_dir.add_color_quantity('direction_RGB', _gabor,
                          defined_on='edges', enabled=True)
cn_dir.set_enabled(False)

# Original (pre-MS) positions for reference
pc_orig = ps.register_point_cloud(
    'original', pts_orig, radius=0.0008)
pc_orig.set_color((0.50, 0.50, 0.55))
pc_orig.set_enabled(False)

def _refresh_picked(idx: int) -> None:
    # the 'picked' viz cloud was removed (it only highlighted the
    # main-MS-loop selection and sat stale at pts_orig[0] during
    # main-MS-loop selection and sat stale at pts_orig[0] during curve
    state['picked'] = int(idx)


def _do_step() -> None:
    """One MS iteration on state['pts']."""
    s = state
    cur_pts  = s['pts']
    cur_dirs = s['dirs']

    bw_cur = _scheduled_bw(int(s['iter']), int(s['n_iter']),
                           float(s['bw_start']), float(s['bw_end']))
    ms_t0 = time.time()
    if bool(s.get('ms_gpu')) and _cupy_available:
        delta, new_dirs = _ms_step_gpu(
            cur_pts, cur_dirs,
            bw_cur, float(s['dir_alpha']),
            int(s['k_search']),
            update_dirs=bool(s['update_dirs']),
            dir_asym_gamma=float(s['dir_asym_gamma']),
            gauss_penalty_par=float(s['gauss_penalty_par']),
            gauss_penalty_perp=float(s['gauss_penalty_perp']),
            bw_aniso_ratio=float(s['bw_aniso_ratio']))
        backend = 'gpu'
    else:
        delta, new_dirs = _ms_step(
            cur_pts, cur_dirs,
            bw_cur, float(s['dir_alpha']),
            int(s['k_search']),
            update_dirs=bool(s['update_dirs']),
            dir_asym_gamma=float(s['dir_asym_gamma']),
            gauss_penalty_par=float(s['gauss_penalty_par']),
            gauss_penalty_perp=float(s['gauss_penalty_perp']),
            bw_aniso_ratio=float(s['bw_aniso_ratio']))
        backend = 'cpu'
    ms_dt = time.time() - ms_t0
    if new_dirs is not None:
        s['dirs'] = new_dirs
    d_norm = np.linalg.norm(delta, axis=1)
    new_pts = (cur_pts + delta).astype(np.float32)
    # iter > topo_lock_iter: two-sided spring holds the frozen topology
    # (applied BEFORE s['pts'] so history / scrubber stay consistent)
    if (int(s['iter']) + 1 > int(s.get('topo_lock_iter', 5))
            and bool(s.get('topo_lock_on', True))
            and s.get('topo_links_edges') is not None):
        new_pts, _tn, _tmx = _apply_topo_force(new_pts)
    else:
        _tn, _tmx = 0, 0.0

    s['pts'] = new_pts
    s['history'].append(new_pts.copy())
    s['iter'] += 1
    s['global_iter'] = int(s.get('global_iter', 0)) + 1
    s['history_full'].append(_snapshot_state())
    s['view_iter']  = s['iter']
    s['shown_iter'] = s['iter']

    if (int(s['iter']) == int(s.get('topo_lock_iter', 5))
            and bool(s.get('topo_lock_on', True))):
        _compute_topo_links()       # freeze the local-graph here

    head = f'iter {s["iter"]:>3d}'
    print(f'  {head}  ms shift  '
          f'mean={float(d_norm.mean()):.4f}  '
          f'p99={float(np.percentile(d_norm, 99)):.4f}  '
          f'max={float(d_norm.max()):.4f}  '
          f'(bw={bw_cur:.2f}  k={int(s["k_search"])}  '
          f'{backend}{"  update_dirs" if bool(s["update_dirs"]) else ""})  '
          f'  total={ms_dt * 1000:.0f}ms'
          f'{f"  +topo pull {_tn}pts max{_tmx:.2f}" if _tn else ""}')


def _topo_links_euler(P: np.ndarray, dirs: np.ndarray, tree,
                      cone_deg: float, reach: float) -> np.ndarray:
    """STREAMLINE topo-links — a true GEOMETRIC cone query (no kNN cutoff).
    For each point: take ALL points within `reach` vox (radius graph),
    keep those that fall inside its forward (+dir) / backward (−dir) cone
    of half-angle `cone_deg`, and link all of them — long-range axial
    links that span the strand.  Pure-
    lateral (perpendicular) cross-section pairs are excluded, so MS can
    still thin the tube and strands aren't strutted across.  A point with
    no direction gets no link.  Returns (E,2) undirected unique int32
    edges."""
    D = np.asarray(dirs, np.float64)
    nrm = np.linalg.norm(D, axis=1)
    Dn = D / np.maximum(nrm[:, None], 1e-12)
    has = nrm > 1e-9
    cos_t = float(np.cos(np.radians(cone_deg)))
    E = tree.query_pairs(float(reach), output_type='ndarray')   # i<j uniq
    n_pairs = len(E)
    if n_pairs == 0:
        print(f'  [topo-euler] query_pairs={n_pairs}: no pair within '
              f'reach={reach:g}')
        return np.zeros((0, 2), np.int32)
    a, b = E[:, 0], E[:, 1]
    seg = (P[b] - P[a]).astype(np.float64)
    d = np.linalg.norm(seg, axis=1)
    ev = seg / np.maximum(d[:, None], 1e-12)
    both = has[a] & has[b]
    in_a = both & (np.abs(np.einsum('ij,ij->i', ev, Dn[a])) >= cos_t)  # b∈cone(a)
    in_b = both & (np.abs(np.einsum('ij,ij->i', ev, Dn[b])) >= cos_t)  # a∈cone(b)
    n_in_a = int(in_a.sum()); n_in_b = int(in_b.sum())
    print(f'  [topo-euler] query_pairs={n_pairs:,}  cone={cone_deg:g}°  '
          f'reach={reach:g}  '
          f'cone-pass: b∈cone(a)={n_in_a:,}  a∈cone(b)={n_in_b:,}')
    # directed candidates (src's cone contains dst)
    src = np.concatenate([a[in_a], b[in_b]])
    dst = np.concatenate([b[in_a], a[in_b]])
    n_directed = len(src)
    if n_directed == 0:
        print(f'  [topo-euler] no directed cone candidate')
        return np.zeros((0, 2), np.int32)
    E = np.unique(np.sort(np.stack([src, dst], axis=1), axis=1), axis=0)
    E = E[E[:, 0] != E[:, 1]]
    print(f'  [topo-euler] dedup → undirected unique edges={len(E):,}')
    return E.astype(np.int32)


# Topology-lock parameters, fixed (formerly GUI sliders)
_TOPO_LINK_REACH = 10.0   # vox: cone length (search radius) + per-step force clamp
_TOPO_FORCE_W    = 1.0    # two-sided spring weight after the lock iter (0 = off)


def _compute_topo_links() -> None:
    """At iter == topo_lock_iter (toggle on): freeze a STREAMLINE spring
    lattice (`_topo_links_euler`: per point, all points inside its ±dir
    cone, within `_TOPO_LINK_REACH` vox).  Each
    edge's freeze-iter length becomes its rest length for the iter>lock
    two-sided spring.  Visualise."""
    s = state
    P = np.asarray(s['pts'], np.float64)
    n = len(P)
    if n < 3:
        print('  [topo] too few points'); return
    t0 = time.perf_counter()
    _dirs = s.get('dirs')
    cone = float(s.get('topo_link_cone_deg', 30.0))
    reach = _TOPO_LINK_REACH
    if ps.has_curve_network('topo_links'):
        ps.remove_curve_network('topo_links')
    if _dirs is None or len(_dirs) != n:
        s['topo_links_edges'] = np.zeros((0, 2), np.int32)
        s['topo_links_rest']  = np.zeros(0)
        s['topo_links_iter']  = int(s['iter'])
        print(f'  [topo] iter={s["iter"]}  no valid dir field — no links  '
              f'[{time.perf_counter() - t0:.2f}s]')
        return
    tree = cKDTree(P)
    E = _topo_links_euler(P, _dirs, tree, cone, reach)
    _info = f'cone={cone:g}° reach={reach:g}vox'
    if len(E) == 0:
        s['topo_links_edges'] = np.zeros((0, 2), np.int32)
        s['topo_links_rest']  = np.zeros(0)
        s['topo_links_iter']  = int(s['iter'])
        print(f'  [topo] iter={s["iter"]}  ({_info})  NO edges — check '
              f'dirs / loosen cone / reach  '
              f'[{time.perf_counter() - t0:.2f}s]')
        return
    E = E.astype(np.int32)
    R0 = np.linalg.norm(P[E[:, 0]] - P[E[:, 1]], axis=1)
    s['topo_links_edges'] = E                     # GLOBAL idx (frozen)
    s['topo_links_rest']  = R0                    # freeze-iter rest lengths
    s['topo_links_iter']  = int(s['iter'])
    # register ONLY points that take part in an edge (no stray nodes)
    uniq, inv = np.unique(E.reshape(-1), return_inverse=True)
    cn = ps.register_curve_network(
        'topo_links', P[uniq].astype(np.float32),
        inv.reshape(-1, 2).astype(np.int32))
    cn.set_color((0.15, 0.95, 0.45))
    cn.set_radius(0.00015)
    cn.set_enabled(False)        # default OFF — overlay clutters main viz
    deg = np.bincount(E.reshape(-1), minlength=n)
    n_iso = int((deg == 0).sum())
    print(f'  [topo] iter={s["iter"]}  STREAMLINE ({_info})  '
          f'edges={len(E):,}  nodes={len(uniq):,}/{n:,} '
          f'({100.0 * len(uniq) / n:.0f}% covered)  '
          f'isolated={n_iso:,}  '
          f'[{time.perf_counter() - t0:.2f}s]')


def _apply_topo_force_cpu(P: np.ndarray):
    """CPU path.  iter>10 TWO-SIDED spring (constant weight
    `_TOPO_FORCE_W`) to the
    iter-10 rest length: for each FROZEN topo edge, restore its endpoint
    distance toward what it was at iter 10 — pull together if MS has
    STRETCHED it, push apart if MS has SQUEEZED it.  The push-apart half
    is what stops mean-shift from compressing the axial structure into
    one centre (the one-sided version could not).  An interior chain
    point is in 2 edges → feels both neighbours (np.add.at sums).  Extra
    term on top of the MS shift.  Vectorised; returns
    (new_P, n_points_moved, max_correction)."""
    s = state
    E = s.get('topo_links_edges')
    R0 = s.get('topo_links_rest')
    if E is None or R0 is None or len(E) == 0:
        return P, 0, 0.0
    w = _TOPO_FORCE_W
    if w <= 0.0:
        return P, 0, 0.0
    Pf = np.asarray(P, np.float64)
    a = E[:, 0].astype(np.int64)
    b = E[:, 1].astype(np.int64)
    seg = Pf[b] - Pf[a]                            # a→b  (M,3)
    cur = np.linalg.norm(seg, axis=1)              # (M,)
    dev = cur - np.asarray(R0, np.float64)         # signed: + stretch − squeeze
    m = cur > 1e-9                                 # safe (skip coincident)
    if not m.any():
        return P, 0, 0.0
    u = seg[m] / cur[m, None]                      # a→b unit
    # signed Hookean: dev>0 → pull a,b together; dev<0 → push them apart
    f = (w * dev[m])[:, None] * u
    disp = np.zeros_like(Pf)
    np.add.at(disp, a[m],  f)
    np.add.at(disp, b[m], -f)
    # STABILITY: AVERAGE (not sum) at shared nodes — a degree-2 chain
    # point must not get 2× the push; summed-Jacobi on a coupled chain
    # is unstable (the exponential blow-up).  Then HARD-CLAMP each
    # point's per-step move to topo_max_hop so no transient can ever
    # snowball, whatever w / MS does.  No new parameter.
    deg = np.zeros(len(Pf))
    np.add.at(deg, a[m], 1.0)
    np.add.at(deg, b[m], 1.0)
    np.divide(disp, deg[:, None], out=disp, where=deg[:, None] > 0.0)
    cap = _TOPO_LINK_REACH
    dn = np.linalg.norm(disp, axis=1)
    np.multiply(disp, (cap / np.maximum(dn, 1e-12))[:, None],
                out=disp, where=(dn > cap)[:, None])
    out = (Pf + disp).astype(np.float32)
    moved = np.unique(np.concatenate([a[m], b[m]]))
    return out, int(len(moved)), float(np.linalg.norm(disp, axis=1).max())


# GPU mirror (cupyx.scatter_add).  The FROZEN graph (a,b,R0) is
# uploaded ONCE per freeze and kept resident — only the per-iter
# positions transfer.  Math in float64 to match the CPU path (scatter
# accumulation order differs → ~1e-9, negligible).
_TOPO_GPU = {'iter': -2, 'a': None, 'b': None, 'R0': None}


def _apply_topo_force_gpu(P: np.ndarray):
    """GPU path — numerically equivalent to _apply_topo_force_cpu."""
    s = state
    E = s.get('topo_links_edges')
    R0 = s.get('topo_links_rest')
    if E is None or R0 is None or len(E) == 0:
        return P, 0, 0.0
    w = _TOPO_FORCE_W
    if w <= 0.0:
        return P, 0, 0.0
    it = int(s.get('topo_links_iter', -1))
    if _TOPO_GPU['iter'] != it or _TOPO_GPU['a'] is None:
        _TOPO_GPU['a'] = _cp.asarray(E[:, 0].astype(np.int64))
        _TOPO_GPU['b'] = _cp.asarray(E[:, 1].astype(np.int64))
        _TOPO_GPU['R0'] = _cp.asarray(np.asarray(R0, np.float64))
        _TOPO_GPU['iter'] = it
    a, b, R0g = _TOPO_GPU['a'], _TOPO_GPU['b'], _TOPO_GPU['R0']
    Pg = _cp.asarray(np.asarray(P, np.float64))
    seg = Pg[b] - Pg[a]
    cur = _cp.linalg.norm(seg, axis=1)
    dev = cur - R0g                                # signed
    m = cur > 1e-9
    if not bool(m.any()):
        return P, 0, 0.0
    am, bm = a[m], b[m]
    u = seg[m] / cur[m, None]
    f = (w * dev[m])[:, None] * u
    disp = _cp.zeros_like(Pg)
    _cpx.scatter_add(disp, am,  f)
    _cpx.scatter_add(disp, bm, -f)
    deg = _cp.zeros(len(Pg))
    _cpx.scatter_add(deg, am, 1.0)
    _cpx.scatter_add(deg, bm, 1.0)
    nz = deg > 0.0
    disp[nz] = disp[nz] / deg[nz, None]            # average@shared
    cap = _TOPO_LINK_REACH
    dn = _cp.linalg.norm(disp, axis=1)
    big = dn > cap
    if bool(big.any()):                            # hard per-step clamp
        disp[big] = disp[big] * (
            cap / _cp.maximum(dn[big], 1e-12))[:, None]
    out = _cp.asnumpy(Pg + disp).astype(np.float32)
    moved = int(_cp.unique(_cp.concatenate([am, bm])).size)
    return out, moved, float(_cp.linalg.norm(disp, axis=1).max())

def _apply_topo_force(P: np.ndarray):
    """Dispatch CPU/GPU exactly like the MS step (ms_gpu + cupy)."""
    if bool(state.get('ms_gpu')) and _cupy_available:
        return _apply_topo_force_gpu(P)
    return _apply_topo_force_cpu(P)

def _hide_orig_viz() -> None:
    """Disable the raw `points` cloud + direction_lines so that derived
    output (the fitted curves) is the dominant signal in the view."""
    if ps.has_point_cloud('points'):
        ps.get_point_cloud('points').set_enabled(False)
    if ps.has_curve_network('direction_lines'):
        ps.get_curve_network('direction_lines').set_enabled(False)

def _maybe_show_orig_viz() -> None:
    """Re-enable raw viz only when every derived layer has been cleared."""
    if _curve_data:
        return
    if ps.has_point_cloud('points'):
        ps.get_point_cloud('points').set_enabled(True)
    if ps.has_curve_network('direction_lines'):
        ps.get_curve_network('direction_lines').set_enabled(True)

def _snapshot_state() -> dict:
    """Capture current visualization-relevant state for history rewind."""
    s = state
    return {
        'pts':  s['pts'].copy(),
        'dirs': s['dirs'].copy(),
    }


def _render_iter(it: int) -> None:
    """Restore all viz to a past iter snapshot (does NOT mutate live state).
    Updates: points, direction_lines, selected_pt, alive subsets, inspect."""
    s = state
    if not s['history_full'] or it < 0 or it >= len(s['history_full']):
        return
    snap = s['history_full'][it]

    pts_snap = snap['pts']
    dirs_snap = snap['dirs']
    if ps.has_point_cloud('points'):
        ps.get_point_cloud('points').update_point_positions(pts_snap)
    if ps.has_curve_network('direction_lines'):
        dir_nodes = np.vstack(
            [pts_snap, pts_snap + DIR_LEN * dirs_snap]).astype(np.float32)
        ps.get_curve_network('direction_lines').update_node_positions(dir_nodes)
    _refresh_datapt_colors(dirs_snap)
    pi = int(s['picked'])
    if 0 <= pi < N and ps.has_point_cloud('selected_pt'):
        ps.get_point_cloud('selected_pt').update_point_positions(
            pts_snap[pi:pi + 1])

    # Inspect viz follows the scrubber
    _ipi = int(s.get('inspect_picked_idx', -1))
    if _ipi >= 0:
        _inspect_picked(_ipi)


def _gabor_color(dirs_arr: np.ndarray) -> np.ndarray:
    """Standard Gabor mapping used everywhere: |d|^0.6 clipped to [0,1]."""
    return np.clip(np.abs(dirs_arr) ** 0.6, 0.0, 1.0).astype(np.float32)


def _refresh_datapt_colors(dirs_arr: np.ndarray) -> None:
    """Re-apply Gabor color from the given dirs to the data-point viz
    (`points` cloud + `direction_lines` per-edge color). MS may have
    updated dirs in place, but polyscope colors stay frozen unless we
    re-push them — that's why the toy looked 'noisy' post-MS even when
    the math was equivalent to meanshift."""
    gabor = _gabor_color(dirs_arr)
    if ps.has_point_cloud('points'):
        ps.get_point_cloud('points').add_color_quantity(
            'direction_RGB', gabor, enabled=True)
    if ps.has_curve_network('direction_lines'):
        ps.get_curve_network('direction_lines').add_color_quantity(
            'direction_RGB', gabor, defined_on='edges', enabled=True)



# ── Click-inspect: bw ellipsoid + kNN highlight at picked point ──────────
_INSPECT_NAME = 'inspect_bw_ellipsoid'


def _clear_inspect_viz() -> None:
    """Remove the inspect viz (used when picking is cleared or before
    re-registering with new data)."""
    try:
        if ps.has_curve_network(_INSPECT_NAME):
            ps.remove_curve_network(_INSPECT_NAME)
    except Exception:
        pass


def _displayed_state() -> dict:
    """Return the {'pts','dirs'} dict corresponding to view_iter (or live
    state if no history). Used by inspect so the bw ball / kNN follow the
    history scrubber."""
    s = state
    vi = int(s.get('view_iter', s['iter']))
    if s.get('history_full') and 0 <= vi < len(s['history_full']):
        return s['history_full'][vi]
    return {'pts': s['pts'], 'dirs': s['dirs']}


def _draw_bw_ellipsoid(center: np.ndarray, axis: np.ndarray,
                       sigma_par: float, sigma_perp: float,
                       n_seg: int = 48,
                       name: str = _INSPECT_NAME,
                       color: tuple[float, float, float] = (0.30, 0.85, 0.95),
                       radius: float = 0.04) -> None:
    """Render the anisotropic kernel boundary as a 3-ring wireframe
    ellipsoid: prolate along `axis` (σ∥=sigma_par), circular ⊥ (σ⊥=
    sigma_perp). Rings are great-circle slices through axis × e1, axis
    × e2, and e1 × e2."""
    axis = np.asarray(axis, dtype=np.float64)
    an   = np.linalg.norm(axis)
    if an < 1e-8:
        axis = np.array([0.0, 0.0, 1.0])
    else:
        axis = axis / an
    if abs(axis[2]) < 0.9:
        e1 = np.cross(axis, np.array([0.0, 0.0, 1.0]))
    else:
        e1 = np.cross(axis, np.array([1.0, 0.0, 0.0]))
    e1 /= max(np.linalg.norm(e1), 1e-8)
    e2 = np.cross(axis, e1)

    theta = np.linspace(0, 2 * np.pi, n_seg, endpoint=False)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    sp = float(sigma_par)
    sl = float(sigma_perp)
    # 3 rings: (axis, e1), (axis, e2), (e1, e2)
    r1 = (center[None, :]
          + sp * cos_t[:, None] * axis[None, :]
          + sl * sin_t[:, None] * e1[None, :])
    r2 = (center[None, :]
          + sp * cos_t[:, None] * axis[None, :]
          + sl * sin_t[:, None] * e2[None, :])
    r3 = (center[None, :]
          + sl * cos_t[:, None] * e1[None, :]
          + sl * sin_t[:, None] * e2[None, :])
    nodes = np.vstack([r1, r2, r3]).astype(np.float32)
    n = n_seg
    arr_n  = np.arange(n, dtype=np.int32)
    arr_n2 = np.roll(arr_n, -1)
    edges = np.vstack([
        np.column_stack([arr_n,       arr_n2]),
        np.column_stack([arr_n + n,   arr_n2 + n]),
        np.column_stack([arr_n + 2*n, arr_n2 + 2*n]),
    ])

    if ps.has_curve_network(name):
        ps.remove_curve_network(name)
    cn = ps.register_curve_network(name, nodes, edges)
    cn.set_color(color)
    cn.set_transparency(0.55)
    cn.set_radius(radius, relative=False)


def _inspect_picked(idx: int) -> None:
    """Render bw ellipsoid + kNN highlight for the picked point at the
    CURRENTLY-DISPLAYED iter state (via _displayed_state). Re-callable;
    each call refreshes the viz. idx < 0 → clear inspection."""
    s = state
    if idx < 0:
        _clear_inspect_viz()
        return
    snap = _displayed_state()
    pts_cur  = snap.get('pts')
    dirs_cur = snap.get('dirs')
    if pts_cur is None or dirs_cur is None or idx >= len(pts_cur):
        _clear_inspect_viz()
        return

    p = pts_cur[idx].astype(np.float64)
    d = dirs_cur[idx].astype(np.float64)
    dn = np.linalg.norm(d)
    if dn > 1e-8:
        d = d / dn

    # bw at displayed iter (clamped to schedule end)
    vi = int(s.get('view_iter', s['iter']))
    n_iter = max(int(s['n_iter']), 1)
    vi_for_bw = max(0, min(vi, n_iter - 1))
    bw = _scheduled_bw(vi_for_bw, n_iter,
                       float(s['bw_start']), float(s['bw_end']))
    ratio = float(s['bw_aniso_ratio'])
    sigma_par  = float(bw * ratio)
    sigma_perp = float(bw)

    # Keep ONLY the blue bw ellipsoid on click (kNN highlight + golden
    # centre ball + console summary removed per user request).
    _clear_inspect_viz()
    _draw_bw_ellipsoid(p, d, sigma_par, sigma_perp)
    return None






_LDIR_HALF = 2   # local direction window half-size: a NON-tip point uses
#                  _LDIR_HALF before + _LDIR_HALF after (= 4 pts at 2);
#                  a TIP has no "after" → take 2*_LDIR_HALF inward instead.








def _save_yarn(tag: str = 'manual', backup: bool = True) -> None:
    """Unified npz save (replaces the old curves_state.pkl + yarn.npz pair).
    Writes a SELF-CONTAINED npz that is BOTH (a) the crochet-verification
    export and (b) fully reloadable by `_load_yarn` to resume editing.
    Stores curve geometry (`curve_<sub_id>`, plus a single `yarn` key when
    everything is ONE strand) AND the reload metadata that curves_state.pkl
    used to hold: seg labels, K, sub→cc and smooth.  Always
    `yarn_latest.npz`; when `backup` (the manual Save), also writes a
    timestamped `yarn_<ts>.npz` that is never overwritten.  Pure arrays — no
    pickle (loads with allow_pickle=False)."""
    if not _curve_data:
        print('  [save-yarn] no curves — Fit curves / connect first')
        return
    s = state

    def _alen(P):
        return (float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())
                if len(P) > 1 else 0.0)

    curves = [(int(sid), np.asarray(sp, np.float64))
              for sid, sp in _curve_data]
    k = len(curves)
    out = {f'curve_{sid}': sp for sid, sp in curves}
    out['sub_ids']  = np.asarray([sid for sid, _ in curves], np.int64)
    out['n_curves'] = np.asarray(k)
    out['fmt']      = np.asarray('yarn_state_v1')
    if k == 1:
        out['yarn'] = curves[0][1]                  # the single crochet strand
    # ── reload metadata (mirror of the old curves_state payload) ──
    out['curve_K']      = np.asarray(int(s.get('curve_K', 0)))
    out['curve_smooth'] = np.asarray(_CURVE_SMOOTH)
    sub2cc = dict(s.get('curve_sub_to_cc') or {})
    if sub2cc:                                       # dict → two int arrays
        out['sub2cc_keys'] = np.asarray(list(sub2cc.keys()), np.int64)
        out['sub2cc_vals'] = np.asarray(list(sub2cc.values()), np.int64)
    if s.get('curve_seg_labels') is not None:
        out['curve_seg_labels'] = np.asarray(s['curve_seg_labels'])
    files = [os.path.join(_OUT_CURVES, 'yarn_latest.npz')]
    if backup:
        files.append(os.path.join(
            _OUT_CURVES,
            f'yarn_{time.strftime("%Y%m%d_%H%M%S")}.npz'))
    for fn in files:
        np.savez(fn, **out)
    saved = ' + '.join(files)
    if k == 1:
        sp = curves[0][1]
        print(f'  [save-yarn] {saved}  [{tag}]  ✓ SINGLE yarn: {len(sp)} '
              f'pts, length {_alen(sp):.0f} vox  → crochet-ready (key '
              f'"yarn"); reload via "Load yarn"')
    else:
        tot = sum(_alen(sp) for _, sp in curves)
        print(f'  [save-yarn] {saved}  [{tag}]  {k} separate curves '
              f'(total {tot:.0f} vox) — NOT one strand yet; reloadable '
              f'via "Load yarn"')


def _save_curves_state(tag: str = 'fit') -> None:
    """Back-compat shim — the auto-checkpoint after Fit/heatmap now writes
    the UNIFIED yarn npz (canonical `yarn_latest.npz`, no timestamped backup
    so fits don't spam files).  Manual "Save yarn" still writes a timestamped
    deliverable.  Replaces the old curves_state.pkl dual-write."""
    _save_yarn(tag, backup=False)


def _load_yarn(path: str | None = None) -> None:
    """Reload a yarn npz written by `_save_yarn` (the UNIFIED save) back into
    the editor — restores _curve_data + labels/K/sub→cc/smooth,
    undo, then re-detects so the sticks/balls are ready to pick.  `path`
    defaults to the canonical `yarn_latest.npz`, else the newest
    `yarn_*.npz`.  Also accepts an OLD geometry-only export (just
    `yarn`/`curve_*`, no metadata).  READ-ONLY (allow_pickle=False)."""
    global _curve_data
    import glob
    s = state
    if path is None:
        # search order: output/<stem>/curves/ → CWD (legacy)
        _candidates = [os.path.join(_OUT_CURVES, 'yarn_latest.npz'),
                       'yarn_latest.npz']
        for _p in _candidates:
            if os.path.exists(_p):
                path = _p; break
        if path is None:
            cand = (sorted(glob.glob(
                        os.path.join(_OUT_CURVES, 'yarn_*.npz')))
                    + sorted(glob.glob('yarn_*.npz')))
            if not cand:
                print(f'  [load-yarn] no yarn_*.npz in '
                      f'{_OUT_CURVES}/ or CWD — Save yarn / Fit first')
                return
            path = cand[-1]
    if not os.path.exists(path):
        print(f'  [load-yarn] {path} not found'); return
    d = np.load(path, allow_pickle=False)            # read-only, no pickle
    f = set(d.files)
    # ── reconstruct curves (ordered by sub_ids; fall back to old export) ──
    if 'sub_ids' in f:
        cd = [(int(sid), np.asarray(d[f'curve_{sid}'], np.float64))
              for sid in d['sub_ids'] if f'curve_{int(sid)}' in f]
    else:
        cd = [(int(k.split('_', 1)[1]), np.asarray(d[k], np.float64))
              for k in sorted(f) if k.startswith('curve_')]
    if not cd and 'yarn' in f:                       # old single-strand export
        cd = [(0, np.asarray(d['yarn'], np.float64))]
    if not cd:
        print(f'  [load-yarn] {path}: no curve_*/yarn arrays; keys={d.files}')
        return
    _curve_data = cd
    s['curve_K'] = (int(d['curve_K']) if 'curve_K' in f
                    else max(sid for sid, _ in cd) + 1)
    s['curve_sub_to_cc'] = ({int(a): int(b) for a, b in
                             zip(d['sub2cc_keys'], d['sub2cc_vals'])}
                            if {'sub2cc_keys', 'sub2cc_vals'} <= f else {})
    s['curve_seg_labels'] = (d['curve_seg_labels']
                             if 'curve_seg_labels' in f else None)
    Kpal = max([sid for sid, _ in _curve_data], default=-1) + 1
    _register_curves(_curve_data, max(Kpal, 1))
    fn = path
    _hide_orig_viz()                   # also hide raw `points`
    print(f'  [load-yarn] restored {os.path.basename(fn)}: '
          f'{len(_curve_data)} curves.')


















# ── Curve fitting (ported from meanshift_centers.py) ─────────────────────
# Pipeline: _compute_segments → seg_labels (CC of a proximity graph) →
# per-segment _order_points (MST diameter path) → _fit_spline (B-spline) →
# polyscope curve_network registration.
_PHI               = 0.618033988749895
                               # arc-length-proportional sampling in _fit_spline
_SPLINE_PTS_MAX    = 200000    # hard cap on samples/curve (guards tiny spacing)
_SPLINE_SUBSAMPLE  = 10 ** 9   # cap on points fed into MST ordering (off)
_CURVE_RADIUS      = 0.09      # × _SCALE, absolute tube radius of 'curves'
# Curve-fit parameters, fixed (formerly GUI sliders)
_CURVE_SMOOTH         = 1.5     # B-spline smoothing: s = _CURVE_SMOOTH * n_pts
_CURVE_POINT_SPACING  = 1.0     # vox per sample along a fitted curve (∝ arc length)
_CURVE_SPLIT_TURN_DEG = 180.0   # cut at a local-max windowed turn above this (180 = off)
_CURVE_MIN_LENGTH_VOX = 20.0    # drop fitted curves shorter than this (vox)
_curve_data: list[tuple[int, np.ndarray]] = []   # (seg_id, spline_pts)


def _seg_palette(K: int) -> np.ndarray:
    """(K, 3) float64 RGB with golden-ratio hue spacing."""
    pal = np.zeros((max(K, 1), 3), dtype=np.float64)
    for i in range(K):
        h = (i * _PHI) % 1.0
        sat = 0.80 + 0.15 * (i % 2)
        val = 0.95 - 0.15 * ((i // 4) % 2)
        pal[i] = colorsys.hsv_to_rgb(h, sat, val)
    return pal


def _compute_segments(pts: np.ndarray, radius: float, min_pts: int
                      ) -> tuple[np.ndarray, int]:
    """Radius-proximity graph → connected components → drop components
    smaller than min_pts. seg_labels[i] = -1 → point discarded."""
    print(f'\n  Building proximity graph  radius={radius:.2f} ...')
    tree  = KDTree(pts)
    pairs = tree.query_pairs(r=radius, output_type='ndarray')
    print(f'  Edges: {len(pairs):,}')
    if len(pairs) == 0:
        print('  [warn] No edges — try a larger radius.')
        return np.full(len(pts), -1, dtype=np.int32), 0

    M = len(pts)
    rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
    cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
    graph = csr_matrix((np.ones(len(rows), dtype=np.float32),
                        (rows, cols)), shape=(M, M))

    n_comp, labels = connected_components(graph, directed=False)
    print(f'  Raw components: {n_comp:,}')
    sizes = np.bincount(labels, minlength=n_comp)
    valid = np.where(sizes >= min_pts)[0]
    order = valid[np.argsort(-sizes[valid])]            # largest first
    old_to_new = np.full(n_comp, -1, dtype=np.int32)
    for new_id, old_id in enumerate(order):
        old_to_new[old_id] = new_id
    seg_labels = old_to_new[labels]
    K = int(seg_labels.max()) + 1 if (seg_labels >= 0).any() else 0
    kept = int((seg_labels >= 0).sum())
    print(f'  After min_pts={min_pts}: {K} segments, '
          f'{kept:,} / {M:,} points kept')
    return seg_labels.astype(np.int32), K


def _order_via_topo_mst(global_indices: np.ndarray,
                         src: np.ndarray,
                         global_adj: dict,
                         n_orig: int | None = None) -> np.ndarray:
    """Order a CC's points along their MST diameter using a PRECOMPUTED
    global adjacency (typically `state['topo_mst_global_adj']`).

    Unlike `_order_points` it does NOT rebuild the MST, so every stitch or
    edit made to topo_mst flows through to Fit curves (a >-----< loop
    stitched into one chain yields ONE B-spline, not two).  The diameter
    comes from two arc-length-weighted BFS, which is exact on a tree.  Only
    the LARGEST connected sub-component is used, and synthetic centerline
    pts (gidx ≥ n_orig) joined through `global_adj` are included, so a
    stitched centerline becomes part of the fitted spline.
    """
    if len(global_indices) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    if n_orig is None:
        n_orig = len(src)
    cc_set = set(int(g) for g in global_indices)
    if len(src) > n_orig:
        # synthetic centerline gidx (from a stitched loop) — let BFS
        # reach them so the spline goes through the centerline
        cc_set |= set(range(n_orig, len(src)))

    # ── Find the largest connected component within cc_set under
    # `global_adj`.  After a stitch, half-B becomes a set of
    # isolated singletons (or small fragments) and half-A keeps the
    # bulk — we want the bulk.
    seen_all: set[int] = set()
    best_comp: set[int] = set()
    for seed0 in cc_set:
        if seed0 in seen_all:
            continue
        comp: set[int] = set()
        stk = [seed0]
        while stk:
            u = stk.pop()
            if u in comp:
                continue
            comp.add(u)
            for v in global_adj.get(u, ()):
                iv = int(v)
                if iv in cc_set and iv not in comp:
                    stk.append(iv)
        seen_all |= comp
        if len(comp) > len(best_comp):
            best_comp = comp
    if len(best_comp) < 2:
        return (src[[next(iter(best_comp))]].astype(np.float64)
                if best_comp else np.zeros((0, 3), dtype=np.float64))

    def _bfs_far(start: int) -> tuple:
        dist:   dict[int, float] = {start: 0.0}
        parent: dict[int, int]   = {start: -1}
        queue = [start]
        qi = 0
        far_u, far_d = start, 0.0
        while qi < len(queue):
            u = queue[qi]; qi += 1
            for v in global_adj.get(u, ()):
                iv = int(v)
                if iv in best_comp and iv not in dist:
                    edge_len = float(np.linalg.norm(src[u] - src[iv]))
                    dist[iv]   = dist[u] + edge_len
                    parent[iv] = u
                    queue.append(iv)
                    if dist[iv] > far_d:
                        far_d, far_u = dist[iv], iv
        return far_u, parent

    seed = next(iter(best_comp))
    far_u, _      = _bfs_far(seed)
    far_v, parent = _bfs_far(far_u)
    # reconstruct path far_v → ... → far_u
    path: list[int] = []
    cur = far_v
    while cur != -1:
        path.append(cur)
        cur = parent[cur]
    if len(path) < 2:
        return src[[far_v]].astype(np.float64)
    return src[np.asarray(path)].astype(np.float64)


def _peel_diameter_paths(global_indices: np.ndarray,
                         src: np.ndarray,
                         global_adj: dict,
                         n_orig: int,
                         min_len: int = 4) -> list:
    """Iteratively peel diameter paths from the MST sub-graph induced by
    `global_indices` plus any synthetic gidx (≥ n_orig) reachable through
    `global_adj`.

    Repeatedly takes the LARGEST connected component, extracts its
    arc-length-weighted diameter path, removes those nodes and repeats,
    stopping once every component is shorter than `min_len`.  One path per
    chain in the tree, so the user gets a curve per branch instead of just
    the trunk.

    Returns (paths, n_uncovered): ordered (M, 3) arrays with M ≥ min_len,
    and how many real-cloud gidx never landed on a path."""
    if len(global_indices) == 0:
        return [], 0
    cc_set = set(int(g) for g in global_indices)
    if len(src) > n_orig:
        cc_set |= set(range(n_orig, len(src)))
    # adj_loc[g] = neighbours restricted to cc_set
    adj_loc: dict[int, set] = {}
    for g in cc_set:
        nbs = {int(v) for v in global_adj.get(int(g), ())
               if int(v) in cc_set}
        if nbs:
            adj_loc[int(g)] = nbs
    paths: list = []

    def _largest_cc(_adj):
        seen: set = set()
        best: set = set()
        for s0 in _adj:
            if s0 in seen:
                continue
            comp: set = set()
            stk = [s0]
            while stk:
                u = stk.pop()
                if u in comp:
                    continue
                comp.add(u)
                for v in _adj.get(u, ()):
                    if v not in comp:
                        stk.append(v)
            seen |= comp
            if len(comp) > len(best):
                best = comp
        return best

    def _diameter(_adj, cc):
        def _bfs_far(start):
            dist = {start: 0.0}
            par  = {start: -1}
            q = [start]; qi = 0
            far_u, far_d = start, 0.0
            while qi < len(q):
                u = q[qi]; qi += 1
                for v in _adj.get(u, ()):
                    if v in cc and v not in dist:
                        el = float(np.linalg.norm(src[u] - src[v]))
                        dist[v] = dist[u] + el
                        par[v]  = u
                        q.append(v)
                        if dist[v] > far_d:
                            far_d, far_u = dist[v], v
            return far_u, par
        seed = next(iter(cc))
        a, _   = _bfs_far(seed)
        b, par = _bfs_far(a)
        out = []; cur = b
        while cur != -1:
            out.append(cur); cur = par[cur]
        return out

    while True:
        cc = _largest_cc(adj_loc)
        if len(cc) < min_len:
            break
        path_gidx = _diameter(adj_loc, cc)
        if len(path_gidx) < min_len:
            # this CC has no path ≥ min_len; clear it so we move on
            for g in cc:
                if g in adj_loc:
                    for nb in list(adj_loc[g]):
                        if nb in adj_loc:
                            adj_loc[nb].discard(g)
                    del adj_loc[g]
            continue
        paths.append(src[np.asarray(path_gidx)].astype(np.float64))
        # remove the path's nodes from adj_loc — the rest of the CC
        # will fall apart into smaller pieces in the next iteration.
        for g in path_gidx:
            if g in adj_loc:
                for nb in list(adj_loc[g]):
                    if nb in adj_loc:
                        adj_loc[nb].discard(g)
                del adj_loc[g]
    # count real-cloud gidx never covered (small leftover bits)
    n_uncovered = sum(1 for g in adj_loc
                      if 0 <= int(g) < n_orig)
    return paths, n_uncovered


def _order_points(pts: np.ndarray) -> np.ndarray:
    """Order 3-D points along a curve via MST diameter path."""
    M = len(pts)
    if M <= 2:
        return pts

    if M > _SPLINE_SUBSAMPLE:
        rng      = np.random.default_rng(42)
        centered = pts - pts.mean(axis=0, keepdims=True)
        axis     = np.linalg.svd(centered, full_matrices=False)[2][0]
        proj     = centered @ axis
        end_a, end_b = int(np.argmin(proj)), int(np.argmax(proj))
        seed = np.unique([end_a, end_b])
        pool = np.setdiff1d(np.arange(M), seed)
        take = rng.choice(pool,
                          size=min(_SPLINE_SUBSAMPLE - len(seed), len(pool)),
                          replace=False)
        idx  = np.sort(np.concatenate([seed, take]))
        pts  = pts[idx]
        M    = len(pts)

    local_tree = KDTree(pts)
    g = None
    for k in [min(k, M - 1) for k in [8, 12, 16, 24, 32, 48, M - 1]]:
        if k <= 0:
            continue
        d_nn, i_nn = local_tree.query(pts, k=k + 1)
        rows = np.repeat(np.arange(M, dtype=np.int32), k)
        cols = i_nn[:, 1:k + 1].reshape(-1).astype(np.int32)
        data = d_nn[:, 1:k + 1].reshape(-1).astype(np.float64)
        valid_e = np.isfinite(data) & (rows != cols) & (data <= 4.0)
        g = csr_matrix((data[valid_e], (rows[valid_e], cols[valid_e])),
                       shape=(M, M))
        g = g.maximum(g.T)
        n_cc, _ = connected_components(g, directed=False)
        if n_cc == 1:
            break
    if g is None:
        return pts

    mst = minimum_spanning_tree(g)
    mst = mst + mst.T

    n_cc, cc_lbl = connected_components(mst, directed=False)
    if n_cc > 1:                       # split graph → keep largest only
        keep = cc_lbl == int(np.argmax(np.bincount(cc_lbl)))
        pts  = pts[keep]
        mst  = mst[keep][:, keep]
        M    = len(pts)
        if M <= 2:
            return pts

    sp  = shortest_path(mst, directed=False, indices=0, method='D')
    u   = int(np.argmax(np.where(np.isfinite(sp), sp, -1)))
    sp2 = shortest_path(mst, directed=False, indices=u, method='D')
    v   = int(np.argmax(np.where(np.isfinite(sp2), sp2, -1)))
    _, pred = shortest_path(mst, directed=False, indices=u,
                            method='D', return_predecessors=True)
    path: list[int] = []
    cur = v
    for _ in range(M + 2):
        path.append(cur)
        if cur == u:
            break
        cur = pred[cur]
        if cur < 0:
            return pts
    path.reverse()
    return pts[path] if len(path) >= 2 else pts


_SPLIT_TURN_WINDOW = 3   # look this many points before/after i for the chord-vs-chord angle


def _split_at_sharp_turns(ordered: np.ndarray,
                          max_turn_deg: float) -> list[np.ndarray]:
    """Cut an ordered polyline at LOCAL MAXIMA of the windowed turn angle.

    At each interior point i, compute the angle between two short chord
    vectors:
        v_before = pts[i]   - pts[i - W]
        v_after  = pts[i+W] - pts[i]
    Window W = _SPLIT_TURN_WINDOW averages out per-step jitter so noisy
    but globally smooth curves don't trigger; only real kinks do.
    A point becomes a cut iff its windowed angle > max_turn_deg AND it
    is the local max within ±W (non-max suppression). Sub-segments share
    the cut point and must be ≥ 3 points to be kept."""
    n = len(ordered)
    W = _SPLIT_TURN_WINDOW
    if n < 2 * W + 2 or max_turn_deg >= 180.0:
        return [ordered]

    angles = np.zeros(n, dtype=np.float64)
    for i in range(W, n - W):
        va = ordered[i]     - ordered[i - W]
        vb = ordered[i + W] - ordered[i]
        na = float(np.linalg.norm(va))
        nb = float(np.linalg.norm(vb))
        if na < 1e-8 or nb < 1e-8:
            continue
        cos_a = float(np.clip((va @ vb) / (na * nb), -1.0, 1.0))
        angles[i] = float(np.degrees(np.arccos(cos_a)))

    above = angles > max_turn_deg
    if not above.any():
        return [ordered]

    cuts: list[int] = []
    for i in range(W, n - W):
        if not above[i]:
            continue
        lo = max(W, i - W)
        hi = min(n - W, i + W + 1)
        if angles[i] >= angles[lo:hi].max() - 1e-9:
            cuts.append(i)

    if not cuts:
        return [ordered]

    out: list[np.ndarray] = []
    prev = 0
    for c in cuts:
        sub = ordered[prev:c + 1]
        if len(sub) >= 3:
            out.append(sub)
        prev = c
    last = ordered[prev:]
    if len(last) >= 3:
        out.append(last)
    return out if out else [ordered]


def _fit_spline(ordered: np.ndarray, smooth: float, k: int = 3
                ) -> np.ndarray | None:
    """Fit a degree-k B-spline and sample it at a density PROPORTIONAL to
    arc length — 1 point per `_CURVE_POINT_SPACING` vox (default
    1.0) — so a long strand and a short strand get the SAME spacing (no
    longer a fixed 300 pts/curve).  Returns (n,3) or None on failure."""
    if ordered is None or len(ordered) < k + 1:
        return None                       # too few pts (e.g. trimmed CC)
    dists = np.linalg.norm(np.diff(ordered, axis=0), axis=1)
    keep  = np.concatenate([[True], dists > 1e-6])
    pts   = ordered[keep]
    if len(pts) < k + 1:
        return None
    try:
        tck, _ = splprep(pts.T, s=smooth * len(pts), k=k)
        # sample count ∝ arc length (constant point spacing).  arclen ≈ the
        # ordered polyline length (kept segments); B-spline is slightly
        # shorter but this is a good estimate.
        spacing = _CURVE_POINT_SPACING
        arclen  = float(dists[dists > 1e-6].sum())
        n_pts   = int(round(arclen / max(spacing, 1e-6))) + 1
        n_pts   = max(k + 1, min(n_pts, _SPLINE_PTS_MAX))
        xyz = np.ascontiguousarray(
            np.array(splev(np.linspace(0, 1, n_pts), tck)).T,
            dtype=np.float64)
        return xyz
    except Exception:
        return None


def _hide_topo_mst_viz() -> None:
    """Hide the topo-MST overlay + its junction/leaf/diameter SPHERES.  Once
    curves are fitted/loaded that stage is done, so they're off by default
    (re-enable per-structure in the Structures panel, or re-click "Topo MST
    viz").  Safe no-op if a structure isn't present."""
    for nm in ('topo_mst', 'topo_mst_extras'):
        if ps.has_curve_network(nm):
            ps.get_curve_network(nm).set_enabled(False)
    for nm in ('topo_mst_junctions',
               'topo_mst_leaves',
               'topo_mst_diam_a', 'topo_mst_diam_b'):
        if ps.has_point_cloud(nm):
            ps.get_point_cloud(nm).set_enabled(False)


def _register_curves(curve_data: list[tuple[int, np.ndarray]],
                     K: int, rainbow: bool = False) -> None:
    """Register ALL fitted splines as ONE polyscope curve_network
    ('curves'); per-node coloured by _seg_palette so segments stay
    visually distinct without scattering into many networks. Stores
    node→sub / node→local / edges maps so a click still resolves which
    sub-curve + node was hit.

    With `rainbow`, the per-CC palette is swapped for a blue->red ramp along
    each curve's arc length (the Solve Collapse preview colouring); Accept
    re-registers without it, restoring the CC palette."""
    if ps.has_curve_network('curves'):
        ps.remove_curve_network('curves')
    if ps.has_curve_network('curve_sel'):    # stale after re-fit/merge
        ps.remove_curve_network('curve_sel')
    if not curve_data:
        state['curve_node_sub'] = None
        state['curve_node_local'] = None
        state['curve_edges'] = None
        if ps.has_surface_mesh('curve_endcaps'):
            ps.remove_surface_mesh('curve_endcaps')
        return
    # Colour by the curve's CC label (stable across re-fits — a CC
    # keeps its label; a merge only relabels the merged CC), NOT by
    # sub_id (which _fit_curves re-numbers every Detect → all colours
    # would churn on each merge).
    sub2cc = state.get('curve_sub_to_cc') or {}
    ccs = [int(sub2cc.get(int(sid), int(sid))) for sid, _ in curve_data]
    pal = _seg_palette(max(K, (max(ccs) + 1 if ccs else 1), 1))
    nodes_l, edges_l, cols_l, nsub_l, nloc_l = [], [], [], [], []
    off = 0
    for sid, spline in curve_data:
        sp = np.asarray(spline, dtype=np.float32)
        n  = len(sp)
        nodes_l.append(sp)
        if n >= 2:
            e = np.stack([np.arange(n - 1), np.arange(1, n)], axis=1)
            edges_l.append((e + off).astype(np.int32))
        if rainbow:
            # hue 0.70 (blue) at the start -> 0.0 (red) at the end, spaced by
            # ARC LENGTH so the ramp tracks travel distance, not node count.
            _seg = (np.linalg.norm(np.diff(sp.astype(np.float64), axis=0),
                                   axis=1) if n >= 2 else np.zeros(0))
            _al = np.concatenate([[0.0], np.cumsum(_seg)])
            _t = _al / (_al[-1] if _al[-1] > 0 else 1.0)
            cols_l.append(np.array(
                [colorsys.hsv_to_rgb(float((1.0 - x) * 0.70), 1.0, 1.0)
                 for x in _t], dtype=np.float64))
        else:
            cc = int(sub2cc.get(int(sid), int(sid)))
            col = (pal[cc] if 0 <= cc < len(pal)
                   else np.array([0.7, 0.7, 0.7]))
            cols_l.append(np.tile(col, (n, 1)))
        nsub_l.append(np.full(n, sid, dtype=np.int64))
        nloc_l.append(np.arange(n, dtype=np.int64))
        off += n
    nodes = np.concatenate(nodes_l, axis=0)
    edges = (np.concatenate(edges_l, axis=0) if edges_l
             else np.empty((0, 2), dtype=np.int32))
    cn = ps.register_curve_network('curves', nodes, edges)
    cn.set_radius(_CURVE_RADIUS * _SCALE, relative=False)
    cn.add_color_quantity('segment',
                          np.concatenate(cols_l, axis=0).astype(np.float64),
                          enabled=True)
    state['curve_node_sub']   = np.concatenate(nsub_l)
    state['curve_node_local'] = np.concatenate(nloc_l)
    state['curve_edges']      = edges
    # Endpoint cylinders ride along with 'curves' — re-register on every fit.
    _register_curve_endcaps(curve_data, K)
    # curves are the view now → hide the topo-MST overlay + spheres by default
    _hide_topo_mst_viz()


def _tube_mesh(poly: np.ndarray, radius, nseg: int = 14):
    """(verts, faces) of an open tube around polyline `poly` (M,3, M>=2).
    `radius` is a scalar (cylinder) OR a length-M array (per-vertex radius
    → a tapered cone/horn).  A rotation-minimising frame (carry the normal
    forward, projecting out each tangent) keeps the ring from twisting.
    Returns a real triangulated SURFACE — no node spheres — so it renders
    as one clean translucent shape (unlike a curve_network, which draws a
    sphere at every node and piles them up at a big radius)."""
    P = np.asarray(poly, np.float64)
    M = len(P)
    if M < 2:
        return None
    r = (np.full(M, float(radius)) if np.ndim(radius) == 0
         else np.asarray(radius, np.float64))
    T = np.empty_like(P)
    T[1:-1] = P[2:] - P[:-2]
    T[0] = P[1] - P[0]; T[-1] = P[-1] - P[-2]
    T /= (np.linalg.norm(T, axis=1, keepdims=True) + 1e-12)
    a = np.array([0.0, 0.0, 1.0])
    if abs(float(T[0] @ a)) > 0.9:
        a = np.array([0.0, 1.0, 0.0])
    normals = np.empty_like(P)
    n = a - (a @ T[0]) * T[0]; normals[0] = n / (np.linalg.norm(n) + 1e-12)
    for k in range(1, M):
        n = normals[k - 1] - (normals[k - 1] @ T[k]) * T[k]
        nn = np.linalg.norm(n)
        if nn < 1e-8:                       # tangent flipped ~180° — reseed
            a2 = np.array([1.0, 0.0, 0.0])
            if abs(float(T[k] @ a2)) > 0.9:
                a2 = np.array([0.0, 1.0, 0.0])
            n = a2 - (a2 @ T[k]) * T[k]; nn = np.linalg.norm(n)
        normals[k] = n / (nn + 1e-12)
    binorm = np.cross(T, normals)
    ang = np.linspace(0.0, 2.0 * np.pi, nseg, endpoint=False)
    cos = np.cos(ang)[:, None]; sin = np.sin(ang)[:, None]   # (nseg,1)
    verts = np.concatenate(
        [P[k] + r[k] * (cos * normals[k] + sin * binorm[k])
         for k in range(M)], axis=0)                          # (M*nseg,3)
    faces = []
    for k in range(M - 1):
        b0, b1 = k * nseg, (k + 1) * nseg
        for j in range(nseg):
            jn = (j + 1) % nseg
            faces.append([b0 + j, b1 + j, b1 + jn])
            faces.append([b0 + j, b1 + jn, b0 + jn])
    return verts, np.asarray(faces, np.int32)


_ENDCAP_NSEG = 18                          # cone ring resolution (smooth flare)


def _endcap_cone(sp: np.ndarray, side: str):
    """(pl, rvec) for ONE curve end's cone/horn — the SAME geometry the
    'curve_endcaps' mesh is built from, so the rendered flare matches
    the geometry used here exactly.
    `side` ∈ {'start','end'}.  pl is the cone AXIS ordered INNER (deepest
    in the curve) → OUTER (far tip); rvec is the per-vertex radius (base at
    inner, max at outer).  Returns None if the curve is too short.  Driven
    by the same `curve_endcap_*` state knobs as the renderer."""
    sp = np.asarray(sp, np.float64)
    n  = len(sp)
    if n < 2:
        return None
    wrap     = float(state.get('curve_endcap_len', 30.0))
    rad_max  = float(state.get('curve_endcap_radius', 28.0))
    rad_base = float(state.get('curve_endcap_base_radius', 3.0))
    extend   = float(state.get('curve_endcap_extend', 45.0))
    step     = max(rad_base, 3.0)
    ext_d = (np.append(np.arange(step, extend - 1e-6, step), extend)
             if extend > 1e-6 else np.empty(0))
    seg = np.linalg.norm(np.diff(sp, axis=0), axis=1)
    al  = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(al[-1])
    i_start = max(int(np.searchsorted(al, min(wrap, total), 'right')), 2)
    i_end   = min(int(np.searchsorted(al, max(total - wrap, 0.0),
                                      'left')), n - 2)
    if side == 'start':
        cap = sp[0:i_start]
        if len(cap) < 2:
            return None
        tip = sp[0]; out = tip - sp[min(4, i_start - 1)]
        out /= (np.linalg.norm(out) + 1e-12)
        ext = tip[None, :] + ext_d[:, None] * out[None, :]   # near→far
        poly = np.vstack([cap[::-1], ext]) if len(ext) else cap[::-1]
    else:
        cap = sp[i_end:n]
        if len(cap) < 2:
            return None
        tip = sp[-1]; out = tip - sp[max(n - 5, i_end)]
        out /= (np.linalg.norm(out) + 1e-12)
        ext = tip[None, :] + ext_d[:, None] * out[None, :]   # near→far
        poly = np.vstack([cap, ext]) if len(ext) else cap
    # resample to ~step spacing (keep both ends)
    d = np.concatenate([[0.0], np.cumsum(
        np.linalg.norm(np.diff(poly, axis=0), axis=1))])
    keep = [0]
    for t in range(1, len(poly)):
        if d[t] - d[keep[-1]] >= step:
            keep.append(t)
    if keep[-1] != len(poly) - 1:
        keep.append(len(poly) - 1)
    pl = poly[keep]
    dd = np.concatenate([[0.0], np.cumsum(
        np.linalg.norm(np.diff(pl, axis=0), axis=1))])
    rvec = rad_base + (rad_max - rad_base) * (dd / (dd[-1] + 1e-12))
    return pl, rvec




def _register_curve_endcaps(curve_data: list[tuple[int, np.ndarray]],
                            K: int) -> None:
    """At BOTH ends of every fitted curve, draw a semi-transparent CONE in
    that curve's colour: `curve_endcap_base_radius` where it meets the
    curve, flaring to `curve_endcap_radius`, and projecting
    `curve_endcap_extend` vox past the tip along the tangent.

    Built as ONE triangulated surface mesh, NOT a curve_network: the latter
    draws a sphere per node, which at this radius piles up into opaque
    balls.  The global transparency mode is flipped to 'pretty' so alpha
    takes effect.  Auto-re-registered by `_register_curves`."""
    if ps.has_surface_mesh('curve_endcaps'):
        ps.remove_surface_mesh('curve_endcaps')
    if not curve_data or not state.get('curve_endcap_show', False):
        return
    # polyscope ignores per-structure transparency unless the global mode
    # allows it — default is 'none' (→ everything opaque).  Enable once.
    try:
        ps.set_transparency_mode('pretty')
    except Exception:
        pass
    sub2cc = state.get('curve_sub_to_cc') or {}
    ccs = [int(sub2cc.get(int(sid), int(sid))) for sid, _ in curve_data]
    pal = _seg_palette(max(K, (max(ccs) + 1 if ccs else 1), 1))
    V_l, F_l, C_l = [], [], []
    voff = 0
    for sid, spline in curve_data:
        cc  = int(sub2cc.get(int(sid), int(sid)))
        col = (pal[cc] if 0 <= cc < len(pal)
               else np.array([0.7, 0.7, 0.7]))
        for side in ('start', 'end'):
            cone = _endcap_cone(np.asarray(spline, np.float64), side)
            if cone is None:
                continue
            pl, rvec = cone
            tm = _tube_mesh(pl, rvec, _ENDCAP_NSEG)
            if tm is None:
                continue
            v, f = tm
            V_l.append(v); F_l.append(f + voff)
            C_l.append(np.tile(col, (len(v), 1)))
            voff += len(v)
    if not V_l:
        return
    m = ps.register_surface_mesh('curve_endcaps',
                                 np.concatenate(V_l, axis=0),
                                 np.concatenate(F_l, axis=0))
    m.add_color_quantity('curve colour',
                         np.concatenate(C_l, axis=0).astype(np.float64),
                         defined_on='vertices', enabled=True)
    m.set_transparency(float(state.get('curve_endcap_alpha', 0.45)))
    try:                                   # show the inner wall too
        m.set_back_face_policy('identical')
    except Exception:
        pass
    try:
        m.set_enabled(bool(state.get('curve_endcap_show', False)))
    except Exception:
        pass




















def _polys_to_network(name: str, polys: list, color, radius_mul: float):
    """Register `polys` (list of point-lists) as a polyline curve_network."""
    nodes, edges = [], []
    for poly in polys:
        i0 = len(nodes)
        for p in poly:
            nodes.append(np.asarray(p, np.float32))
        for t in range(len(poly) - 1):
            edges.append([i0 + t, i0 + t + 1])
    if ps.has_curve_network(name):
        ps.remove_curve_network(name)
    if nodes:
        cn = ps.register_curve_network(name, np.asarray(nodes, np.float32),
                                       np.asarray(edges, np.int32))
        cn.set_color(color)
        cn.set_radius(_ARM_BREAK_RADIUS * radius_mul, relative=False)
        cn.set_enabled(True)










def _al(P) -> float:
    return (float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())
            if len(P) > 1 else 0.0)

def _resample_arclen(P, n):
    """Resample polyline P to exactly n points uniformly spaced by ARC
    LENGTH (linear interp).  Endpoints & shape preserved; does NOT
    re-smooth (unlike _fit_spline) — purely fixes point DENSITY so a
    join's vstack doesn't pile up into an ultra-dense curve that grows
    every merge, at a fitted curve's sample density."""
    P = np.asarray(P, float)
    if len(P) < 2 or int(n) < 2:
        return P
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    tot = float(s[-1])
    if tot <= 0.0:
        return P
    u = np.linspace(0.0, tot, int(n))
    return np.column_stack([np.interp(u, s, P[:, k]) for k in range(3)])

def _resample_dense(P):
    """Resample P to a point count STRICTLY PROPORTIONAL to its arc
    length: n = arclen / curve_radius (≈ one point per curve_radius
    vox).  So density is constant — a long merged curve auto-gets more
    points (no jaggies) and a short one fewer (not over-dense); it
    rises with length, no fixed count, no cap.  No new knob: spacing
    is the existing `curve_radius` slider (raise it = sparser)."""
    sp = max(float(state.get('curve_radius', 4.0)), 1e-3)
    n = max(2, int(round(_al(P) / sp)))
    return _resample_arclen(P, n)

def _trim_from_end(P, back):
    """Drop the tail of P whose arc-length-from-P[-1] is < `back` (so
    the curve ends `back` voxels SHORT of its old end).  Capped at 45%
    of P; ≥4 points kept."""
    P = np.asarray(P, float)
    if len(P) < 6 or back <= 0:
        return P
    s = np.concatenate([[0.0],
                        np.cumsum(np.linalg.norm(np.diff(P, axis=0),
                                                 axis=1))])
    tot = s[-1]
    back = min(back, 0.45 * tot)
    idx = max(int(np.searchsorted(s, tot - back)), 4)
    return P[:idx]

def _trim_from_start(P, back):
    return _trim_from_end(np.asarray(P, float)[::-1], back)[::-1]

def _bridge_through(tail3, waypt, head3, n=60):
    """Tangent-continuous interpolating cubic bridging the tail tip
    (`tail3[-1]`) THROUGH `waypt` to the head tip (`head3[0]`).  `tail3` /
    `head3` are the ~3 boundary points on each side (for C1 continuity at the
    seams).  Returns the `n`-sample arc between the two tips; falls back to a
    3-point polyline if splprep fails.  Shared bridge builder for
    `_endpoint_join` / `_self_loop_join`."""
    tail3 = np.asarray(tail3, float)
    head3 = np.asarray(head3, float)
    waypt = np.asarray(waypt, float)
    ctrl = np.vstack([tail3, waypt[None, :], head3])
    i_tail = len(tail3) - 1                 # ctrl index of the tail tip
    i_head = len(tail3) + 1                 # ctrl index of the head tip (waypt between)
    try:
        tck, u = splprep(ctrl.T, s=0.0, k=min(3, len(ctrl) - 1))
        return np.array(splev(np.linspace(u[i_tail], u[i_head], n), tck)).T
    except Exception:
        return np.vstack([tail3[-1], waypt, head3[0]])

def _endpoint_join(spS, e, spT, q, smooth, via=None):
    """Two TRUE endpoints (the cyan-ball case) → PURE end-to-end join.
    BOTH curves kept WHOLE: orient seg to END at e, tgt to START at q,
    retreat both ≈gap, bridge via an interpolating cubic through a waypoint
    (tangent-continuous).  The waypoint is the gap midpoint by default, OR an
    explicit `via` point (e.g. an X-crossing midpoint, so the merged yarn
    PASSES THROUGH it and the over/under choice changes the geometry).  NO
    argmin/cut — cyan picks are endpoints BY CONSTRUCTION."""
    spS = np.asarray(spS, float).copy()
    spT = np.asarray(spT, float).copy()
    e = np.asarray(e, float)
    q = np.asarray(q, float)
    if len(spS) < 6 or len(spT) < 6:
        return None, 'curve too short'
    if np.linalg.norm(spS[0] - e) < np.linalg.norm(spS[-1] - e):
        spS = spS[::-1]                       # seg ENDS at e
    if np.linalg.norm(spT[0] - q) > np.linalg.norm(spT[-1] - q):
        spT = spT[::-1]                       # tgt STARTS at q
    gap = float(np.linalg.norm(e - q))
    spS_t = _trim_from_end(spS, gap)
    spT_t = _trim_from_start(spT, gap)
    if len(spS_t) < 3 or len(spT_t) < 3:
        return None, 'curve too short after retreat'
    waypt = (np.asarray(via, float) if via is not None
             else 0.5 * (e + q))              # bridge passes through this
    bridge = _bridge_through(spS_t[-3:], waypt, spT_t[:3])
    full = np.vstack([spS_t[:-1], bridge, spT_t[1:]])
    if len(full) < 2:
        return None, 'join failed'
    full = _resample_dense(full)   # density ∝ arc length
    return full, {'e': e, 'q': q, 'mid': waypt, 'gap': gap,
                  'kept_len': float(_al(spS) + _al(spT)),
                  'drop_len': 0.0}

def _self_loop_join(sp, smooth, via=None):
    """CLOSE one curve into a loop: bridge its tail (sp[-1]) back to its head
    (sp[0]) through `via` (or the gap midpoint), retreating both ends ≈gap so
    the join is smooth.  Used when a delete-region pair maps to BOTH ends of
    the SAME curve — we allow it (loop), pruning real loops is a later step.
    Returns (loop_pts, geom)."""
    sp = np.asarray(sp, float).copy()
    if len(sp) < 6:
        return None, 'curve too short'
    e, q = sp[-1], sp[0]
    gap = float(np.linalg.norm(e - q))
    body = _trim_from_start(_trim_from_end(sp, gap), gap)   # retreat both ends
    if len(body) < 4:
        return None, 'curve too short after retreat'
    waypt = np.asarray(via, float) if via is not None else 0.5 * (e + q)
    bridge = _bridge_through(body[-3:], waypt, body[:3])
    loop = np.vstack([body, bridge[1:]])    # append closing arc → ends ≈ head
    loop = _resample_dense(loop)
    loop[-1] = loop[0]                       # close EXACTLY (start == end)
    return loop, {'gap': gap, 'mid': waypt, 'loop': True,
                  'kept_len': float(_al(sp)), 'drop_len': 0.0}

def _box_mesh(lo: np.ndarray, hi: np.ndarray):
    """8 corners + 12 triangles (6 quad faces) of the AABB [lo, hi]."""
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    v = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]],
                 np.float64)
    f = np.array([[0, 2, 1], [0, 3, 2],   # z0 (bottom)
                  [4, 5, 6], [4, 6, 7],   # z1 (top)
                  [0, 1, 5], [0, 5, 4],   # y0
                  [3, 6, 2], [3, 7, 6],   # y1
                  [0, 7, 3], [0, 4, 7],   # x0
                  [1, 2, 6], [1, 6, 5]],  # x1
                 np.int32)
    return v, f

def _register_delete_aabbs() -> None:
    """For each D-delete record (`topo_del_arm_breaks`), draw a TRANSLUCENT
    axis-aligned bounding box around its 4 arm break-points (+ its two
    junctions, when their coords are available) as a clickable surface mesh
    `delete_aabb_<i>`.  Clicking the box (handled in `callback`) picks that
    delete's 4 break points so they can be connected together — the 4-way
    connection algorithm is TBD; for now the click just reports + highlights
    them.  Margin/alpha/visibility are state-driven.  Shown by default."""
    recs = state.get('topo_del_arm_breaks') or []
    # clear ALL existing boxes first (indices may be non-contiguous once some
    # regions are connected/skipped, so don't stop at the first gap)
    for i in range(max(len(recs), 0) + 1):
        if ps.has_surface_mesh(f'delete_aabb_{i}'):
            ps.remove_surface_mesh(f'delete_aabb_{i}')
    if not recs or not state.get('delete_aabb_show', True):
        return
    margin = float(state.get('delete_aabb_margin', 12.0))
    alpha  = float(state.get('delete_aabb_alpha', 0.18))
    pts = state.get('pts')
    pts = np.asarray(pts) if pts is not None else None
    nreal = 0 if pts is None else len(pts)
    pal = _seg_palette(max(len(recs), 1))
    try:
        ps.set_transparency_mode('pretty')   # so alpha actually shows
    except Exception:
        pass
    done = state.get('delete_conn_done') or set()
    shown = 0
    for di, rec in enumerate(recs):
        if di in done:                      # already connected → no box
            continue
        P = []
        for a in rec.get('arms', []):
            P.extend(a.get('break_pos') or [])
        for jk in ('J', 'J2'):              # add junction coords if real
            g = int(rec.get(jk, -1))
            if 0 <= g < nreal:
                P.append([float(c) for c in pts[g]])
        if len(P) < 2:
            continue
        P = np.asarray(P, np.float64)
        v, f = _box_mesh(P.min(0) - margin, P.max(0) + margin)
        m = ps.register_surface_mesh(f'delete_aabb_{di}', v, f)
        m.set_color(tuple(float(c) for c in pal[di % len(pal)]))
        m.set_transparency(alpha)
        try:
            m.set_back_face_policy('identical')
        except Exception:
            pass
        shown += 1
    if shown:
        print(f'  [delete-aabb] {shown} delete-region box(es) — click one to '
              f'pick its arm break points for a 4-way connection (TBD)')

def _compute_delete_conn_ways(rec: dict) -> list:
    """The candidate ways to connect a delete's arm break points.  The 4 arms
    split by side (J0 / J1); a valid yarn join pairs each J0-side arm with a
    J1-side arm (OPPOSITE sides) — same-side arms are NEVER joined (that would
    U-turn one yarn, the excluded matching).  For the normal 2-vs-2 case that
    yields exactly TWO perfect matchings; one usually CROSSES (an X) and one
    runs quasi-parallel.  Each matching is ONE way — the front/rear
    over-under ambiguity of an X is NOT enumerated, since it leaves the
    connectivity unchanged.

    Returns a list of way-dicts:
      {'pairs': [(ptA, ptB, armA_idx, armB_idx), ...],
       'cross': None | (c1, c2, segi, segj),   # crossing pts on each bar
       'over':  None | 0}                       # 0 = X routed via midpoint
    """
    import itertools

    def rep(a):
        bp = a.get('break_pos') or []
        return np.asarray(bp[0], np.float64) if bp else None
    A = [(int(a.get('arm', -1)), rep(a)) for a in rec.get('arms', [])
         if a.get('side') == 'J0' and rep(a) is not None]
    B = [(int(a.get('arm', -1)), rep(a)) for a in rec.get('arms', [])
         if a.get('side') == 'J1' and rep(a) is not None]
    if not A or not B:
        return []
    small, large = (A, B) if len(A) <= len(B) else (B, A)
    matchings = []
    for perm in itertools.permutations(range(len(large)), len(small)):
        pairs = []
        for i in range(len(small)):
            (ai, pa), (bi, pb) = small[i], large[perm[i]]
            pairs.append((pa, pb, ai, bi))
        matchings.append(pairs)
        if len(matchings) >= 8:             # safety cap (deletes are 2v2 → 2)
            break
    # X detection uses the AABB's 3 AXIS-ALIGNED planes (no fragile PCA): a
    # matching is an X if its two bars cross in ANY of them.  `_AXIS_PLANES`
    # = (dropped-axis, plane-name): drop X→YZ view, drop Y→XZ, drop Z→XY.
    _AXIS_PLANES = ((0, 'YZ'), (1, 'XZ'), (2, 'XY'))
    ways = []
    for pairs in matchings:
        segs = [(np.asarray(pa, float), np.asarray(pb, float))
                for (pa, pb, _a, _b) in pairs]
        cross = None
        planes = []
        if len(segs) >= 2:
            ci = cj = None
            for ax, nm in _AXIS_PLANES:
                kp = [c for c in range(3) if c != ax]   # the 2 kept coords

                def _pj(p):
                    return np.array([p[kp[0]], p[kp[1]]])
                hit = False
                for i in range(len(segs)):
                    for j in range(i + 1, len(segs)):
                        if _seg2d_cross(_pj(segs[i][0]), _pj(segs[i][1]),
                                        _pj(segs[j][0]), _pj(segs[j][1])):
                            hit = True; ci, cj = i, j
                if hit:
                    planes.append(nm)
            if planes:
                _s, _t, c1, c2 = _seg_seg_closest(
                    segs[ci][0], segs[ci][1], segs[cj][0], segs[cj][1])
                cross = (c1, c2, ci, cj)
        # ONE way per matching — the paper's two cases per collapse
        # ({(a,c),(b,d)} vs {(a,d),(b,c)}).  The X matching used to be
        # expanded into a second way here (the over/under switch), but that
        # only changes WHICH strand passes in front, never which arms are
        # joined, so it duplicated every loop-free solution with identical
        # connectivity.  `over` stays 0 so _way_via still routes an X pair
        # through the crossing midpoint.
        ways.append({'pairs': pairs, 'cross': cross,
                     'over': None if cross is None else 0,
                     'planes': planes})
    return ways

def _seg_seg_closest(p1, q1, p2, q2):
    """Closest points between 3D segments p1q1 and p2q2 (Ericson RTCD).
    Returns (s, t, c1, c2): params s,t∈[0,1] and the closest points
    c1=p1+s(q1-p1), c2=p2+t(q2-p2)."""
    p1 = np.asarray(p1, float); q1 = np.asarray(q1, float)
    p2 = np.asarray(p2, float); q2 = np.asarray(q2, float)
    d1 = q1 - p1; d2 = q2 - p2; r = p1 - p2
    a = float(d1 @ d1); e = float(d2 @ d2); f = float(d2 @ r)
    eps = 1e-12
    if a < eps and e < eps:
        return 0.0, 0.0, p1, p2
    if a < eps:
        s = 0.0; t = min(max(f / e, 0.0), 1.0)
    else:
        c = float(d1 @ r)
        if e < eps:
            t = 0.0; s = min(max(-c / a, 0.0), 1.0)
        else:
            b = float(d1 @ d2); denom = a * e - b * b
            s = (min(max((b * f - c * e) / denom, 0.0), 1.0)
                 if denom > eps else 0.0)
            t = (b * s + f) / e
            if t < 0.0:
                t = 0.0; s = min(max(-c / a, 0.0), 1.0)
            elif t > 1.0:
                t = 1.0; s = min(max((b - c) / a, 0.0), 1.0)
    return s, t, p1 + d1 * s, p2 + d2 * t

def _seg2d_cross(a, b, c, d) -> bool:
    """True iff 2D segments ab and cd PROPERLY cross (interior intersection,
    via the orientation/CCW test).  Collinear / endpoint touches → False."""
    def o(p, q, r):
        return ((q[0] - p[0]) * (r[1] - p[1])
                - (q[1] - p[1]) * (r[0] - p[0]))
    return (o(a, b, c) * o(a, b, d) < 0.0
            and o(c, d, a) * o(c, d, b) < 0.0)

def _join_two_curve_endpoints(ptA, ptB, smooth=None, via=None):
    """Smooth-MERGE the two curves whose ENDPOINTS are nearest ptA / ptB —
    the SAME `_endpoint_join` + _curve_data integration, callable
    programmatically.  `via`
    (optional) forces the bridge THROUGH that point (the X-crossing midpoint).
    Mutates `_curve_data` (removes the 2 source curves, adds the merged one).
    Returns (new_sub | None, message)."""
    global _curve_data
    s = state
    if smooth is None:
        smooth = _CURVE_SMOOTH
    ptA = np.asarray(ptA, float); ptB = np.asarray(ptB, float)

    def _nearest_end(pt):
        best, bd = None, 1e18
        for sid, sp in _curve_data:
            sp = np.asarray(sp, float)
            if len(sp) < 2:
                continue
            for end in (sp[0], sp[-1]):
                d = float(np.linalg.norm(end - pt))
                if d < bd:
                    bd, best = d, (int(sid), end.copy())
        return best, bd
    (resA, dA) = _nearest_end(ptA)
    (resB, dB) = _nearest_end(ptB)
    if resA is None or resB is None:
        return None, 'no curves'
    sA, eA = resA; sB, eB = resB
    cd_map = {int(sid): np.asarray(sp, float) for sid, sp in _curve_data}
    if sA == sB:
        # both ends on the SAME curve → allow it: CLOSE that curve into a
        # loop (real-loop detection/pruning is a later step).
        out, geom = _self_loop_join(cd_map[sA], smooth, via=via)
        inv = [sA]
    else:
        out, geom = _endpoint_join(cd_map[sA], eA, cd_map[sB], eB, smooth,
                                   via=via)
        inv = [sA, sB]
    if out is None:
        return None, str(geom)
    # integrate: merge CCs, replace subs
    sub2cc = dict(s.get('curve_sub_to_cc') or {})
    ccs = sorted({int(sub2cc.get(sb, -1)) for sb in inv
                  if int(sub2cc.get(sb, -1)) >= 0})
    lab = s.get('curve_seg_labels')
    if lab is not None and len(ccs) >= 1:
        rep = ccs[0]; lab = np.asarray(lab).copy()
        for c in ccs[1:]:
            lab[lab == c] = rep
        s['curve_seg_labels'] = lab
    else:
        rep = ccs[0] if ccs else -1
    new_sub = max([sid for sid, _ in _curve_data], default=-1) + 1
    keep = [(sid, sp) for sid, sp in _curve_data if sid not in inv]
    keep.append((new_sub, np.asarray(out, np.float64)))
    _curve_data = keep
    n2 = {sid: sub2cc.get(sid) for sid, _ in keep if sid != new_sub}
    n2[new_sub] = rep if rep >= 0 else sub2cc.get(inv[0])
    s['curve_sub_to_cc'] = n2
    # track loops TOPOLOGICALLY: a loop is created EXACTLY when a curve is
    # connected to itself (sA==sB) → no distance threshold needed.
    loop_subs = s.setdefault('loop_subs', set())
    loop_subs.difference_update(inv)           # consumed subs are gone
    if sA == sB:
        loop_subs.add(new_sub)
    if sA == sB:
        return new_sub, (f'sub{sA} CLOSED into loop →sub{new_sub} '
                         f'(gap {geom.get("gap", 0):.0f}vox; '
                         f'map dist {dA:.1f}/{dB:.1f})')
    return new_sub, (f'sub{sA}↔sub{sB}→sub{new_sub} (gap {geom.get("gap", 0):.0f}'
                     f'vox; map dist {dA:.1f}/{dB:.1f})')

def _way_via(way, idx):
    """The via-point (X crossing midpoint) for pair `idx` of `way`, honouring
    the over/under switch; None for a parallel matching."""
    cross = way.get('cross')
    if cross is None:
        return None
    si, sj = int(cross[2]), int(cross[3])
    over = int(way.get('over') or 0)
    Mi = np.asarray(cross[0], float); Mj = np.asarray(cross[1], float)
    if idx == si:
        return Mi if over == 0 else Mj
    if idx == sj:
        return Mj if over == 0 else Mi
    return None

def _apply_way(way):
    """Connect ONE way's break-point pairs (headless smooth-merge, routing X
    pairs through their crossing midpoint).  Mutates `_curve_data`.  Returns a
    list of (armA, armB, new_sub|None, msg) per pair — shared by the
    interactive confirm and the possibility enumeration."""
    res = []
    for idx, (pa, pb, ai, bi) in enumerate(way['pairs']):
        ns, msg = _join_two_curve_endpoints(pa, pb, via=_way_via(way, idx))
        res.append((int(ai), int(bi), ns, msg))
    return res

def _combo_has_loops() -> bool:
    """Quiet (no-viz) loop test for the possibility enumeration: True iff any
    current curve is a tracked self-loop OR closes on itself exactly."""
    if set(state.get('loop_subs') or set()) & {int(s) for s, _ in _curve_data}:
        return True
    for _sid, sp in _curve_data:
        sp = np.asarray(sp, float)
        if len(sp) >= 4 and float(np.linalg.norm(sp[0] - sp[-1])) <= 1e-2:
            return True
    return False

# ── Knot determinant (transplanted verbatim from v6_view_yarn.py) ──
# Ranks single-strand solutions: the SMALLEST log10|det| is the
# least-knotted weave, which is the one a real yarn most likely takes.
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

def _knot_rank_solutions(kept):
    """(det, log10|det|, n_crossings) per entry of `kept`, or None when that
    solution is NOT a single strand -- the invariant closes the polyline, so it
    only means something for one curve.  Returns (dets, best_index | None)."""
    dets = []
    for _combo, curves in kept:
        if len(curves) != 1:
            dets.append(None)
            continue
        try:
            dets.append(knot_determinant(np.asarray(curves[0][1], float)))
        except Exception as _ex:
            print(f'  [collapse] knot determinant failed: {_ex}')
            dets.append(None)
    ok = [(d[1], i) for i, d in enumerate(dets) if d is not None]
    return dets, (min(ok)[1] if ok else None)


# ── Collapse solving: enumerate the delete regions' matchings, keep only the
#    loop-free results, and let the user step through them and commit one. ──
def _solve_collapse() -> None:
    """Enumerate every combination of delete-region matchings, apply each on a
    fresh copy of the current curves, and KEEP only the loop-free ones
    (`_combo_has_loops`).  Results are held in `state['collapse_solutions']`
    for the </> + Accept buttons; nothing is written to disk."""
    global _curve_data
    s = state
    recs = s.get('topo_del_arm_breaks') or []
    if not recs:
        print('  [collapse] no delete regions — D-delete a >-----< first'); return
    region_ways = [(di, _compute_delete_conn_ways(rec))
                   for di, rec in enumerate(recs)]
    region_ways = [(di, ws) for di, ws in region_ways if ws]
    if not region_ways:
        print('  [collapse] no connectable regions'); return
    import itertools
    n_combo = 1
    for _di, ws in region_ways:
        n_combo *= len(ws)
    print(f'  [collapse] {len(region_ways)} region(s), ways '
          f'{[len(ws) for _d, ws in region_ways]} -> {n_combo} combinations; '
          f'keeping the loop-free ones...')
    if n_combo > 20000:
        print(f'  [collapse] {n_combo} combos too many - aborting'); return
    base_cd = [(int(sid), np.asarray(sp, float).copy())
               for sid, sp in _curve_data]
    base_sub2cc = dict(s.get('curve_sub_to_cc') or {})
    base_lab = (None if s.get('curve_seg_labels') is None
                else np.asarray(s['curve_seg_labels']).copy())
    kept, n_loop = [], 0
    for combo in itertools.product(*[range(len(ws)) for _d, ws in region_ways]):
        _curve_data = [(sid, sp.copy()) for sid, sp in base_cd]
        s['curve_sub_to_cc'] = dict(base_sub2cc)
        s['curve_seg_labels'] = (None if base_lab is None else base_lab.copy())
        s['loop_subs'] = set()
        for (_di, ws), wi in zip(region_ways, combo):
            _apply_way(ws[wi])
        if _combo_has_loops():
            n_loop += 1
        else:
            kept.append((tuple(int(x) for x in combo),
                         [(int(sid), np.asarray(sp, np.float32))
                          for sid, sp in _curve_data]))
    _curve_data = [(sid, sp) for sid, sp in base_cd]
    s['curve_sub_to_cc'] = dict(base_sub2cc)
    s['curve_seg_labels'] = base_lab
    s['loop_subs'] = set()
    print(f'  [collapse] {len(kept)}/{n_combo} loop-free, {n_loop} had loops')
    if not kept:
        print('  [collapse] every combination closed a loop - nothing to pick')
        s['collapse_solutions'] = None
        s['collapse_shown'] = 0
        _register_curves(_curve_data,
                         max([sid for sid, _ in _curve_data], default=-1) + 1)
        return
    ncs = sorted({len(cs) for _cb, cs in kept})
    print(f'  [collapse] curve-counts across solutions: {ncs}  '
          f'(1 = a single strand)')
    s['collapse_base'] = base_cd
    s['collapse_solutions'] = kept
    # Rank the single-strand solutions by knot complexity and SHOW the
    # least-knotted one: loop-freeness alone can leave several options
    # (a region whose two matchings are topologically equivalent), and
    # among those the simplest knot is the one a real yarn takes.
    _dets, _best = _knot_rank_solutions(kept)
    s['collapse_dets'] = _dets
    if len(kept) > 1 and _best is not None:
        _order = sorted((d[1], i) for i, d in enumerate(_dets)
                        if d is not None)
        print(f'  [collapse] {len(_order)} single-strand solution(s) by '
              f'knot complexity (log10|det|, smallest = least knotted):')
        for _lg, _i in _order:
            _dv, _l10, _ncr = _dets[_i]
            _ds = str(_dv) if _dv is not None else f'~10^{_l10:.1f}'
            print(f'      solution {_i + 1}: matching {list(kept[_i][0])}'
                  f'  {_ncr} crossings  det={_ds}  log10={_l10:.2f}'
                  + ('   <= AUTO-SELECTED' if _i == _best else ''))
        s['collapse_shown'] = int(_best)
        _collapse_show(int(_best))
    else:
        s['collapse_shown'] = 0
        _collapse_show(0)


def _collapse_show(k: int) -> None:
    """Display loop-free solution `k` (wraps around) on the curve viz."""
    global _curve_data
    s = state
    sols = s.get('collapse_solutions') or []
    if not sols:
        print('  [collapse] no solutions - press "Solve Collapse" first'); return
    k = int(k) % len(sols)
    s['collapse_shown'] = k
    combo, curves = sols[k]
    _curve_data = [(int(sid), np.asarray(sp, float).copy())
                   for sid, sp in curves]
    Kpal = max([sid for sid, _ in _curve_data], default=-1) + 1
    _register_curves(_curve_data, max(Kpal, 1), rainbow=True)
    tot = sum(float(np.linalg.norm(np.diff(np.asarray(sp, float), axis=0),
                                   axis=1).sum()) for _sid, sp in _curve_data)
    print(f'  [collapse] solution {k + 1}/{len(sols)}: {len(_curve_data)} '
          f'curve(s), total {tot:.0f} vox, matching per region = {list(combo)}')


def _hide_delete_region_viz() -> None:
    """Switch off the D-delete overlays: the translucent `delete_aabb_<i>`
    boxes and the `dbg_arm_breaks` balls.  They are only disabled, not
    removed, so a later delete can register its own again."""
    recs = state.get('topo_del_arm_breaks') or []
    n = 0
    # indices can be non-contiguous, so sweep the whole range like
    # _register_delete_aabbs does rather than stopping at the first gap
    for i in range(len(recs) + 1):
        nm = f'delete_aabb_{i}'
        if ps.has_surface_mesh(nm):
            ps.get_surface_mesh(nm).set_enabled(False)
            n += 1
    if ps.has_point_cloud('dbg_arm_breaks'):
        ps.get_point_cloud('dbg_arm_breaks').set_enabled(False)
        n += 1
    return n


def _collapse_accept() -> None:
    """Commit the solution on screen: it becomes the live curves and the
    candidate list is dropped."""
    s = state
    sols = s.get('collapse_solutions') or []
    if not sols:
        print('  [collapse] nothing to accept'); return
    k = int(s.get('collapse_shown', 0)) % len(sols)
    combo, _curves = sols[k]
    s['collapse_solutions'] = None
    s['collapse_base'] = None
    s['collapse_dets'] = None
    s['collapse_shown'] = 0
    # back to the per-CC palette the rest of the session uses
    _register_curves(_curve_data,
                     max([sid for sid, _ in _curve_data], default=-1) + 1)
    n = len(_curve_data)
    print(f'  [collapse] ACCEPTED solution {k + 1}: matching per region = '
          f'{list(combo)}; {n} curve(s) kept'
          + ('  -> single strand' if n == 1 else ''))
    # the delete regions are resolved now, so their overlays only clutter the view
    _hidden = _hide_delete_region_viz()
    if _hidden:
        print(f'  [collapse] hid {_hidden} delete-region overlay(s)')


def _dbg_show_arm_breaks(recs=None) -> None:
    """DEBUG: scatter every recorded D-delete arm break-point at its stored
    coordinate (`break_pos`) as `dbg_arm_breaks` — RED = the break is on a
    search/loop synthetic node (gidx≥N), GREEN = a real node.  Bigger than
    the curve-endpoint balls so you can eyeball how far each break sits from
    the nearest curve end (epm_breaks)."""
    s = state
    breaks = recs if recs is not None else (s.get('topo_del_arm_breaks') or [])
    if ps.has_point_cloud('dbg_arm_breaks'):
        ps.remove_point_cloud('dbg_arm_breaks')
    if not breaks:
        print('  [dbg-arm] no arm-break records'); return
    N = int(len(s['pts']))
    pos, col, meta = [], [], []
    for rec in breaks:
        for a in rec.get('arms', []):
            is_syn = any(int(g) >= N for g in a.get('break_gidx', []))
            for p in (a.get('break_pos') or []):
                pos.append(p)
                col.append((1.0, 0.1, 0.1) if is_syn else (0.1, 1.0, 0.3))
                meta.append((a['arm'], a['side'], int(a['junction']), is_syn))
    if not pos:
        print('  [dbg-arm] records carry no break_pos (old pkl)'); return
    pc = ps.register_point_cloud('dbg_arm_breaks',
                                 np.asarray(pos, np.float32))
    pc.set_radius(_ARM_BREAK_RADIUS, relative=False)   # fixed vox
    pc.add_color_quantity('synthetic=red / real=green',
                          np.asarray(col, np.float32), enabled=True)
    try:
        pc.set_enabled(True)
    except Exception:
        pass
    n_syn = sum(1 for m in meta if m[3])
    print(f'  [dbg-arm] {len(pos)} arm break point(s) '
          f'(red={n_syn} on synthetic bridge pts, '
          f'green={len(pos) - n_syn} real):')
    for (a, side, j, syn), p in zip(meta, np.asarray(pos)):
        print(f'      arm[{a}] {side} J=gidx{j} @ {np.round(p, 1)}'
              f'{"  <- SYNTHETIC (bridge pt)" if syn else ""}')

def _topo_pairing_delete() -> None:
    """'D' on a shown >-----<: DELETE the highlighted local tangle instead
    of re-pairing it.  Clears every MST edge touching the 4 CAPPED arm
    subtrees + the shared trunk + both junctions (payload['viz_gidx']) from
    `topo_mst_global_adj` IN PLACE, then refreshes from that cached adj (so
    earlier stitches survive).  state['pts'] is never touched
    (same mechanism as the short-arm CUT); the deleted nodes drop to
    degree 0 and vanish from topo_mst, and the yarn BEYOND the capped arms
    survives as separate fragments.  Undoable via 'Undo stitch'."""
    s = state
    payload = s.get('topo_pairing_payload')
    if (payload is None
            or not ps.has_point_cloud('topo_branch_pairing_0')):
        print('  [pairing] no active >-----< to delete (click a '
              '>-----< junction first)')
        return
    adj = s.get('topo_mst_global_adj')
    if adj is None:
        print('  [pairing] no MST adj — run "Topo MST viz" first')
        return
    # Prefer the shorter delete set (arms capped at _TOPO_DELETE_ARM_HOPS
    # + full trunk + junctions); fall back to the full highlight set.
    viz = payload.get('del_gidx')
    if viz is None or len(viz) == 0:
        viz = payload.get('viz_gidx')
    if viz is None or len(viz) == 0:
        print('  [pairing] payload has no delete node set — abort')
        return
    _topo_stitch_snapshot(
        f"delete >-----< @ J{payload.get('J')}↔J{payload.get('J2')} "
        f"({len(viz)} nodes)")
    rm = set(int(g) for g in viz)
    # Disconnect `rm` in the LIVE cached adj (in-place, exactly like
    # arm-cut / loop / accept / click-delete) so EARLIER adj-only edits —
    # Straight/Sketch Connect bridges above all — survive.  A from-scratch
    # rebuild used to run here instead, which dropped topo_mst_extra_pts and
    # proximity-reconnected everything, silently undoing every prior stitch.
    # rm still goes into topo_mst_deleted so the NEXT full rebuild keeps them
    # buried; state['pts'] is never touched and the snapshot above undoes it.
    adj_live = s.get('topo_mst_global_adj') or {}
    n_cut = 0
    for _g in rm:
        _ig = int(_g)
        if _ig not in adj_live:
            continue
        for _nb in list(adj_live[_ig]):
            _inb = int(_nb)
            if _inb in adj_live:
                adj_live[_inb].discard(_ig)
            n_cut += 1
        adj_live[_ig] = set()
    _del = s.setdefault('topo_mst_deleted', set())
    _del |= rm
    print(f'  [pairing] DELETED highlighted >-----<: {len(rm)} node(s) '
          f'(4 capped arms + trunk + both junctions); cleared ~{n_cut} adj '
          f'entries, total deleted={len(_del)}.  Refreshing from cached adj '
          f'— prior edits preserved.  Undo via "Undo stitch".')
    # ── Record the 4 arm break-points for downstream CURVE linking.
    # Arms off the SAME junction are the SAME SIDE of the crossing, so
    # they must NOT be reconnected to each other (that would U-turn one
    # yarn back on itself); only opposite-side arms may be joined.
    _breaks = payload.get('arm_breaks') or []
    if _breaks:
        by_side: dict = {}
        for b in _breaks:
            by_side.setdefault(b['side'], []).append(b)
        print(f'  [arm-breaks] the 4 arm break points of this >-----< '
              f'(J0=gidx{payload.get("J")}  J1=gidx{payload.get("J2")}):')
        for b in _breaks:
            bg = b['break_gidx']
            bgs = ('gidx' + str(bg[0]) if len(bg) == 1 else
                   (', '.join('gidx' + str(x) for x in bg) if bg else
                    '(arm shorter than the delete depth — nothing left)'))
            print(f'      arm[{b["arm"]}]  side={b["side"]} '
                  f'(off junction gidx{b["junction"]})  breaks = {bgs}  '
                  f'(arm length {b["arm_len"]} pts)')
        no_connect = []
        for side, lst in by_side.items():
            ids = [x['arm'] for x in lst]
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    no_connect.append((ids[i], ids[j]))
                    print(f'      ⚠ same side, not joinable: '
                          f'arm[{ids[i]}] & arm[{ids[j]}] both on {side} '
                          f'(gidx{lst[0]["junction"]}) — never join these two')
        rec = {'J': int(payload.get('J')), 'J2': int(payload.get('J2')),
               'arms': _breaks, 'same_side_no_connect': no_connect}
        s.setdefault('topo_del_arm_breaks', []).append(rec)
        print(f'      (recorded in state["topo_del_arm_breaks"]; '
              f'{len(s["topo_del_arm_breaks"])} delete(s) so far)')
    _clear_topo_branch_click_viz()
    s['topo_pairing_payload'] = None
    s['topo_pairing_combos'] = None
    try:
        _topo_mst_viz(use_cached_adj=True)    # keep prior stitches alive
    except Exception as _ex:
        print(f'  [pairing] refresh warn: {_ex}')
    # DEBUG: drop the 4 stored break_pos balls for THIS delete right now, so
    # you can eyeball whether the saved coords sit at the real cut location
    # (red = break on a synthetic bridge node, green = real).
    if _breaks:
        try:
            _dbg_show_arm_breaks([rec])
        except Exception:
            pass
    try:
        _register_delete_aabbs()       # refresh the clickable region boxes
    except Exception:
        pass
















def _fit_curves(seg_labels: np.ndarray, src_pts: np.ndarray,
                K: int, smooth: float, max_turn_deg: float = 180.0
                ) -> None:
    """For each MST connected component:
        - K == 1 (one component, the whole yarn): just the main MST
          diameter as a single curve.  Peeling side branches would
          produce noisy short sub-curves we don't want at that stage.
        - K > 1: iteratively peel ALL diameter paths off the component
          (longest → remove → next-largest CC → repeat) so every
          branch becomes its own curve, not just the trunk.

    Each peeled path is split-at-sharp-turns and B-spline-fit.  Curves
    shorter than `_CURVE_MIN_LENGTH_VOX` are filtered out.
    A sorted lengths table is printed at the end."""
    global _curve_data
    _curve_data = []
    sub_to_cc: dict[int, int] = {}
    ok = fail = n_split = n_short = 0
    t0 = time.perf_counter()
    sub_idx = 0
    n_total_paths = 0
    n_total_uncov = 0
    min_L = _CURVE_MIN_LENGTH_VOX
    peel_min = max(4, int(state.get('curve_min_pts', 30)) // 3)

    def _arc_len(sp) -> float:
        if sp is None or len(sp) < 2:
            return 0.0
        a = np.asarray(sp, np.float64)
        return float(np.linalg.norm(np.diff(a, axis=0), axis=1).sum())

    for sid in range(K):
        cc_mask = seg_labels == sid
        if not cc_mask.any():
            continue
        topo_adj = state.get('topo_mst_global_adj')
        extras_pts_fit = state.get('topo_mst_extra_pts')
        if (topo_adj is not None
                and extras_pts_fit is not None
                and len(extras_pts_fit) > 0):
            src_eff_fit = np.vstack(
                [src_pts, np.asarray(extras_pts_fit, np.float64)])
        else:
            src_eff_fit = src_pts
        n_orig_fit = len(src_pts)
        if int(cc_mask.sum()) < 4:
            continue
        # Build the list of ordered point arrays for this CC.
        if topo_adj is not None:
            cc_globals = np.where(cc_mask)[0]
            if K == 1:
                # Single MST → main diameter only (no peeling).
                ordered = _order_via_topo_mst(
                    cc_globals, src_eff_fit, topo_adj,
                    n_orig=n_orig_fit)
                paths = ([ordered] if len(ordered) >= 4 else [])
                n_uncov = 0
            else:
                paths, n_uncov = _peel_diameter_paths(
                    cc_globals, src_eff_fit, topo_adj,
                    n_orig=n_orig_fit, min_len=peel_min)
            if not paths:
                comp_pts = src_pts[cc_mask].astype(np.float64)
                paths = [_order_points(comp_pts)]
                n_uncov = 0
            n_total_uncov += n_uncov
        else:
            comp_pts = src_pts[cc_mask].astype(np.float64)
            paths = [_order_points(comp_pts)]
        n_total_paths += len(paths)
        for ordered in paths:
            if len(ordered) < 4:
                continue
            subs = _split_at_sharp_turns(ordered, max_turn_deg)
            if len(subs) > 1:
                n_split += 1
            for sub in subs:
                spline = _fit_spline(sub, smooth)
                if spline is None or len(spline) < 2:
                    fail += 1
                    continue
                if _arc_len(spline) < min_L:
                    n_short += 1
                    continue           # filter too-short curves
                _curve_data.append((sub_idx, spline))
                sub_to_cc[sub_idx] = sid
                sub_idx += 1
                ok += 1
    # ── fit-dedup: drop curves that duplicate longer ones.  Twig /
    # combine leftovers peel into paths hugging the trunk within ~yarn
    # radius (measured 100% overlap @6 vox), while genuine second strands
    # overlap ~0%.  Greedy accept longest-first.
    _tau = float(state.get('curve_dedup_tau', 6.0))
    _frac = float(state.get('curve_dedup_frac', 0.6))
    if _tau > 0 and len(_curve_data) > 1:
        def _alen_(sp):
            a = np.asarray(sp, np.float64)
            return float(np.linalg.norm(np.diff(a, axis=0), axis=1).sum())
        _order = sorted(range(len(_curve_data)),
                        key=lambda k: -_alen_(_curve_data[k][1]))
        _acc = None
        _tree = None
        _keep = []
        _ndup = 0
        for _k in _order:
            _sp = np.asarray(_curve_data[_k][1], np.float64)
            if _acc is None:
                _keep.append(_k)
                _acc = _sp
                _tree = cKDTree(_acc)
                continue
            _dd, _ = _tree.query(_sp)
            _ov = float((_dd < _tau).mean())
            if _ov >= _frac:
                _ndup += 1
                print(f'  [fit-dedup] drop curve len={_alen_(_sp):.0f} vox '
                      f'({_ov:.0%} within {_tau:.0f} vox of longer curves)')
                continue
            _keep.append(_k)
            _acc = np.vstack([_acc, _sp])
            _tree = cKDTree(_acc)
        if _ndup:
            _old_data = _curve_data
            _new_data = []
            _new_map = {}
            for _new_i, _k in enumerate(sorted(_keep)):
                _old_sub, _sp = _old_data[_k]
                _new_data.append((_new_i, _sp))
                _new_map[_new_i] = sub_to_cc.get(_old_sub, -1)
            _curve_data = _new_data
            sub_to_cc = _new_map
            sub_idx = len(_curve_data)
            ok = sub_idx
            print(f'  [fit-dedup] removed {_ndup} duplicate curve(s), '
                  f'{sub_idx} remain')
    _register_curves(_curve_data, sub_idx)
    state['curve_sub_to_cc'] = sub_to_cc
    mode_str = ('single-MST diameter only' if K == 1
                else f'peeled diameters (peel_min={peel_min} pts)')
    print(f'  Fit curves: {ok} ok  {fail} failed  '
          f'{n_short} filtered (< {min_L:.1f} vox)  '
          f'(smooth={smooth:.2f}, max_turn={max_turn_deg:.0f}°, '
          f'{n_split} sub-paths split-at-turn, {n_total_paths} path(s) '
          f'from {mode_str}; {n_total_uncov} src pt(s) uncovered '
          f'[< peel_min in residual pieces])  '
          f'[{time.perf_counter() - t0:.2f}s]')
    # ── Per-curve length table, sorted longest → shortest ───────────
    if _curve_data:
        sized = sorted(((_arc_len(sp), si) for si, sp in _curve_data),
                       reverse=True)
        total_L = sum(L for L, _ in sized)
        print(f'  Curve lengths (vox)  total={total_L:.1f}  '
              f'count={len(sized)}:')
        for rank, (L, si) in enumerate(sized, start=1):
            cc_sid = sub_to_cc.get(int(si), -1)
            print(f'    [{rank:>3}] sub={si:<4d} cc_sid={cc_sid:<3d} '
                  f'L={L:8.2f}')


def _fit_curves_topo() -> None:
    """Fit per MST CONNECTED COMPONENT: one component, one curve.

    Segmentation comes from `topo_mst_global_adj`, so every manual edge
    (Sketch bridge, Connect 1↔2, >-----< accept) is honoured, including a
    connection spanning a proximity gap wider than curve_radius.  Deleted
    nodes (topo_mst_deleted, Delete A/B/C) are already disconnected there,
    so they sit at degree 0 and join no component.

    `_compute_segments(radius=4)` cannot be used here: it segments purely by
    proximity, so it never sees synthetic sketch points or bridge edges and
    splits a single MST component into dozens of pieces.
    """
    s = state
    src_pts = np.asarray(s['pts'], np.float64)
    n_orig  = len(src_pts)
    min_pts  = int(s.get('curve_min_pts', 30))
    smooth   = _CURVE_SMOOTH
    turn_deg = _CURVE_SPLIT_TURN_DEG
    topo_adj = s.get('topo_mst_global_adj')
    if not topo_adj:
        print('  [curve-topo] no topo_mst_global_adj cache — click '
              '"Topo MST viz" once (otherwise manual sketch / connect '
              'edges are ignored)'); return
    # (1) separate displacement: fit on the pulled-apart coords.
    ov = s.get('topo_mst_pos_override')
    if ov:
        src_pts = src_pts.copy()
        _oi = np.fromiter(ov.keys(), np.int64, len(ov))
        _op = np.array(list(ov.values()), np.float64)
        _m = (_oi >= 0) & (_oi < n_orig)
        src_pts[_oi[_m]] = _op[_m]
    # (2) segment by MST connected components (manual bridges count).
    comps = _global_adj_components(topo_adj)
    _del_set = s.get('topo_mst_deleted') or set()
    _del_arr = (np.fromiter(_del_set, np.int64, len(_del_set))
                if _del_set else np.zeros(0, np.int64))
    seg_labels = np.full(n_orig, -1, dtype=np.int64)
    K = 0
    sizes_real: list[int] = []
    n_dropped_small = 0
    for comp in comps:
        # only REAL gidx contribute to seg_labels; synthetic gidx
        # (≥ n_orig, e.g. sketch-bridge centerline pts) are handled
        # inside _fit_curves → _order_via_topo_mst via the topo_adj.
        real_gidx = comp[(comp >= 0) & (comp < n_orig)]
        if len(_del_arr) and len(real_gidx):
            real_gidx = real_gidx[~np.isin(real_gidx, _del_arr)]
        if len(real_gidx) < min_pts:
            n_dropped_small += 1
            continue
        seg_labels[real_gidx] = K
        sizes_real.append(int(len(real_gidx)))
        K += 1
    s['curve_seg_labels'] = seg_labels
    s['curve_K']          = K
    if K == 0:
        print(f'  [curve-topo] 0 MST components ≥ min_pts={min_pts} — '
              f'nothing to fit'); return
    n_del_real = int(np.sum((_del_arr >= 0) & (_del_arr < n_orig))) \
                 if len(_del_arr) else 0
    _sz_str = (f'sizes={sizes_real}' if len(sizes_real) <= 6
               else (f'top sizes={sorted(sizes_real, reverse=True)[:6]}'
                     f', …'))
    print(f'  [curve-topo] segmented by MST connected components '
          f'(manual sketch/connect bridges honoured; {n_del_real} deleted '
          f'excluded; override applied) → {K} segment(s), dropped '
          f'{n_dropped_small} '
          f'sub-min_pts components.  ({_sz_str})')
    _fit_curves(seg_labels, src_pts, K, smooth, turn_deg)
    _hide_orig_viz()
    _save_curves_state('fit-topo')


def _topo_pairing_show(k) -> None:
    """Show junction-pairing CANDIDATE k (0..3); the rest hidden.  Bound to
    the ← / → arrow keys in callback() while the 4 candidate clouds exist."""
    names = [f'topo_branch_pairing_{i}' for i in range(4)]
    if not any(ps.has_point_cloud(n) for n in names):
        return
    k = int(k) % 4
    for i, n in enumerate(names):
        if ps.has_point_cloud(n):
            try:
                ps.get_point_cloud(n).set_enabled(i == k)
            except Exception:
                pass
    if state.get('topo_branch_pairing_shown') != k:
        state['topo_branch_pairing_shown'] = k
        labs = state.get('topo_pairing_labels') or []
        _lbl = labs[k] if 0 <= k < len(labs) else ''
        print(f'  [pairing] candidate {k}: {_lbl}')


def _topo_pairing_accept() -> None:
    """ENTER: commit the currently-shown junction pairing (A or B) into the
    LOCAL MST — exactly process_all's mechanism (pull the >-----< apart to
    the shown side's split coords, then re-MST), but applied to ONLY this
    one connected component and written back into `topo_mst_global_adj`.

    The pulled-apart coords can't go into `state['pts']` (read-only), so we
    rewire the two strands on the SAME real gidx (no synthetic nodes — that
    keeps colour / junction / leaf / diameter recomputation working) and
    stash the pulled-apart coords as a VIZ-ONLY `topo_mst_pos_override`, so
    `_topo_mst_viz` segments the region at the split coords → two real,
    coloured, marker'd strands.  Every OTHER segment is left untouched (no
    global recompute).  Snapshot first so "Undo stitch" reverts it."""
    s = state
    payload = s.get('topo_pairing_payload')
    combos = s.get('topo_pairing_combos')
    if (payload is None or not combos
            or not ps.has_point_cloud('topo_branch_pairing_0')):
        print('  [pairing] no active candidate preview to accept (click a '
              '>-----< junction first)')
        return
    global_adj = s.get('topo_mst_global_adj')
    if global_adj is None:
        print('  [pairing] no MST adj — run "Topo MST viz" first')
        return
    shown = int(s.get('topo_branch_pairing_shown', 0)) % len(combos)
    _labs = s.get('topo_pairing_labels') or []
    _shown_lbl = _labs[shown] if shown < len(_labs) else str(shown)
    loc = np.asarray(payload['gidx'], np.int64)
    _cid_k, pos = combos[shown]
    pos = np.asarray(pos, np.float64).copy()
    if len(loc) != len(pos) or len(loc) < 4:
        print('  [pairing] payload malformed — abort'); return
    radius = float(s.get('curve_radius', 4.0))
    min_pts = int(s.get('curve_min_pts', 30))
    # ① PRESERVE earlier separates — already baked in UPSTREAM, do NOT
    # re-impose here.  `pos` (= pts_acc from combos) was built in
    # _topo_branch_explore from `src`, and `src` already has every
    # earlier-accepted split coord applied (topo_mst_pos_override).  So
    # `pos` = prior splits + THIS candidate's new displacement.  The old
    # code re-copied _ov_prev OVER pos, which RESET this junction's split
    # nodes back to their prior coords and silently DROPPED the new
    # separation whenever the two separates overlap (≈always — same
    # component); that left the clicked junction still deg-3 after ENTER.
    _ov_prev = s.get('topo_mst_pos_override') or {}
    _n_prior = sum(1 for g in loc if int(g) in _ov_prev)
    if _n_prior:
        print(f'  [pairing] {_n_prior}/{len(loc)} component nodes already '
              f'carry earlier-accepted split coords (baked into the '
              f'candidate geometry via src — prior separate(s) preserved)')
    # re-MST this component on the COMBINED split coords (this candidate +
    # all earlier separates, already composed into `pos`)
    edges_local, _seg = _mst_edges_for_positions(pos, radius, min_pts)
    if len(edges_local) == 0:
        print('  [pairing] re-MST produced no edges — abort'); return
    _topo_stitch_snapshot(
        f"accept candidate {shown} ({_shown_lbl}) @ "
        f"J{payload['J']}↔J{payload['J2']}")
    # ② rewire the component on its REAL gidx to the two re-MST'd strands
    # (clear old edges among loc, then add the new within-strand edges)
    loc_set = set(int(g) for g in loc)
    for g in loc_set:
        for nb in list(global_adj.get(g, ())):
            if int(nb) in global_adj:
                global_adj[int(nb)].discard(g)
        global_adj[g] = set()
    for a, b in edges_local:
        ga, gb = int(loc[int(a)]), int(loc[int(b)])
        global_adj[ga].add(gb); global_adj[gb].add(ga)
    # ③ stash pulled-apart coords as a VIZ-ONLY override so the next
    # _topo_mst_viz segments these nodes into two separated strands.
    # `pos` already carries both this candidate's AND any overlapping
    # earlier separate's displacement, so merging it keeps them all.
    ov = dict(_ov_prev)
    for i in range(len(loc)):
        ov[int(loc[i])] = pos[i]
    s['topo_mst_pos_override'] = ov
    print(f'  [pairing] ACCEPTED candidate {shown} ({_shown_lbl}): re-MST '
          f'{len(loc)} component nodes on split coords → {len(edges_local)} '
          f'edges; two strands rewired on real gidx (shown pulled-apart; '
          f'colour + jct/leaf/diameter recomputed).  Other segments '
          f'untouched.')
    # clear ALL junction-click overlays (the 4 pairing clouds + arms,
    # probe spheres, explorer/closest markers, next-junction ball, …) so
    # nothing from the click lingers after committing.
    _clear_topo_branch_click_viz()
    s['topo_pairing_payload'] = None
    s['topo_pairing_combos'] = None
    try:
        _topo_mst_viz(use_cached_adj=True)
    except Exception as _ex:
        print(f'  [pairing] refresh warn: {_ex}')


def _cc_mst_adj(cpts, radius: float = 4.0):
    """Radius-proximity graph (IDENTICAL to `_compute_segments`'s
    `query_pairs(r=radius)`) → MST → adjacency dict.  Returns
    (adj, M) or (None, 0).

    Using `query_pairs` (the SAME builder as CC seg) instead of a
    kNN + radius filter guarantees that wherever CC seg sees a
    connected component, MST sees the same — no spurious
    disconnects from k-NN saturation on one side of dense regions.
    """
    M = len(cpts)
    if M < 4:
        return None, 0
    tree = cKDTree(cpts)
    pairs = tree.query_pairs(r=float(radius), output_type='ndarray')
    if len(pairs) == 0:
        return None, 0
    a, b = pairs[:, 0].astype(np.int64), pairs[:, 1].astype(np.int64)
    dists = np.linalg.norm(cpts[a] - cpts[b], axis=1).astype(np.float64)
    g = csr_matrix((dists, (a, b)), shape=(M, M))
    g = g.maximum(g.T)
    if connected_components(g, directed=False)[0] != 1:
        # CC seg already deemed this connected; if the per-CC graph
        # isn't, it's a degenerate edge case (e.g. all-duplicate
        # points filtered out by query_pairs).  Bail.
        return None, 0
    mst = minimum_spanning_tree(g)
    mst = (mst + mst.T).tocoo()
    adj: dict[int, set[int]] = {i: set() for i in range(M)}
    for i, j in zip(mst.row, mst.col):
        if i != j:
            adj[int(i)].add(int(j)); adj[int(j)].add(int(i))
    return adj, M


def _mst_diameter_endpoints(adj: dict, cpts: np.ndarray):
    """Two endpoints (local idx) of the arc-length-weighted MST diameter
    (the longest internal path) via double-Dijkstra.  Returns (u, v) or
    None.  Same diameter that `_order_via_topo_mst` walks."""
    M = len(cpts)
    if M < 2:
        return None
    ii, jj, ww = [], [], []
    for i, neigh in adj.items():
        for j in neigh:
            if i < j:
                ii.append(i); jj.append(j)
                ww.append(float(np.linalg.norm(cpts[i] - cpts[j])))
    if not ii:
        return None
    g = csr_matrix((ww, (ii, jj)), shape=(M, M))
    g = g.maximum(g.T)
    d0 = shortest_path(g, method='D', indices=0)
    u = int(np.argmax(np.where(np.isfinite(d0), d0, -1.0)))
    du = shortest_path(g, method='D', indices=u)
    v = int(np.argmax(np.where(np.isfinite(du), du, -1.0)))
    return (u, v) if u != v else None


def _global_adj_components(adj: dict) -> list:
    """Connected components of the cached topo-MST adjacency `adj`
    (gidx → set(gidx), real + synthetic nodes).  Returns a list of
    np.int64 gidx arrays, largest first; degree-0 nodes are skipped.

    A search-stitch wires a bridge (other-MST diameter tip → synthetic
    chain → this stem) INTO global_adj, so the two proximity MSTs it
    joins share ONE component here.  Grouping the viz by these
    components (instead of raw proximity segments) is what MERGES two
    isolated MSTs into a single tree after a stitch.  With no bridge,
    these components are exactly the proximity components."""
    seen: set = set()
    comps: list = []
    for start in adj:
        s0 = int(start)
        if s0 in seen or len(adj[start]) == 0:
            continue
        stack = [s0]
        seen.add(s0)
        comp = [s0]
        while stack:
            u = stack.pop()
            for v in adj.get(u, ()):
                iv = int(v)
                if iv not in seen:
                    seen.add(iv)
                    comp.append(iv)
                    stack.append(iv)
        comps.append(np.asarray(sorted(comp), np.int64))
    comps.sort(key=lambda a: -len(a))
    return comps


def _refresh_pick_markers_from_gidx() -> None:
    """Re-emit polyscope `topo_mst_pick` (yellow) / `topo_mst_pick2`
    (cyan) point-cloud markers from the current
    `state['topo_mst_pick_gidx']` / `pick2_gidx`.  Also back-fills
    `topo_mst_pick_node` / `pick2_node` (curve-network node index) when
    `topo_mst_node_gidx` is available, so a subsequent Delete A/B or
    Connect 1↔2 click acts on the loaded picks without a re-click."""
    s = state
    pts = np.asarray(s.get('pts'), np.float64)
    ex  = s.get('topo_mst_extra_pts')
    if pts is None or len(pts) == 0:
        return
    if ex is not None and len(ex) > 0:
        all_pos = np.vstack([pts, np.asarray(ex, np.float64)])
    else:
        all_pos = pts
    node_gidx = s.get('topo_mst_node_gidx')
    for slot, gidx_key, node_key, name, col in (
            (1, 'topo_mst_pick_gidx',  'topo_mst_pick_node',
             'topo_mst_pick',  (1.0, 0.95, 0.10)),
            (2, 'topo_mst_pick2_gidx', 'topo_mst_pick2_node',
             'topo_mst_pick2', (0.10, 0.95, 1.0)),
    ):
        g = int(s.get(gidx_key, -1))
        if ps.has_point_cloud(name):
            ps.remove_point_cloud(name)
        if g < 0 or g >= len(all_pos):
            s[node_key] = -1
            continue
        pos = np.asarray(all_pos[g], np.float32)[None, :]
        pc = ps.register_point_cloud(name, pos)
        pc.set_color(col)
        pc.set_radius(3.0, relative=False)
        pc.set_enabled(True)
        # Backfill node_idx so Delete A/B / Connect 1↔2 work on the
        # restored pick without requiring a re-click.
        if node_gidx is not None and len(node_gidx):
            hit = np.where(np.asarray(node_gidx) == g)[0]
            s[node_key] = int(hit[0]) if len(hit) else -1
        else:
            s[node_key] = -1


def _load_camera_state(path: str | None = None) -> str | None:
    """Restore a saved camera JSON.  Default path = newest
    `segs/camera_*.json` (auto-pairs with the most-recent Save camera /
    or any saved by `r_mst_view.py`).  Applies via `ps.look_at_dir`."""
    import json, glob
    if path:
        candidates = [path]
    else:
        latest = os.path.join(_OUT_SEGS, 'camera_latest.json')
        ts_glob = sorted(glob.glob(os.path.join(_OUT_SEGS, 'camera_*.json')),
                         reverse=True)
        candidates = [latest] + ts_glob
    cam_path = next((p for p in candidates if os.path.exists(p)), None)
    if cam_path is None:
        print(f'  [cam-load] no camera_*.json under {_OUT_SEGS}/')
        return None
    try:
        with open(cam_path) as f:
            cam = json.load(f)
        pos = np.asarray(cam['cam_position'], np.float64)
        iso = np.asarray(cam['iso_dir'],      np.float64)
        up  = np.asarray(cam['up_dir'],       np.float64)
    except Exception as e:
        print(f'  [cam-load] failed to read {cam_path}: {e}')
        return None
    # iso = direction from target → camera (Mitsuba/Pipeline convention),
    # so target = pos - iso * span.  Span just sets the look distance;
    # use the current cloud's bbox so the target stays near the data.
    pts = np.asarray(state.get('pts'), np.float64)
    if pts is not None and len(pts) >= 2:
        span = float(np.linalg.norm(pts.max(0) - pts.min(0))) or 1.0
    else:
        span = 1.0
    target = pos - iso * span
    try:
        ps.look_at_dir(pos.tolist(), target.tolist(), up.tolist(),
                       fly_to=False)
        print(f'  [cam-load] applied {cam_path}\n'
              f'    iso_dir={cam["iso_dir"]}\n'
              f'    up_dir ={cam["up_dir"]}')
    except Exception as e:
        print(f'  [cam-load] look_at_dir failed: {e}')
    return cam_path


def _save_camera_state(timestamped: bool = True) -> str | None:
    """Save the current polyscope camera (position / look_dir / up_dir +
    iso_dir for Mitsuba's orthographic build_camera) to a JSON next to
    seg_state files, so r_mst_render_subset.py / r_centerline.py can
    reproduce the angle.  Writes `segs/camera_latest.json` always, plus
    `segs/camera_<ts>.json` when timestamped."""
    import json, time as _t
    try:
        cp = ps.get_view_camera_parameters()
        try:
            cam_pos  = np.asarray(cp.get_position(), np.float64)
            look_dir = np.asarray(cp.get_look_dir(), np.float64)
            up_dir   = np.asarray(cp.get_up_dir(),  np.float64)
        except Exception:
            V = np.asarray(cp.get_view_mat(), np.float64)
            R = V[:3, :3]
            cam_pos  = -R.T @ V[:3, 3]
            look_dir = R.T @ np.array([0.0, 0.0, -1.0])
            up_dir   = R.T @ np.array([0.0, 1.0,  0.0])
        iso = -look_dir / (np.linalg.norm(look_dir) + 1e-12)
        up  =  up_dir   / (np.linalg.norm(up_dir)  + 1e-12)
        cam = {
            'iso_dir':          [float(x) for x in iso],
            'up_dir':           [float(x) for x in up],
            'cam_position':     [float(x) for x in cam_pos],
            'fov_vertical_deg': float(cp.get_fov_vertical_deg()),
        }
    except Exception as e:
        print(f'  [cam-save] failed to read polyscope camera: {e}')
        return None
    os.makedirs(_OUT_SEGS, exist_ok=True)
    paths = [os.path.join(_OUT_SEGS, 'camera_latest.json')]
    if timestamped:
        paths.append(os.path.join(
            _OUT_SEGS, f'camera_{_t.strftime("%Y%m%d_%H%M%S")}.json'))
    blob = json.dumps(cam, indent=2)
    for p in paths:
        with open(p, 'w') as f:
            f.write(blob)
    print(f'  [cam-save] wrote {"  +  ".join(paths)}\n'
          f'    iso_dir={cam["iso_dir"]}\n'
          f'    up_dir ={cam["up_dir"]}')
    return paths[-1]


def _save_seg_state(timestamped: bool = True,
                    quiet: bool = False) -> None:
    """Save the current topo-MST segments/components state to
    seg_state_<ts>.npz: the point cloud `pts` (so --load_seg can show it
    DIRECTLY, no MS replay), plus `topo_mst_global_adj` (components),
    `topo_mst_extra_pts` (stitch/loop synthetic pts), `topo_mst_pos_override`
    (separate split coords) and `topo_mst_deleted` (D-deleted nodes).
    Always refreshes `seg_state_latest.npz`; when `timestamped=True`
    (manual Save) also writes a fresh, never-overwritten timestamped
    backup.  `quiet=True` skips the verbose console line (auto-saves).

    On-disk layout (npz): binary arrays for the big stuff; a small JSON
    blob in `meta` carries the scalars.
    `global_adj` is flattened CSR-style (`adj_keys` / `adj_offsets` /
    `adj_neigh`); `pos_override` as `ov_keys` + `ov_vals (M,3)`."""
    import json, time as _t
    s = state
    adj = s.get('topo_mst_global_adj')
    if not adj:
        print('  [seg-save] no topo_mst_global_adj — run "Topo MST viz" '
              'first'); return
    ov = s.get('topo_mst_pos_override') or {}
    ex = s.get('topo_mst_extra_pts')
    # ── flatten global_adj → CSR (sorted keys for determinism) ─────────
    keys_sorted = sorted(int(k) for k in adj.keys())
    neigh_lens  = [len(adj[k]) for k in keys_sorted]
    adj_keys    = np.asarray(keys_sorted, np.int64)
    adj_offsets = np.zeros(len(keys_sorted) + 1, np.int64)
    adj_offsets[1:] = np.cumsum(neigh_lens, dtype=np.int64)
    adj_neigh   = np.fromiter(
        (int(x) for k in keys_sorted for x in adj[k]),
        dtype=np.int64, count=int(adj_offsets[-1]))
    # ── pos_override → parallel int + (M,3) float arrays ───────────────
    if ov:
        ov_keys = np.asarray(sorted(int(k) for k in ov.keys()), np.int64)
        ov_vals = np.asarray([np.asarray(ov[int(k)], np.float64)
                              for k in ov_keys], np.float64)
    else:
        ov_keys = np.zeros(0, np.int64)
        ov_vals = np.zeros((0, 3), np.float64)
    deleted = np.asarray(
        sorted(int(x) for x in (s.get('topo_mst_deleted') or set())),
        np.int64)
    pts_arr = np.asarray(s['pts'], np.float32)
    extra_arr = (np.zeros((0, 3), np.float64) if ex is None or len(ex) == 0
                 else np.asarray(ex, np.float64))
    # ── Connect picks (yellow slot #1 / cyan slot #2) ──────────────────
    # Persist the user-clicked junction picks so a re-loaded session can
    # resume the same Connect workflow.  Saved as fixed-shape (2,) int64
    # array `picks_gidx = [yellow_gidx, cyan_gidx]` (−1 = unset).
    picks_gidx = np.asarray([
        int(s.get('topo_mst_pick_gidx', -1)),
        int(s.get('topo_mst_pick2_gidx', -1)),
    ], np.int64)
    # ── View-time per-node colours (so the user's "Recolour picked
    # MST" overrides round-trip through r_mst_segments) ──────────────
    _ng = s.get('topo_mst_node_gidx')
    _nc = s.get('topo_mst_node_cols')
    view_node_gidx = (np.zeros(0, np.int64) if _ng is None
                      else np.asarray(_ng, np.int64))
    view_node_cols = (np.zeros((0, 3), np.float32) if _nc is None
                      else np.asarray(_nc, np.float32))
    # Per-CC palette so polyscope's "Recolour picked MST" persists
    # across save / load (re-applied by `_topo_mst_viz`).
    _cc_pal = s.get('topo_mst_cc_palette') or {}
    cc_pal_sids = np.asarray(sorted(int(k) for k in _cc_pal.keys()),
                             np.int32)
    cc_pal_rgbs = (np.asarray([_cc_pal[int(k)] for k in cc_pal_sids],
                              np.float32)
                   if len(cc_pal_sids)
                   else np.zeros((0, 3), np.float32))
    # ── scalars → JSON blob ───────────────────────────────────────────
    meta = {
        'n_orig': int(len(s['pts'])),
        'iter':   int(s.get('iter', 0)),
        'del_arm_breaks': s.get('topo_del_arm_breaks') or [],
        'pick_gidx':  int(picks_gidx[0]),
        'pick2_gidx': int(picks_gidx[1]),
    }
    meta_json = np.asarray(json.dumps(meta, ensure_ascii=False))
    latest_path = os.path.join(_OUT_SEGS, 'seg_state_latest.npz')
    paths = [latest_path]
    if timestamped:
        paths.append(os.path.join(
            _OUT_SEGS,
            f'seg_state_{_t.strftime("%Y%m%d_%H%M%S")}.npz'))
    for path in paths:
        np.savez_compressed(
            path,
            pts=pts_arr,
            extra_pts=extra_arr,
            adj_keys=adj_keys,
            adj_offsets=adj_offsets,
            adj_neigh=adj_neigh,
            ov_keys=ov_keys,
            ov_vals=ov_vals,
            deleted=deleted,
            picks_gidx=picks_gidx,
            view_node_gidx=view_node_gidx,
            view_node_cols=view_node_cols,
            cc_palette_sids=cc_pal_sids,
            cc_palette_rgbs=cc_pal_rgbs,
            meta=meta_json,
        )
    if not quiet:
        wrote = '  +  '.join(paths)
        pick_s = ''
        if int(picks_gidx[0]) >= 0 or int(picks_gidx[1]) >= 0:
            pick_s = (f', picks=[yellow={int(picks_gidx[0])}, '
                      f'cyan={int(picks_gidx[1])}]')
        print(f'  [seg-save] wrote {wrote}: '
              f'{len(adj_keys)} nodes, {len(extra_arr)} synth pts, '
              f'{len(ov_keys)} override, {len(deleted)} deleted  '
              f'(n_orig={meta["n_orig"]}, iter={meta["iter"]}){pick_s}.  '
              f'Reload next time with: --load_seg  (no path; '
              f'auto-finds seg_state_latest.npz)')


def _save_current_points(timestamped: bool = True) -> None:
    """Save the CURRENT mean-shifted points + updated directions as an npz in
    the SAME format as the input cloud — keys `points`, `<_DIR_KEY>` (whichever
    of `directions`/`dirs` the input used), plus `energy`/`linearity` when they
    still line up (no points added/deleted since load).  Always refreshes
    ms_points_latest.npz; when `timestamped`, also writes a never-overwritten
    ms_points_<ts>.npz, both under output/<stem>/ms_points/.  Re-loadable with
        --npz output/<stem>/ms_points/ms_points_latest.npz
    Points are saved float32 (mean-shift is sub-voxel); the loader casts to
    float32 anyway, so this stays a drop-in for the input format."""
    import time as _t
    s = state
    pts  = np.ascontiguousarray(s['pts'],  np.float32)
    dirs = np.ascontiguousarray(s['dirs'], np.float32)
    out  = {'points': pts, _DIR_KEY: dirs}
    n    = len(pts)
    have, dropped = [], []
    for name, arr in (('energy', energy_orig), ('linearity', linearity_orig)):
        if arr is None:
            continue
        if len(arr) == n:
            out[name] = np.ascontiguousarray(arr, np.float32); have.append(name)
        else:
            dropped.append(name)
    paths = [os.path.join(_OUT_MS, 'ms_points_latest.npz')]
    if timestamped:
        paths.append(os.path.join(
            _OUT_MS, f'ms_points_{_t.strftime("%Y%m%d_%H%M%S")}.npz'))
    for p in paths:
        np.savez_compressed(p, **out)
    keys = ', '.join(['points', _DIR_KEY] + have)
    print(f"  [save-points] wrote {'  +  '.join(paths)}: "
          f"{n:,} pts  keys=[{keys}]  @ iter={int(s.get('iter', -1))}")
    if dropped:
        print(f"  [save-points] note: skipped {', '.join(dropped)} — length no "
              f"longer matches {n:,} live points (points were added/deleted "
              f"since load); points + {_DIR_KEY} are still correct.")


def _adopt_seg_cloud(new_pts) -> None:
    """Switch the live working cloud to a seg_state's OWN saved points.

    Used when --load_seg's saved cloud differs in size from the BinnedPcds
    source npz: per user request we IGNORE the source npz and display the
    saved segmentation directly on its own cloud.  Rebinds the module-level
    cloud globals and re-registers the polyscope 'points' / 'direction_lines'
    / 'original' structures at the new vertex count, so direct display AND
    later click / kNN ops all operate on the loaded cloud.  A seg_state does
    not store per-point directions, so directions are set to zero."""
    global pts_orig, dirs_orig, energy_orig, linearity_orig, N
    global pc, cn_dir, pc_orig, _gabor, _dir_nodes, _dir_edges, DIR_LEN, _SCALE
    new_pts   = np.ascontiguousarray(new_pts, np.float32)
    pts_orig  = new_pts
    dirs_orig = np.zeros_like(new_pts)
    energy_orig = None
    linearity_orig = None
    N = len(new_pts)
    # live working state mirrors the new cloud
    state['pts']          = new_pts.astype(np.float64)
    state['dirs']         = np.zeros_like(state['pts'])
    state['history']      = [state['pts'].copy()]
    state['history_full'] = []
    state['picked']       = 0
    # re-register the polyscope structures at the new count (the old ones were
    # built for the source-npz cloud and have a fixed, now-wrong vertex count)
    _gabor = np.zeros((N, 3), np.float32)
    pc = ps.register_point_cloud('points', pts_orig, radius=0.0016)
    pc.add_color_quantity('direction_RGB', _gabor, enabled=True)
    _bbox  = pts_orig.max(axis=0) - pts_orig.min(axis=0)
    _span  = float(np.linalg.norm(_bbox))
    _SCALE = max(_span / 60.0, 1.0)
    DIR_LEN = 1.0 * _SCALE
    _dir_nodes = np.vstack(
        [pts_orig, pts_orig + DIR_LEN * dirs_orig]).astype(np.float32)
    _dir_edges = np.column_stack(
        [np.arange(N), np.arange(N) + N]).astype(np.int32)
    cn_dir = ps.register_curve_network(
        'direction_lines', _dir_nodes, _dir_edges, radius=0.0005)
    cn_dir.add_color_quantity('direction_RGB', _gabor,
                              defined_on='edges', enabled=True)
    cn_dir.set_enabled(False)
    pc_orig = ps.register_point_cloud('original', pts_orig, radius=0.0008)
    pc_orig.set_color((0.50, 0.50, 0.55))
    pc_orig.set_enabled(False)
    print(f'  [seg-load] adopted seg cloud: N={N:,} pts  '
          f'(bbox diag {_span:.1f}; source npz ignored; directions=0)')


def _load_seg_state(path: str | None = None) -> None:
    """Restore a saved topo-MST segments/components state and SHOW IT
    DIRECTLY: render the loaded segments/components immediately.

    `path=None` → auto-find (priority):
        1. output/<stem>/segs/seg_state_latest.npz
        2. newest output/<stem>/segs/seg_state_*.npz
        3. CWD seg_state_latest.npz (legacy)
        4. newest CWD seg_state_*.npz / *.pkl (legacy)

    Accepts the new .npz format AND legacy .pkl (auto-detected by
    file content)."""
    import json, glob
    s = state
    n_now = int(len(s['pts']))
    if not path:
        candidates = [
            os.path.join(_OUT_SEGS, 'seg_state_latest.npz'),
            *sorted(glob.glob(os.path.join(_OUT_SEGS,
                                           'seg_state_*.npz')),
                    reverse=True),
            'seg_state_latest.npz',
            *sorted(glob.glob('seg_state_*.npz'), reverse=True),
            *sorted(glob.glob('seg_state_*.pkl'), reverse=True),
        ]
        path = next((p for p in candidates if os.path.exists(p)), None)
        if path is None:
            print(f'  [seg-load] auto-find: nothing in {_OUT_SEGS}/ '
                  f'or CWD — Save seg state first'); return
        print(f'  [seg-load] auto-find → {path}')
    # ── try .npz first; fall back to legacy pickle ─────────────────────
    d: dict | None = None
    is_pkl = False
    try:
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z['meta'].item()))
        adj_keys    = z['adj_keys']
        adj_offsets = z['adj_offsets']
        adj_neigh   = z['adj_neigh']
        ov_keys     = z['ov_keys']
        ov_vals     = z['ov_vals']
        deleted_arr = z['deleted']
        pts_arr     = z['pts']
        extra_arr   = z['extra_pts']
        # picks_gidx is new (older files won't have it — meta fallback below)
        picks_arr   = (z['picks_gidx'] if 'picks_gidx' in z.files
                       else np.array([-1, -1], np.int64))
        # Per-CC custom palette (only present in newer saves).
        if ('cc_palette_sids' in z.files
                and 'cc_palette_rgbs' in z.files):
            _ps = np.asarray(z['cc_palette_sids'], np.int64)
            _pr = np.asarray(z['cc_palette_rgbs'], np.float32)
            cc_palette_load = {int(s_): [float(c) for c in rgb_]
                               for s_, rgb_ in zip(_ps, _pr)}
        else:
            cc_palette_load = {}
        adj_dict = {int(adj_keys[i]):
                    set(int(x) for x in
                        adj_neigh[int(adj_offsets[i]):
                                  int(adj_offsets[i + 1])])
                    for i in range(len(adj_keys))}
        pos_override = ({int(ov_keys[i]):
                         np.asarray(ov_vals[i], np.float64)
                         for i in range(len(ov_keys))}
                        or None)
        d = {
            'n_orig':       int(meta.get('n_orig', -1)),
            'iter':         int(meta.get('iter', 0)),
            'pts':          pts_arr,
            'global_adj':   adj_dict,
            'extra_pts':    (None if len(extra_arr) == 0
                             else np.asarray(extra_arr, np.float64)),
            'pos_override': pos_override,
            'deleted':      [int(x) for x in deleted_arr],
            'del_arm_breaks': list(meta.get('del_arm_breaks') or []),
            'pick_gidx':    (int(picks_arr[0])
                             if len(picks_arr) > 0
                             else int(meta.get('pick_gidx', -1))),
            'pick2_gidx':   (int(picks_arr[1])
                             if len(picks_arr) > 1
                             else int(meta.get('pick2_gidx', -1))),
            'cc_palette':   cc_palette_load,
        }
    except Exception as _ex_npz:
        # legacy pickle fallback
        try:
            import pickle
            with open(path, 'rb') as f:
                raw = pickle.load(f)
            d = {
                'n_orig':       int(raw.get('n_orig', -1)),
                'iter':         int(raw.get('iter', 0)),
                'pts':          raw.get('pts'),
                'global_adj':   {int(k): set(int(x) for x in v)
                                 for k, v in raw.get('global_adj',
                                                      {}).items()},
                'extra_pts':    raw.get('extra_pts'),
                'pos_override': ({int(k):
                                  np.asarray(v, np.float64)
                                  for k, v in raw.get('pos_override',
                                                       {}).items()}
                                 or None),
                'deleted':      list(raw.get('deleted') or []),
            }
            is_pkl = True
            print(f'  [seg-load] (legacy .pkl detected — npz failed: '
                  f'{_ex_npz})')
        except Exception as ex2:
            print(f'  [seg-load] FAILED to read {path}: npz={_ex_npz}; '
                  f'pkl={ex2}')
            return
    n_saved = int(d.get('n_orig', -1))
    _saved_pts0 = d.get('pts')
    _have_cloud = (_saved_pts0 is not None and n_saved > 0
                   and len(_saved_pts0) == n_saved)
    if n_saved != n_now:
        if _have_cloud:
            # Self-contained seg_state (its own pts + adjacency).  The BinnedPcds
            # source npz differs in size — ignore it and show the saved
            # segmentation directly on its OWN cloud (per user request).
            print(f'  [seg-load] point-count differs (saved {n_saved} vs source '
                  f'npz {n_now}) — IGNORING the BinnedPcds npz and showing the '
                  f'seg cloud directly.')
            _adopt_seg_cloud(_saved_pts0)
            n_now = n_saved            # fire the direct-display branch below
        else:
            print(f'  [seg-load] ⚠ point-count mismatch: saved n_orig={n_saved} '
                  f'but this cloud has {n_now}, and this file has NO saved cloud '
                  f'(legacy pkl) — cannot show directly.  ABORT.')
            return
    adj = (d['global_adj'] if is_pkl
           else d['global_adj'])
    s['topo_mst_global_adj'] = adj
    ex = d.get('extra_pts')
    s['topo_mst_extra_pts'] = (None if ex is None
                               else np.asarray(ex, np.float64))
    s['topo_mst_pos_override'] = d.get('pos_override')
    s['topo_mst_deleted'] = set(int(x) for x in d.get('deleted', []))
    s['topo_del_arm_breaks'] = list(d.get('del_arm_breaks') or [])
    # Restore Connect picks (yellow slot #1 / cyan slot #2) if the saved
    # file carried them.  Polyscope markers (`topo_mst_pick` /
    # `topo_mst_pick2`) get re-emitted on the next viz refresh; here we
    # just put the gidx back into state so a Connect 1↔2 click sees them.
    pg1 = int(d.get('pick_gidx', -1))
    pg2 = int(d.get('pick2_gidx', -1))
    s['topo_mst_pick_gidx']  = pg1
    s['topo_mst_pick2_gidx'] = pg2
    # Restore the user's "Recolour picked MST" palette so the next
    # `_topo_mst_viz` re-paints each CC with the custom colour.
    _pal = d.get('cc_palette') or {}
    if _pal:
        s['topo_mst_cc_palette'] = {int(k): list(v) for k, v in _pal.items()}
        print(f'  [seg-load] restored {len(_pal)} MST colour override(s)')
    else:
        s['topo_mst_cc_palette'] = {}
    s['seg_loaded'] = True
    _saved_pts = d.get('pts')
    _n_ex = (0 if s['topo_mst_extra_pts'] is None
             else len(s['topo_mst_extra_pts']))
    if _saved_pts is not None and len(_saved_pts) == n_now:
        # DIRECT display: restore the saved cloud (load-snapshot, like a
        # reset) and render the loaded topo now.
        s['pts'] = np.asarray(_saved_pts, np.float64)
        s['history'] = [s['pts'].copy()]
        s['history_full'] = []
        s['iter'] = int(d.get('iter', 0))
        s['view_iter'] = int(d.get('iter', 0))
        s['shown_iter'] = -1
        print(f'  [seg-load] restored from {path}: {len(adj)} nodes, '
              f'{_n_ex} synth pts, {len(s["topo_mst_pos_override"] or {})} '
              f'override, {len(s["topo_mst_deleted"])} deleted (@ iter '
              f'{s["iter"]}).  Showing the loaded segments directly.')
        try:
            _update_viz()
        except Exception as ex:
            print(f'  [seg-load] viz update warn: {ex}')
        try:
            _topo_mst_viz(use_cached_adj=True)
        except Exception as ex:
            print(f'  [seg-load] topo render warn: {ex}')
        # Re-emit the yellow / cyan pick markers AFTER topo viz so
        # `topo_mst_node_gidx` is populated for the node-idx back-fill.
        if (int(s.get('topo_mst_pick_gidx', -1)) >= 0
                or int(s.get('topo_mst_pick2_gidx', -1)) >= 0):
            try:
                _refresh_pick_markers_from_gidx()
            except Exception as ex:
                print(f'  [seg-load] pick-marker warn: {ex}')
        if ps.has_point_cloud('points'):
            try:
                ps.get_point_cloud('points').set_enabled(False)
            except Exception:
                pass
    else:
        # OLD pkl (no saved cloud): nothing to display directly.
        print(f'  [seg-load] restored from {path}: {len(adj)} nodes, '
              f'{_n_ex} synth pts, {len(s["topo_mst_pos_override"] or {})} '
              f'override, {len(s["topo_mst_deleted"])} deleted (@ iter '
              f'{int(d.get("iter", 0))}).  ⚠ old pkl has NO saved cloud — '
              f're-Save to enable direct display.')


# Red-mark an MST node as a junction only when EVERY branch's subtree
# holds more than this many points (filters micro-junction clusters).
_TOPO_JCT_MIN_BRANCH_PTS = 20    # fixed (formerly a GUI slider)


def _topo_mst_viz(use_cached_adj: bool = False) -> None:
    """Viz: per CC build the SAME MST that the branch explorer uses
    (kNN + edge-cap 4 vox, just like _order_points / _cc_mst_adj) and
    overlay it as `topo_mst` — edges coloured by CC; junctions (deg≥3)
    marked RED; leaves (deg=1) marked GREEN.  Lets you SEE exactly where
    the kNN graph reaches and where it does NOT (two MST sub-trees that
    are visually adjacent but in different colours = a gap > 4 vox that
    the current reach can't cross).

    If `use_cached_adj=True`, skip the per-CC MST rebuild and use the
    existing `state['topo_mst_global_adj']` as the source of truth.
    This is how a stitch (which mutates global_adj in place) refreshes
    the viz WITHOUT undoing itself — junctions/leaves get re-derived
    from the modified adj, so a 3→2 degree drop makes its red ball
    disappear."""
    s = state
    # Use LIVE state['pts'] directly (always matches what polyscope's
    # `points` cloud is currently showing, since `_update_viz` updates
    # both at the same moment).  Earlier this read `_displayed_state()`
    # which followed the history scrubber and could disagree with the
    # visible cloud.
    src_pts = np.asarray(s['pts'], np.float64)
    if len(src_pts) < 4:
        print('  [topo-mst] too few points'); return
    # A fresh rebuild resets the centerline extras (a stitch is built
    # ON TOP of cached_adj; rebuilding from scratch starts clean).
    if not use_cached_adj:
        s['topo_mst_extra_pts'] = None
        s['topo_mst_pos_override'] = None
    # [pairing-accept] viz-only position override: accepted >-----< nodes
    # are shown at their pulled-apart coords so _compute_segments splits
    # them into two real, colour + marker'd strands (no synthetic nodes).
    _ov = s.get('topo_mst_pos_override')
    if _ov:
        src_pts = src_pts.copy()
        _oidx = np.fromiter(_ov.keys(), np.int64, len(_ov))
        _opos = np.array(list(_ov.values()), np.float64)
        _om = (_oidx >= 0) & (_oidx < len(src_pts))
        src_pts[_oidx[_om]] = _opos[_om]
    # Effective src includes synthetic centerline pts so gidx ≥ N can
    # be indexed directly.  _compute_segments still segments state['pts']
    # only (extras have no CC membership by design).
    extras_pts = s.get('topo_mst_extra_pts')
    n_orig = len(src_pts)
    if extras_pts is not None and len(extras_pts) > 0:
        src = np.vstack([src_pts, np.asarray(extras_pts, np.float64)])
    else:
        src = src_pts
    print(f'  [topo-mst] cloud: state[\'pts\'] @ live_iter='
          f'{int(s.get("iter", -1))}  (view_iter={int(s.get("view_iter", -1))}, '
          f'N={n_orig} pts; +{len(src) - n_orig} synthetic centerline pts)')
    radius  = float(s.get('curve_radius', 4.0))
    min_pts = int(s.get('curve_min_pts', 30))
    jct_min = _TOPO_JCT_MIN_BRANCH_PTS
    # Bury permanently-DELETED nodes (D-key on a >-----<) far away before
    # segmentation so they neither form a component NOR bridge real points
    # — a FULL recompute then CANNOT bring them back, and the regions they
    # used to join split into clean separate strands.  (state['pts'] is
    # untouched; only this segmentation-input copy relocates them.)
    _del = s.get('topo_mst_deleted')
    seg_src = src_pts
    if _del:
        _dl = np.fromiter(_del, np.int64, len(_del))
        _dl = _dl[(_dl >= 0) & (_dl < len(src_pts))]
        if len(_dl):
            seg_src = src_pts.copy()
            seg_src[_dl] = (1e7 + np.arange(len(_dl), dtype=np.float64)
                            * 1000.0)[:, None]
            print(f'  [topo-mst] excluding {len(_dl)} permanently-deleted '
                  f'node(s) from the rebuild')
    seg_labels, K = _compute_segments(seg_src, radius, min_pts)
    if K == 0:
        print('  [topo-mst] 0 segments'); return
    nodes, edges, cols = [], [], []
    node_cc_parts = []            # flat topo_mst node # → CC sid
    base_cc_palette: dict[int, tuple[float, float, float]] = {}
    node_gidx_parts = []          # flat topo_mst node # → global cloud gidx
    diam_a, diam_b = [], []       # per-CC diameter endpoints (purple/orange)
    diam_a_gidx, diam_b_gidx = [], []
    junctions, leaves = [], []
    junction_gidx, leaf_gidx = [], []   # global idx into src (the cloud)
    junction_sid, leaf_sid = [], []     # CC id for each marker
    # per-junction richer info for the click-to-print callback:
    #   list of {'gidx', 'sid', 'branches': [{'cnt','capped','first_gidx'}]}
    junction_info: list[dict] = []
    leaf_info:     list[dict] = []      # one per leaf: {'gidx','sid','neigh_gidx'}
    n_cc_done = 0
    off = 0
    # cap_full controls how far we BFS branches for the click info.
    # Larger = more accurate "true" subtree size, slower for the few
    # real-junctions that pass the speed check above.  10k is plenty
    # to distinguish twig (≤50) vs arm (~100s) vs trunk (1000s+).
    cap_full = 10000
    # Accumulate a GLOBAL MST adj (union of per-CC MSTs) into a single
    # dict; stashed in state so the click-triggered branch explorer
    # doesn't have to rebuild it on every click.
    cached_adj = (s.get('topo_mst_global_adj')
                  if use_cached_adj else None)
    if use_cached_adj and cached_adj is None:
        print('  [topo-mst] use_cached_adj=True but no cached adj — '
              'falling back to rebuild')
        use_cached_adj = False
    global_adj: dict[int, set[int]] = (
        cached_adj if use_cached_adj else {})
    # ── Grouping of rendered sub-trees.
    # Cached path groups by global_adj CONNECTED COMPONENTS (so a stitch
    # that bridged two proximity MSTs renders + analyses them as ONE
    # merged tree — see _global_adj_components).  The synthetic bridge
    # nodes (gidx >= n_orig) ride along in whichever component they were
    # wired into, so the two halves actually connect.  Fresh rebuild
    # keeps the raw proximity segmentation.
    if use_cached_adj:
        _groups = _global_adj_components(cached_adj)
        n_groups = len(_groups)
    else:
        _groups = None
        n_groups = K
    for sid in range(n_groups):
        if use_cached_adj:
            # One merged tree = one global_adj component (already
            # filtered to degree>0 nodes; stitched-away gidx are GONE).
            cc_global = _groups[sid]
            if len(cc_global) < 2:
                continue
            # PRUNE cut/loop leftovers: a cut/loop that severs a junction
            # arm can leave a tiny isolated fragment (e.g. 2–6 pts) dangling
            # in global_adj.  Drop every topo component smaller than the
            # SAME min_pts the proximity segmentation uses — clear its edges
            # in cached_adj (→ degree 0) so it stops showing as a thin line
            # and stops being fit/reported.  Keeps component count aligned
            # with segment count.  state['pts'] is untouched.
            if len(cc_global) < min_pts:
                for _g in cc_global:
                    ig = int(_g)
                    for _nb in list(cached_adj.get(ig, ())):
                        cached_adj[ig].discard(int(_nb))
                        if int(_nb) in cached_adj:
                            cached_adj[int(_nb)].discard(ig)
                print(f'  [topo-mst] pruned tiny component '
                      f'({len(cc_global)} pts < min_pts={min_pts}) — '
                      f'cut/loop leftover')
                continue
            cpts      = src[cc_global]
            gi_to_li = {int(g): i for i, g in enumerate(cc_global)}
            adj = {i: set() for i in range(len(cc_global))}
            for ig, local_i in gi_to_li.items():
                for gn in cached_adj.get(ig, ()):
                    ign = int(gn)
                    if ign in gi_to_li:
                        adj[local_i].add(gi_to_li[ign])
            M = len(cc_global)
        else:
            cc_mask = (seg_labels == sid)
            cpts = src_pts[cc_mask]   # state['pts'] only (cc_mask n_orig-long)
            cc_global = np.where(cc_mask)[0]  # local idx → global cloud idx
            if len(cpts) < 4:
                continue
            adj, M = _cc_mst_adj(cpts, radius=radius)
            if adj is None:
                continue
            # contribute this CC's MST edges into the global adj
            for i_loc, neigh_loc in adj.items():
                gi = int(cc_global[i_loc])
                slot = global_adj.setdefault(gi, set())
                for n_loc in neigh_loc:
                    slot.add(int(cc_global[n_loc]))
        rgb = colorsys.hsv_to_rgb((sid * 0.61803398875) % 1.0, .65, .95)
        nodes.append(cpts.astype(np.float32))
        node_gidx_parts.append(np.asarray(cc_global, np.int64))
        # diameter endpoints of THIS CC's MST (purple u / orange v)
        _ends = _mst_diameter_endpoints(adj, cpts)
        if _ends is not None:
            _u, _v = _ends
            diam_a.append(cpts[_u]); diam_b.append(cpts[_v])
            diam_a_gidx.append(int(cc_global[_u]))
            diam_b_gidx.append(int(cc_global[_v]))
        seen = set()
        loc_e = []
        for i, neigh in adj.items():
            for j in neigh:
                a, b = (i, j) if i < j else (j, i)
                if (a, b) in seen:
                    continue
                seen.add((a, b))
                loc_e.append((a + off, b + off))
        edges.append(np.asarray(loc_e, np.int32) if loc_e
                     else np.zeros((0, 2), np.int32))
        # Apply user-stashed CC colour override (set via the "Recolour
        # picked MST" UI) so re-runs of topo_mst_viz keep the user's
        # custom palette across stitch / delete / connect mutations.
        _user_override = (state.get('topo_mst_cc_palette') or {}).get(int(sid))
        if _user_override is not None:
            rgb = tuple(float(c) for c in _user_override)
        cols.append(np.tile(rgb, (M, 1)).astype(np.float32))
        node_cc_parts.append(np.full(M, int(sid), dtype=np.int32))
        base_cc_palette[int(sid)] = (float(rgb[0]),
                                     float(rgb[1]),
                                     float(rgb[2]))
        for i, neigh in adj.items():
            # Use GLOBAL degree (from cached_adj) when extras exist:
            # synthetic centerline gidx (>= n_orig) are not in cc_global
            # so the local `adj` undercounts.  Without this fix, J and
            # o_jct (which gain an edge to a synthetic gidx after a
            # stitch) drop to local-degree 1 and get falsely marked as
            # leaves (green ball).  With it, they show true degree 2
            # → neither leaf nor junction (correct).
            gi_full = int(cc_global[i])
            d = (len(global_adj.get(gi_full, set()))
                 if global_adj is not None and gi_full in global_adj
                 else len(neigh))
            if d == 1:
                leaves.append(cpts[i])
                leaf_gidx.append(int(cc_global[i]))
                leaf_sid.append(int(sid))
                # leaf has exactly 1 MST neighbour — record it
                ne = next(iter(neigh))
                leaf_info.append({
                    'gidx': int(cc_global[i]),
                    'sid':  int(sid),
                    'neigh_gidx': int(cc_global[ne]),
                })
                continue
            if d < 3:
                continue
            # red ONLY if EVERY neighbour's SUBTREE holds more than `jct_min`
            # (`_TOPO_JCT_MIN_BRANCH_PTS`) nodes, so a tight cluster of micro-
            # junctions reads as one small lump.  (Walking to the first non-deg-2
            # node failed: cluster junctions are 1-2 hops apart and pass for each
            # other.)  PERF: early-exit once a subtree crosses the threshold rather
            # than walking up to ~100k nodes.
            br_info = []                    # subtree_pts (capped)
            real = True
            cap = jct_min + 1               # "> jct_min" → need cap pts
            for n0 in neigh:
                stack = [n0]
                seen = {i, n0}
                sub_cnt = 0
                while stack and sub_cnt < cap:
                    u = stack.pop()
                    sub_cnt += 1
                    for w in adj[u]:
                        if w not in seen:
                            seen.add(w)
                            stack.append(w)
                br_info.append(sub_cnt)
                if sub_cnt <= jct_min:
                    real = False
                    break                   # short-circuit at first fail
            if real:
                junctions.append(cpts[i])
                junction_gidx.append(int(cc_global[i]))
                junction_sid.append(int(sid))
                # Second-pass per-branch BFS with cap_full so the
                # click handler can both show "size ≥ N" AND tint
                # the branch's nodes a different colour.  Collect
                # the actual node list (CC-local idx), not just count.
                branches: list[dict] = []
                for n0 in neigh:
                    stk2 = [n0]; sn2 = {i, n0}
                    local_nodes: list[int] = []
                    while stk2 and len(local_nodes) < cap_full:
                        u2 = stk2.pop()
                        local_nodes.append(int(u2))
                        for w2 in adj[u2]:
                            if w2 not in sn2:
                                sn2.add(w2); stk2.append(w2)
                    branches.append({
                        'cnt':         len(local_nodes),
                        'capped':      len(local_nodes) >= cap_full,
                        'first_gidx':  int(cc_global[n0]),
                        'local_nodes': local_nodes,
                    })
                junction_info.append({
                    'gidx':         int(cc_global[i]),
                    'sid':          int(sid),
                    'pos':          cpts[i].tolist(),
                    'branches':     branches,
                    'cc_offset':    int(off),     # global node-idx offset
                    'cc_n_nodes':   int(M),       # nodes in this CC
                    'cc_base_rgb':  list(rgb),    # base CC colour
                    'junction_local': int(i),     # the junction itself
                })
                br_s = ', '.join(
                    f'(sub{"≥" if b["capped"] else "="}{b["cnt"]}pts)'
                    for b in branches)
                print(f'  [topo-mst][red] sid={sid} deg={d} at '
                      f'{np.round(cpts[i], 1)} → {br_s}  '
                      f'(global_idx={int(cc_global[i])})')
        off += M
        n_cc_done += 1
    # ── Render synthetic centerline chain (extras) as a separate
    # WHITE curve_network `topo_mst_extras`.  Includes both the
    # synthetic gidx (>= n_orig) and any state['pts'] gidx
    # immediately adjacent to them (= chain endpoints J / o_jct).
    if ps.has_curve_network('topo_mst_extras'):
        ps.remove_curve_network('topo_mst_extras')
    if extras_pts is not None and len(extras_pts) > 0:
        synth_set: set[int] = set()
        for _u, _ne in global_adj.items():
            if int(_u) >= n_orig:
                synth_set.add(int(_u))
            for _v in _ne:
                if int(_v) >= n_orig:
                    synth_set.add(int(_v))
        endpoint_set: set[int] = set()
        for _u in synth_set:
            for _v in global_adj.get(_u, ()):
                if int(_v) < n_orig:
                    endpoint_set.add(int(_v))
        ex_nodes_gidx = sorted(synth_set | endpoint_set)
        if ex_nodes_gidx:
            ex_g2l = {g: i for i, g in enumerate(ex_nodes_gidx)}
            ex_pos = src[np.asarray(ex_nodes_gidx)]
            seen_ex: set[tuple[int, int]] = set()
            ex_loc_e: list[tuple[int, int]] = []
            for _u in ex_nodes_gidx:
                for _v in global_adj.get(_u, ()):
                    iv2 = int(_v)
                    if iv2 not in ex_g2l:
                        continue
                    a, b = (_u, iv2) if _u < iv2 else (iv2, _u)
                    if (a, b) in seen_ex:
                        continue
                    seen_ex.add((a, b))
                    ex_loc_e.append((ex_g2l[a], ex_g2l[b]))
            if ex_loc_e:
                cne = ps.register_curve_network(
                    'topo_mst_extras',
                    ex_pos.astype(np.float32),
                    np.asarray(ex_loc_e, np.int32))
                cne.set_radius(0.90, relative=False)
                # User-added stitches / sketch-bridges (touching any
                # synthetic centerline pt) — render uniformly ORANGE so
                # they pop against the per-CC pastel MSTs.
                cne.set_color((1.0, 0.55, 0.10))
    # stash for the click-callback: parallel-indexed with the two
    # point clouds (`topo_mst_junctions` / `topo_mst_leaves` below).
    s['topo_mst_jct_info']  = junction_info
    s['topo_mst_global_adj'] = global_adj    # used by branch explorer
    s['topo_mst_leaf_info'] = leaf_info
    # Map the `topo_mst` curve-network node # (what Polyscope's
    # Selection panel shows on click) → global cloud gidx.
    s['topo_mst_node_gidx'] = (np.concatenate(node_gidx_parts)
                               if node_gidx_parts else np.zeros(0, np.int64))
    # MST recolour bookkeeping — per curve-network node CC sid + the
    # flat colour buffer the curve_network's 'cc' quantity points at.
    # The UI "Recolour picked MST" overwrites slices of `node_cols` and
    # re-adds the colour quantity so the change is live.
    s['topo_mst_node_cc']   = (np.concatenate(node_cc_parts)
                               if node_cc_parts else np.zeros(0, np.int32))
    s['topo_mst_node_cols'] = (np.vstack(cols).astype(np.float32)
                               if cols
                               else np.zeros((0, 3), np.float32))
    s['topo_mst_base_cc_palette'] = base_cc_palette
    if 'topo_mst_cc_palette' not in s or s.get('topo_mst_cc_palette') is None:
        s['topo_mst_cc_palette'] = {}
    # reset click-change gate so the first click on the freshly-rebuilt
    # junction/leaf set always prints (indices may map to different
    # physical points after re-running Topo MST viz).
    s['topo_mst_last_jct']  = -1
    s['topo_mst_last_leaf'] = -1
    if ps.has_curve_network('topo_mst'):
        ps.remove_curve_network('topo_mst')
    if ps.has_point_cloud('topo_mst_junctions'):
        ps.remove_point_cloud('topo_mst_junctions')
    if ps.has_point_cloud('topo_mst_leaves'):
        ps.remove_point_cloud('topo_mst_leaves')
    for _dnm in ('topo_mst_diam_a', 'topo_mst_diam_b'):
        if ps.has_point_cloud(_dnm):
            ps.remove_point_cloud(_dnm)
    n_edges = int(sum(len(e) for e in edges))
    _cn_edges = (np.vstack(edges) if edges
                 else np.zeros((0, 2), np.int32))
    _cn_nodes_xyz = (np.vstack(nodes)
                     if nodes else np.zeros((0, 3), np.float32))
    # cache the curve_network node positions + edge array so the click
    # handler can resolve an edge pick → endpoint and drop a marker.
    s['topo_mst_curve_nodes'] = _cn_nodes_xyz
    s['topo_mst_curve_edges'] = _cn_edges
    s['topo_mst_last_clickdel'] = -1   # reset gate on rebuild
    # node #s shift after a rebuild, so drop any stale pick.
    s['topo_mst_pick_node']  = -1
    s['topo_mst_pick_gidx']  = -1
    s['topo_mst_pick_deg']   = -1
    s['topo_mst_pick2_node'] = -1
    s['topo_mst_pick2_gidx'] = -1
    s['topo_mst_pick2_deg']  = -1
    for _pn_nm in ('topo_mst_pick', 'topo_mst_pick2'):
        if ps.has_point_cloud(_pn_nm):
            ps.remove_point_cloud(_pn_nm)
    if nodes:
        cn = ps.register_curve_network(
            'topo_mst', _cn_nodes_xyz, _cn_edges)
        cn.add_color_quantity('cc', np.vstack(cols),
                              defined_on='nodes', enabled=True)
        # ABSOLUTE vox radius (NOT * _SCALE — that's span-dependent
        # and inflates on big real clouds).  Edges thin (overlay, not
        # dominant); junction/leaf markers modest so they tag the spot
        # without swallowing it.  (1.5× the original sizes — were a bit
        # thin: edges 0.60→0.90, jct 2.40→3.60, leaf 1.00→1.50.)
        cn.set_radius(float(state.get('topo_mst_viz_radius', 0.90)),
                      relative=False)
    if junctions:
        jp = ps.register_point_cloud(
            'topo_mst_junctions',
            np.asarray(junctions, np.float32))
        jp.set_color((1.0, 0.15, 0.15))         # red
        jp.set_radius(3.60, relative=False)     # 1.5× (was 2.40)
        # global cloud index (hover to read; leave OFF so the red
        # colour stays as the primary visual)
        jp.add_scalar_quantity(
            'global_idx (orig cloud)',
            np.asarray(junction_gidx, np.float64),
            enabled=False)
        jp.add_scalar_quantity(
            'cc_sid',
            np.asarray(junction_sid, np.float64),
            enabled=False)
    if leaves:
        lp = ps.register_point_cloud(
            'topo_mst_leaves',
            np.asarray(leaves, np.float32))
        lp.set_color((0.15, 1.0, 0.55))         # green
        lp.set_radius(1.50, relative=False)     # 1.5× (was 1.00)
        # Same: hover any green ball to read its index in the source
        # point cloud (`points` / `_displayed_state()['pts']`); use
        # this idx to investigate "why didn't this link reach?".
        lp.add_scalar_quantity(
            'global_idx (orig cloud)',
            np.asarray(leaf_gidx, np.float64),
            enabled=False)
        lp.add_scalar_quantity(
            'cc_sid',
            np.asarray(leaf_sid, np.float64),
            enabled=False)
    # per-CC MST diameter (longest internal path) endpoints: PURPLE = one
    # end, ORANGE = the other.  One pair per connected MST.
    if diam_a:
        dap = ps.register_point_cloud(
            'topo_mst_diam_a', np.asarray(diam_a, np.float32))
        dap.set_color((0.62, 0.20, 0.92))       # purple
        dap.set_radius(2.60, relative=False)
        dap.add_scalar_quantity('global_idx (orig cloud)',
                                np.asarray(diam_a_gidx, np.float64),
                                enabled=False)
        dap.set_enabled(False)                   # hidden by default
    if diam_b:
        dbp = ps.register_point_cloud(
            'topo_mst_diam_b', np.asarray(diam_b, np.float32))
        dbp.set_color((1.0, 0.55, 0.10))        # orange
        dbp.set_radius(2.60, relative=False)
        dbp.add_scalar_quantity('global_idx (orig cloud)',
                                np.asarray(diam_b_gidx, np.float64),
                                enabled=False)
        dbp.set_enabled(False)                   # hidden by default
    print(f'  [topo-mst] {K} CCs ({n_cc_done} with MST)  '
          f'edges={n_edges}  junctions={len(junctions)} '
          f'leaves={len(leaves)} diam-pairs={len(diam_a)} (kNN cap=4.0 vox) '
          f'→ "topo_mst" + red/green + purple/orange diameter-ends')


# Shared depth cap for every BFS rooted at a clicked red junction
# (trunk-probe next-junction search range, and arm-subtree collection
# for the >-----< highlight).  Hops from the blocked junction; nodes at
# depth > this are NOT collected.  KEEP this large so the trunk-probe can
# still reach a far second junction (e.g. ~312 hops away).
_TOPO_BRANCH_MAX_DEPTH = 1000
# Separate (smaller) threshold for "is the SHORTER arm long enough to be
# treated as a real arm (→ trunk-probe / separate) rather than a short
# dead-end to bridge-or-cut".  Lower = more junctions go to SEPARATE
# instead of auto-cutting a real arm.  (Was sharing the 1000 cap, which
# wrongly cut real ~200-deep arms.)
_TOPO_ARM_LONG_DEPTH = 1000
# How many hops of EACH arm the 'D' delete removes (from the junction).
# The >-----< HIGHLIGHT / pairing still use ARM_MAX_DEPTH (=MAX_DEPTH//10
# =100); this is delete-only and smaller so the arm body survives as a
# separate strand (trunk + both junctions are always fully deleted).
# Click-junction analysis, fixed (formerly GUI sliders)
_TOPO_JCT_SPHERE_R = 10.0        # vox: probe sphere at J; where the branch
#                                  direction vectors are sampled
_TOPO_PAIR_DISPLACE_VOX = 15.0   # vox: >-----< split-apart distance along PCA-2
_TOPO_STEM_DENSITY_WEIGHT = 0.6  # 3-way stem pick: 0 = pure angle, 1 = pure density


def _smooth_bridge_pts(p_prev, p0, p1, p_next, n_interior,
                       handle_frac=0.4):
    """Cubic-Bezier interior samples on the p0→p1 segment.

    The bridge LEAVES p0 along (p0 − p_prev) and ARRIVES at p1 along
    (p_next − p1) — i.e. tangent to the existing chains at both ends —
    so it eases in/out instead of teeing in as a hard straight line.
    Crucially the Bezier handle length scales with the CHORD (× handle_
    frac), NOT with the neighbour spacing: the anchors are typically
    only ~1 curve_radius (a few vox) from the endpoints, so a
    distance-weighted tangent (e.g. centripetal Catmull-Rom) collapses
    to a near-straight line.  Using only the anchor *direction* and a
    chord-scaled magnitude keeps the bend clearly visible.  Returns the
    n_interior interior points only (endpoints excluded), ordered p0→p1.
    """
    p_prev = np.asarray(p_prev, np.float64)
    p0     = np.asarray(p0,     np.float64)
    p1     = np.asarray(p1,     np.float64)
    p_next = np.asarray(p_next, np.float64)
    u = np.linspace(0.0, 1.0, n_interior + 2)[1:-1]
    chord = p1 - p0
    L = float(np.linalg.norm(chord))
    if L < 1e-9:
        return p0[None, :] + u[:, None] * chord[None, :]
    chord_dir = chord / L

    def _dir(v):
        n = float(np.linalg.norm(v))
        return (v / n) if n > 1e-9 else chord_dir
    d0 = _dir(p0 - p_prev)          # depart p0 continuing past the tip
    d1 = _dir(p_next - p1)          # arrive p1 along the stem-forward dir
    h = L * float(handle_frac)
    b0 = p0
    b1 = p0 + d0 * h
    b2 = p1 - d1 * h
    b3 = p1
    uu = u[:, None]
    mt = 1.0 - uu
    return (mt ** 3 * b0 + 3 * mt ** 2 * uu * b1
            + 3 * mt * uu ** 2 * b2 + uu ** 3 * b3)


def _clear_topo_branch_click_viz() -> None:
    """Remove every viz layer that a red-junction click can produce
    (explorer pts, r=5 sphere, closest-to-sphere markers, premerge
    halo, bridge leaf/cands, loop curve_network + half coloring,
    centerline, the OTHER-junction ball + its off-trunk dots).  Used
    by `_clear_all_selection` and after an auto-stitch (since the
    clicked red junction itself ceases to exist post-stitch)."""
    for _nm in ('topo_branch_explorer', 'topo_branch_closest',
                'topo_branch_premerge', 'topo_branch_bridge_leaf',
                'topo_branch_bridge_cands', 'topo_branch_other_jct',
                'topo_branch_other_trunk', 'topo_branch_next_jct',
                'topo_branch_pairing_0', 'topo_branch_pairing_1',
                'topo_branch_pairing_2', 'topo_branch_pairing_3',
                'topo_branch_loose_leaf', 'topo_branch_stem_attach',
                'topo_branch_attach_pts', 'topo_branch_reject_jct',
                'topo_branch_density'):
        if ps.has_point_cloud(_nm):
            ps.remove_point_cloud(_nm)
    for _cn in ('topo_branch_sphere', 'topo_branch_loop',
                'topo_branch_centerline', 'topo_branch_shared_trunk',
                'topo_branch_arms', 'topo_branch_attach_link'):
        if ps.has_curve_network(_cn):
            ps.remove_curve_network(_cn)


def _topo_branch_explore(start_J: int, n_per_branch: int = 100,
                         auto: bool = False,
                         apply_action: str | None = None):
    """Click-triggered on a red MST junction.  BFS-collects up to
    `n_per_branch` points per MST neighbour and renders each branch in its
    own colour, so the user sees where each branch goes.  No sub-junction
    heuristics.  Output: a `topo_branch_explorer` point cloud, replaced on
    every click.

    `auto` / `apply_action` — classify mode, used by the Connect
    auto-process to spot a collapse without touching anything:
      • auto=False (the click path): full viz + apply whatever the junction
        classifies as.
      • auto=True: CLASSIFY and return the action code
        ('search'|'loop'|'cut'|'separate'|'skip'), mutating ONLY when the
        action equals `apply_action`, so a caller can drain one phase at a
        time.  No snapshot and no viz in auto mode; the caller renders and
        snapshots itself.  'separate' never mutates.

    Returns (action, payload); payload is a dict with 'J'/'J2'/'gidx' for
    'separate' (None if that >-----< is malformed, ≠4 arms), else None.
    """
    s = state
    # LIVE state['pts'] + synthetic centerline/bridge pts (gidx ≥ N).
    # The extras MUST be included so clicking a STITCHED junction —
    # whose MST neighbours can be synthetic gidx ≥ N — can index
    # src[synth] without going out of bounds.  `src_pts` (state['pts']
    # only) is kept for the cold-start segmentation fallback below.
    src_pts = np.asarray(s['pts'], np.float64)
    # Apply any ACCEPTED-separate displacement (topo_mst_pos_override) so
    # THIS click sees the CURRENT, already-separated geometry — NOT raw
    # state['pts'].  Otherwise a 2nd separate (its branch BFS, arm means,
    # split coords AND the pairing/accept geometry) is computed on a region
    # an earlier separate already split, and re-merges it.
    _ov = s.get('topo_mst_pos_override')
    if _ov:
        src_pts = src_pts.copy()
        _oi = np.fromiter(_ov.keys(), np.int64, len(_ov))
        _op = np.array(list(_ov.values()), np.float64)
        _om = (_oi >= 0) & (_oi < len(src_pts))
        src_pts[_oi[_om]] = _op[_om]
    _extras = s.get('topo_mst_extra_pts')
    if _extras is not None and len(_extras) > 0:
        src = np.vstack([src_pts, np.asarray(_extras, np.float64)])
    else:
        src = src_pts
    if len(src) < 4:
        return ('skip', None)
    radius  = float(s.get('curve_radius', 4.0))
    # use cached global MST adj from Topo MST viz; rebuild if missing
    global_adj = s.get('topo_mst_global_adj')
    if global_adj is None:
        min_pts = int(s.get('curve_min_pts', 30))
        print(f'  [topo-branch-explore] (no cached MST adj — rebuilding; '
              f'click "Topo MST viz" once to cache.)')
        seg_labels, K = _compute_segments(src_pts, radius, min_pts)
        global_adj = {}
        for sid in range(K):
            cc_mask = (seg_labels == sid)
            cpts = src_pts[cc_mask]
            cc_global = np.where(cc_mask)[0]
            if len(cpts) < 4:
                continue
            adj, _ = _cc_mst_adj(cpts, radius=radius)
            if adj is None:
                continue
            for i_loc, neigh_loc in adj.items():
                gi = int(cc_global[i_loc])
                slot = global_adj.setdefault(gi, set())
                for n_loc in neigh_loc:
                    slot.add(int(cc_global[n_loc]))
        s['topo_mst_global_adj'] = global_adj
    ne = sorted(global_adj.get(int(start_J), set()))
    if not ne:
        print(f'  [topo-branch-explore] J={start_J}: no MST neighbours')
        return ('skip', None)
    # palette: distinct, doesn't clash with red/green junctions/leaves
    palette = np.array([
        [1.00, 0.50, 0.10],   # orange
        [0.80, 0.20, 0.90],   # magenta
        [0.30, 0.85, 0.20],   # lime
        [0.10, 0.85, 0.85],   # teal
        [0.95, 0.85, 0.20],   # ochre
    ], dtype=np.float64)
    print(f'  [topo-branch-explore] J={start_J}  deg={len(ne)}  '
          f'collecting up to {n_per_branch} pts/branch (BFS, '
          f'junction excluded)')
    branches_collected: list[list[int]] = []
    all_pts:  list[np.ndarray] = []
    all_cols: list[np.ndarray] = []
    for ki, n_first in enumerate(ne):
        col = palette[ki % len(palette)]
        # BFS within the subtree reachable from n_first, with J blocked
        stk: list[int]  = [int(n_first)]
        seen: set[int]  = {int(start_J), int(n_first)}
        collected: list[int] = []
        while stk and len(collected) < n_per_branch:
            u = stk.pop()
            collected.append(u)
            for w in global_adj.get(u, ()):
                if int(w) not in seen:
                    seen.add(int(w))
                    stk.append(int(w))
        branches_collected.append(collected)
        if collected:
            all_pts.append(src[np.asarray(collected)].astype(np.float32))
            all_cols.append(np.tile(col,
                                    (len(collected), 1)).astype(np.float64))
    if ps.has_point_cloud('topo_branch_explorer'):
        ps.remove_point_cloud('topo_branch_explorer')
    if all_pts:
        pc = ps.register_point_cloud(
            'topo_branch_explorer', np.vstack(all_pts))
        pc.add_color_quantity(
            'branch', np.vstack(all_cols), enabled=True)
        pc.set_radius(1.00, relative=False)

    # ── r=<sphere_r> vox sphere at J + each branch's closest-to-sphere
    # point ── radius is `_TOPO_JCT_SPHERE_R`.
    sphere_r = _TOPO_JCT_SPHERE_R
    pos_J = src[int(start_J)]
    # sphere wireframe (3 great-circle rings via _draw_bw_ellipsoid;
    # sigma_par = sigma_perp = r → isotropic)
    _draw_bw_ellipsoid(
        pos_J, np.array([0.0, 0.0, 1.0]),
        sphere_r, sphere_r,
        name='topo_branch_sphere',
        color=(0.45, 0.55, 0.95),     # soft blue
        radius=0.16)
    # for each branch, find the collected point whose distance to J is
    # closest to sphere_r (i.e., the point that sits on / just inside /
    # just outside the sphere surface).  Emphasize as a bigger ball.
    closest_pts:  list[np.ndarray] = []
    closest_cols: list[np.ndarray] = []
    closest_idx:  list[int]        = []
    for ki, collected in enumerate(branches_collected):
        if not collected:
            continue
        positions   = src[np.asarray(collected)]
        dists_to_J  = np.linalg.norm(positions - pos_J, axis=1)
        best_local  = int(np.argmin(np.abs(dists_to_J - sphere_r)))
        best_global = int(collected[best_local])
        closest_pts.append(src[best_global])
        closest_cols.append(palette[ki % len(palette)])
        closest_idx.append(best_global)
    if ps.has_point_cloud('topo_branch_closest'):
        ps.remove_point_cloud('topo_branch_closest')
    if closest_pts:
        pc2 = ps.register_point_cloud(
            'topo_branch_closest',
            np.asarray(closest_pts, np.float32))
        pc2.set_radius(2.40, relative=False)             # bigger than 0.50
        pc2.add_color_quantity(
            'branch',
            np.asarray(closest_cols, np.float64), enabled=True)
        pc2.add_scalar_quantity(
            'global_idx',
            np.asarray(closest_idx, np.float64), enabled=False)

    # ── Pairwise angle analysis: 3 vectors J→closest_i give 3 angles.
    # In a ">" merge the two pre-merge arms leave J in nearly the same
    # direction while the merged trunk leaves opposite to both.  So the
    # SMALLEST angle is between the 2 arms, and the remaining vector is the
    # trunk.  The 2 arms get a white halo ball so the user sees which two
    # would be un-merged.
    if ps.has_point_cloud('topo_branch_premerge'):
        ps.remove_point_cloud('topo_branch_premerge')
    if len(closest_pts) == 3:
        vecs = np.stack([np.asarray(p, np.float64) - pos_J
                         for p in closest_pts])     # (3, 3)
        ns   = np.linalg.norm(vecs, axis=1)
        if (ns > 1e-9).all():
            unit_vecs = vecs / ns[:, None]
            pairs: list[tuple[int, int, float]] = []
            for i in range(3):
                for j in range(i + 1, 3):
                    cs = float(np.clip(np.dot(unit_vecs[i], unit_vecs[j]),
                                       -1.0, 1.0))
                    pairs.append((i, j, float(np.degrees(np.arccos(cs)))))
            pairs_sorted = sorted(pairs, key=lambda x: x[2])
            print(f'    pairwise angles (J→closest_i):')
            for i, j, a in pairs:
                print(f'      branch[{i}]–branch[{j}]: {a:6.2f}°')
            # ── STEM/ARM decision: ANGLE + DENSITY (combined) ─────────
            # Pure angle is fragile on a near-symmetric Y: the two smallest angles
            # can be degrees apart, a coin flip.  A >-----< stem is the MERGED
            # section, so inside the probe sphere it carries ~2x the density of a
            # single-yarn arm, and a 2x margin beats a 3° one.  Both are blended as
            # normalised distributions, so a flat (ambiguous) angle distribution
            # cannot sway the pick and density decides; w_dens=0 is angle-only.
            # Density runs in click AND batch since it drives the decision.
            # The KDTree is cached on the state['pts'] OBJECT (we hold a ref, so no
            # id() recycling into a false hit); an MS step reassigns pts → rebuild.
            if s.get('_density_tree_pts') is not s['pts']:
                s['_density_tree'] = cKDTree(src_pts)
                s['_density_tree_pts'] = s['pts']
            _dtree = s['_density_tree']
            _ball = _dtree.query_ball_point(pos_J, r=sphere_r)
            dens = np.zeros(3, np.float64)
            _dpos = None
            _dasg = None
            if _ball:
                _ib = np.asarray(_ball, np.int64)
                _vv = src_pts[_ib] - pos_J
                _nv = np.linalg.norm(_vv, axis=1)
                _ok = _nv > 1e-6
                if _ok.any():
                    _u = _vv[_ok] / _nv[_ok, None]
                    _asg = np.argmax(_u @ unit_vecs.T, axis=1)
                    for _k in range(3):
                        dens[_k] = float((_asg == _k).sum())
                    _dpos = src_pts[_ib][_ok].astype(np.float32)
                    _dasg = _asg
            # angle term: candidate stem i favoured when the OTHER two
            # bundle (small angle between them) → high (180 − that angle)
            _ang = {(i, j): a for (i, j, a) in pairs}
            angle_other = np.array([
                _ang[tuple(sorted(x for x in range(3) if x != c))]
                for c in range(3)], np.float64)
            A = 180.0 - angle_other
            A = A / A.sum() if A.sum() > 1e-9 else np.full(3, 1.0 / 3.0)
            D = (dens / dens.sum() if dens.sum() > 0
                 else np.full(3, 1.0 / 3.0))
            W_DENS = _TOPO_STEM_DENSITY_WEIGHT
            C = (1.0 - W_DENS) * A + W_DENS * D
            trunk_local = int(np.argmax(C))
            premerge_locals = [v for v in range(3) if v != trunk_local]
            _ang_pick = int(np.argmax(A))
            print(f'    stem/arm (ANGLE+DENSITY, w_dens={W_DENS:.2f}):')
            print(f'      angle   A = [{A[0]:.2f} {A[1]:.2f} {A[2]:.2f}] '
                  f'(arms-bundle; pure-angle pick=branch[{_ang_pick}])')
            print(f'      density D = [{D[0]:.2f} {D[1]:.2f} {D[2]:.2f}] '
                  f'(sphere r={sphere_r:.1f} counts '
                  f'{int(dens[0])}/{int(dens[1])}/{int(dens[2])}; '
                  f'densest=branch[{int(np.argmax(dens))}])')
            print(f'      combined C = [{C[0]:.2f} {C[1]:.2f} {C[2]:.2f}] '
                  f'→ trunk=branch[{trunk_local}], arms=branch'
                  f'[{premerge_locals[0]}]+branch[{premerge_locals[1]}]')
            # ── viz (click only): white halo on the 2 arms + the
            # density-bucketed in-sphere points coloured by branch.
            if not auto:
                if ps.has_point_cloud('topo_branch_premerge'):
                    ps.remove_point_cloud('topo_branch_premerge')
                halo_pts = np.asarray(
                    [closest_pts[k] for k in premerge_locals], np.float32)
                pc3 = ps.register_point_cloud(
                    'topo_branch_premerge', halo_pts)
                pc3.set_radius(4.40, relative=False)
                pc3.set_color((1.0, 1.0, 1.0))            # white halo
                try:
                    pc3.set_transparency(0.55)
                except Exception:
                    pass                              # older polyscope
                pc3.add_scalar_quantity(
                    'branch_local_idx',
                    np.asarray(premerge_locals, np.float64),
                    enabled=False)
                if ps.has_point_cloud('topo_branch_density'):
                    ps.remove_point_cloud('topo_branch_density')
                if _dpos is not None and len(_dpos) > 0:
                    pcd = ps.register_point_cloud(
                        'topo_branch_density', _dpos)
                    pcd.set_radius(1.30, relative=False)
                    pcd.add_color_quantity(
                        'branch_dir',
                        palette[_dasg % len(palette)].astype(np.float64),
                        enabled=True)

            # ── Arm-connect probe.  From the SHORTER arm's MST
            # subtree (J blocked), BFS to find its deepest leaf
            # (capped at depth MAX_DEPTH — exceed → abort, no
            # further ops).  Then list points within 5vox of that
            # leaf that lie on the LONGER arm's subtree (= bridge
            # candidates for the >-----< merge).
            for _nm in ('topo_branch_bridge_leaf',
                        'topo_branch_bridge_cands'):
                if ps.has_point_cloud(_nm):
                    ps.remove_point_cloud(_nm)

            def _arm_subtree(first_ne_local: int) -> set:
                stk  = [int(first_ne_local)]
                seen = {int(start_J), int(first_ne_local)}
                while stk:
                    u = stk.pop()
                    for w in global_adj.get(u, ()):
                        iw = int(w)
                        if iw not in seen:
                            seen.add(iw)
                            stk.append(iw)
                seen.discard(int(start_J))
                return seen

            ne_arm0 = int(ne[premerge_locals[0]])
            ne_arm1 = int(ne[premerge_locals[1]])
            sub0    = _arm_subtree(ne_arm0)
            sub1    = _arm_subtree(ne_arm1)
            if len(sub0) <= len(sub1):
                short_ne,  long_ne  = ne_arm0, ne_arm1
                short_sub, long_sub = sub0,    sub1
                short_li,  long_li  = (premerge_locals[0],
                                        premerge_locals[1])
            else:
                short_ne,  long_ne  = ne_arm1, ne_arm0
                short_sub, long_sub = sub1,    sub0
                short_li,  long_li  = (premerge_locals[1],
                                        premerge_locals[0])
            print(f'  [arm-connect] shorter arm = branch[{short_li}] '
                  f'(subtree {len(short_sub)} pts), '
                  f'longer arm = branch[{long_li}] '
                  f'(subtree {len(long_sub)} pts)')

            MAX_DEPTH = _TOPO_BRANCH_MAX_DEPTH
            parent_bfs: dict[int, int] = {int(start_J): -1,
                                          short_ne: int(start_J)}
            depth_bfs:  dict[int, int] = {int(start_J): 0,
                                          short_ne: 1}
            queue = [short_ne]
            deepest_node, deepest_depth = short_ne, 1
            exceeded = False
            qi = 0
            while qi < len(queue):
                u = queue[qi]; qi += 1
                # SHORT-arm "is it actually a long/real arm?" uses the
                # smaller _TOPO_ARM_LONG_DEPTH (not the 1000 trunk-probe
                # cap): a real ~200-deep arm exceeds → go to TRUNK-probe/
                # separate instead of being auto-cut.
                if depth_bfs[u] > _TOPO_ARM_LONG_DEPTH:
                    exceeded = True
                    break
                for w in global_adj.get(u, ()):
                    iw = int(w)
                    if iw not in parent_bfs and iw != int(start_J):
                        parent_bfs[iw] = u
                        depth_bfs[iw]  = depth_bfs[u] + 1
                        queue.append(iw)
                        if depth_bfs[iw] > deepest_depth:
                            deepest_depth = depth_bfs[iw]
                            deepest_node  = iw
            if exceeded:
                print(f'  [arm-connect] shorter arm BFS depth '
                      f'exceeded {_TOPO_ARM_LONG_DEPTH} (real arm, not a '
                      f'short dead-end) → TRUNK-probe '
                      f'(search next real junction on '
                      f'branch[{trunk_local}])')
                # ── Trunk-probe.  Neither arm bridges within
                # MAX_DEPTH (both arms long = a classic >-----<
                # whose merge is not local).  Walk the TRUNK
                # direction; the first red junction found is the
                # other end of the shared trunk, and the MST path
                # J → J2 (cyan-green) is where the two yarns run
                # merged as one chain.  Nothing is stitched here.
                jct_info = s.get('topo_mst_jct_info') or []
                real_jct_set = {int(d['gidx']) for d in jct_info
                                if int(d['gidx']) != int(start_J)}
                ne_trunk = int(ne[trunk_local])
                t_parent: dict[int, int] = {int(start_J): -1,
                                            ne_trunk: int(start_J)}
                t_depth:  dict[int, int] = {int(start_J): 0,
                                            ne_trunk: 1}
                t_queue  = [ne_trunk]
                t_qi     = 0
                next_jct: int | None = None
                while t_qi < len(t_queue):
                    u = t_queue[t_qi]; t_qi += 1
                    if t_depth[u] > MAX_DEPTH:
                        break
                    if u in real_jct_set:
                        next_jct = u
                        break
                    for w in global_adj.get(u, ()):
                        iw = int(w)
                        if iw not in t_parent and iw != int(start_J):
                            t_parent[iw] = u
                            t_depth[iw]  = t_depth[u] + 1
                            t_queue.append(iw)
                if ps.has_curve_network('topo_branch_shared_trunk'):
                    ps.remove_curve_network('topo_branch_shared_trunk')
                if ps.has_curve_network('topo_branch_arms'):
                    ps.remove_curve_network('topo_branch_arms')
                if ps.has_point_cloud('topo_branch_next_jct'):
                    ps.remove_point_cloud('topo_branch_next_jct')
                if next_jct is None:
                    # No next junction on the trunk within MAX_DEPTH,
                    # so instead of aborting, look for a DISCONNECTED
                    # fragment that may belong here: over every OTHER
                    # MST component take its diameter tips (the
                    # longest-path constraint rules out short stubs),
                    # keep the tip closest to J, and attach it to the
                    # nearest trunk node within MAX_DEPTH hops.
                    # No other component → manual intervention.
                    for _nm in ('topo_branch_loose_leaf',
                                'topo_branch_stem_attach'):
                        if ps.has_point_cloud(_nm):
                            ps.remove_point_cloud(_nm)
                    if ps.has_curve_network('topo_branch_attach_link'):
                        ps.remove_curve_network(
                            'topo_branch_attach_link')
                    pos_J = src[int(start_J)]
                    # ---- connected components of global_adj ----
                    cc_of: dict[int, int] = {}
                    cc_members: list[list[int]] = []
                    for seed in global_adj:
                        if seed in cc_of:
                            continue
                        cidx = len(cc_members)
                        comp: list[int] = []
                        cstk = [int(seed)]
                        cc_of[int(seed)] = cidx
                        while cstk:
                            cu = cstk.pop()
                            comp.append(cu)
                            for cv in global_adj.get(cu, ()):
                                icv = int(cv)
                                if icv not in cc_of:
                                    cc_of[icv] = cidx
                                    cstk.append(icv)
                        cc_members.append(comp)
                    J_cc = cc_of.get(int(start_J), -1)

                    def _bfs_farthest(start0: int) -> tuple[int, int]:
                        """Hop BFS from start0 → (farthest_node, hops)."""
                        bdist = {int(start0): 0}
                        bq = [int(start0)]; bqi = 0
                        bfar, bfar_d = int(start0), 0
                        while bqi < len(bq):
                            bu = bq[bqi]; bqi += 1
                            for bv in global_adj.get(bu, ()):
                                ibv = int(bv)
                                if ibv not in bdist:
                                    bdist[ibv] = bdist[bu] + 1
                                    bq.append(ibv)
                                    if bdist[ibv] > bfar_d:
                                        bfar_d = bdist[ibv]; bfar = ibv
                        return bfar, bfar_d

                    # Candidates = the diameter (longest-path) tips of
                    # EVERY other CC (2 per CC).  We pick the one
                    # closest to J — NOT "the CC with the largest
                    # diameter"; any disconnected MST's diameter tip
                    # qualifies, the longest-path constraint just rules
                    # out short-stub leaves.
                    cand_eps: list[int] = []
                    n_other = 0
                    for cidx, comp in enumerate(cc_members):
                        if cidx == J_cc or len(comp) < 2:
                            continue
                        n_other += 1
                        a0, _ = _bfs_farthest(comp[0])
                        b0, _ = _bfs_farthest(a0)
                        cand_eps.append(a0)
                        cand_eps.append(b0)

                    trunk_nodes = [g for g in t_parent
                                   if g != int(start_J)]
                    if not cand_eps or not trunk_nodes:
                        print(f'  [trunk-probe] no next junction within '
                              f'{MAX_DEPTH} hops AND search failed '
                              f'(other-MST diameter tips='
                              f'{len(cand_eps)}, trunk nodes='
                              f'{len(trunk_nodes)}) → **WARNING: this '
                              f'structure needs MANUAL intervention**')
                        return ('skip', None)
                    else:
                        cand_arr = np.asarray(cand_eps, np.int64)
                        d_cand_J = np.linalg.norm(
                            src[cand_arr] - pos_J, axis=1)
                        L_iso = int(cand_arr[int(np.argmin(d_cand_J))])
                        L_iso_pos = src[L_iso]
                        tn_arr = np.asarray(trunk_nodes, np.int64)
                        d_to_leaf = np.linalg.norm(
                            src[tn_arr] - L_iso_pos, axis=1)
                        P_near = int(tn_arr[int(np.argmin(d_to_leaf))])
                        min_d = float(d_to_leaf.min())
                        ATTACH_MAX_VOX = 50.0
                        if min_d > ATTACH_MAX_VOX:
                            # The nearest stem point is still >50vox
                            # from the loose tip → no plausible
                            # attachment (the stem never approaches it;
                            # closest point degenerates to ~J).  Kill.
                            print(f'  [trunk-probe] no next junction → '
                                  f'SEARCH: nearest other-MST diameter '
                                  f'tip = gidx{L_iso} '
                                  f'(|tip−J|='
                                  f'{float(d_cand_J.min()):.2f}vox) but '
                                  f'nearest stem point gidx{P_near} is '
                                  f'{min_d:.2f}vox away '
                                  f'(> {ATTACH_MAX_VOX:.0f}vox cutoff) → '
                                  f'**WARNING: this structure needs '
                                  f'MANUAL intervention**')
                            if not auto:
                                # Rejected, but still show the geometry so
                                # it's inspectable: J (white), the nearest
                                # stem point P_near (red), the rejected
                                # candidate tip L_iso (cyan), and the
                                # J→tip gap (red line = too far).
                                for _nm in (
                                        'topo_branch_loose_leaf',
                                        'topo_branch_stem_attach',
                                        'topo_branch_attach_pts',
                                        'topo_branch_reject_jct'):
                                    if ps.has_point_cloud(_nm):
                                        ps.remove_point_cloud(_nm)
                                if ps.has_curve_network(
                                        'topo_branch_attach_link'):
                                    ps.remove_curve_network(
                                        'topo_branch_attach_link')
                                pcj = ps.register_point_cloud(
                                    'topo_branch_reject_jct',
                                    pos_J.astype(
                                        np.float32).reshape(1, 3))
                                pcj.set_radius(3.40, relative=False)
                                pcj.set_color((1.00, 1.00, 1.00))  # J
                                pcp = ps.register_point_cloud(
                                    'topo_branch_stem_attach',
                                    src[P_near].astype(
                                        np.float32).reshape(1, 3))
                                pcp.set_radius(3.00, relative=False)
                                pcp.set_color((1.00, 0.20, 0.20))  # P_near
                                pcp.add_scalar_quantity(
                                    'global_idx',
                                    np.asarray([float(P_near)],
                                               np.float64),
                                    enabled=False)
                                pct = ps.register_point_cloud(
                                    'topo_branch_loose_leaf',
                                    L_iso_pos.astype(
                                        np.float32).reshape(1, 3))
                                pct.set_radius(3.20, relative=False)
                                pct.set_color((0.00, 1.00, 0.70))  # tip
                                pct.add_scalar_quantity(
                                    'global_idx',
                                    np.asarray([float(L_iso)],
                                               np.float64),
                                    enabled=False)
                                link = np.stack(
                                    [pos_J, L_iso_pos]).astype(np.float32)
                                cnl = ps.register_curve_network(
                                    'topo_branch_attach_link', link,
                                    np.asarray([(0, 1)], np.int32))
                                cnl.set_radius(1.00, relative=False)
                                cnl.set_color((1.00, 0.20, 0.20))  # gap
                            return ('skip', None)
                        else:
                            # Classified as SEARCH.  In batch mode, only
                            # actually stitch during the 'search' phase;
                            # otherwise just report the classification.
                            if auto and apply_action != 'search':
                                return ('search', None)
                            print(f'  [trunk-probe] no next junction → '
                                  f'SEARCH: nearest other-MST diameter '
                                  f'tip to J = gidx{L_iso} '
                                  f'(|tip−J|='
                                  f'{float(d_cand_J.min()):.2f}vox, '
                                  f'from {len(cand_eps)} tips across '
                                  f'{n_other} other MSTs); nearest stem '
                                  f'point to it = gidx{P_near} '
                                  f'(|stem−tip|={min_d:.2f}vox over '
                                  f'{len(trunk_nodes)} trunk nodes)')
                            # ── Back off the attach point toward J:
                            # P_near is the CLOSEST stem point, but
                            # connecting there is too abrupt.  Walk back
                            # toward J along the stem (via t_parent) by
                            # 35% of P_near's hop-distance to J → a
                            # shallower, smoother approach angle.
                            depth_near = int(t_depth.get(P_near, 0))
                            back_steps = int(round(0.35 * depth_near))
                            P_attach = P_near
                            n_back = 0
                            for _ in range(back_steps):
                                par = t_parent.get(P_attach, -1)
                                if par < 0 or par == int(start_J):
                                    break
                                P_attach = par
                                n_back += 1
                            P_attach_pos = src[P_attach]
                            # ── STITCH: bridge L_iso → P_attach.  Sample
                            # the segment at ~curve_radius spacing →
                            # interior points; add them as SYNTHETIC MST
                            # nodes (state['topo_mst_extra_pts'], gidx ≥
                            # N) + chain edges L_iso ↔ s0 ↔ … ↔ sk ↔
                            # P_attach into global_adj, so the
                            # disconnected fragment fuses onto the stem
                            # in topo_mst (and Fit curves).  state['pts']
                            # is NOT touched.
                            seg_dist = float(np.linalg.norm(
                                P_attach_pos - L_iso_pos))
                            # Dense bridge sampling: 0.1vox spacing
                            # (= 10 synthetic pts per vox).
                            step = 0.1
                            n_int = max(1, int(round(seg_dist / step)) - 1)
                            # ── Tangent anchors for a SMOOTH (centripetal
                            # Catmull-Rom) bridge rather than a straight line.
                            # p_prev = L_iso's trailing MST neighbour, so the
                            # curve leaves the fragment tangentially; p_next =
                            # the stem neighbour continuing FORWARD, so it
                            # merges without doubling back.
                            chord = P_attach_pos - L_iso_pos
                            nb_L = [int(x) for x
                                    in global_adj.get(int(L_iso), ())]
                            if nb_L:
                                aL = np.asarray(nb_L, np.int64)
                                p_prev = src[int(aL[int(np.argmin(
                                    (src[aL] - L_iso_pos) @ chord))])]
                            else:
                                p_prev = L_iso_pos - chord
                            nb_S = [int(x) for x
                                    in global_adj.get(int(P_attach), ())]
                            if nb_S:
                                aS = np.asarray(nb_S, np.int64)
                                p_next = src[int(aS[int(np.argmax(
                                    (src[aS] - P_attach_pos) @ chord))])]
                            else:
                                p_next = P_attach_pos + chord
                            sampled = _smooth_bridge_pts(
                                p_prev, L_iso_pos, P_attach_pos,
                                p_next, n_int).astype(np.float64)
                            if not auto:
                                _topo_stitch_snapshot(
                                    f'attach loose-tip gidx{L_iso} → stem '
                                    f'gidx{P_attach} ({len(sampled)} synth '
                                    f'pts)')
                            n_orig_st = len(state['pts'])
                            existing_extras = s.get('topo_mst_extra_pts')
                            if (existing_extras is None
                                    or len(existing_extras) == 0):
                                existing_extras = np.zeros(
                                    (0, 3), dtype=np.float64)
                            n_exist = len(existing_extras)
                            synth_start = n_orig_st + n_exist
                            synth_list = list(range(
                                synth_start, synth_start + len(sampled)))
                            s['topo_mst_extra_pts'] = np.vstack(
                                [existing_extras, sampled])
                            attach_chain = ([int(L_iso)] + synth_list
                                            + [int(P_attach)])
                            for ic in range(len(attach_chain) - 1):
                                uc = int(attach_chain[ic])
                                vc = int(attach_chain[ic + 1])
                                global_adj.setdefault(
                                    uc, set()).add(vc)
                                global_adj.setdefault(
                                    vc, set()).add(uc)
                            print(f'  [trunk-probe] STITCH: P_near='
                                  f'gidx{P_near} backed off {n_back}/'
                                  f'{back_steps} steps (35% of depth '
                                  f'{depth_near}) → attach @ gidx'
                                  f'{P_attach}; bridged gidx{L_iso} → '
                                  f'gidx{P_attach} ({seg_dist:.1f}vox) '
                                  f'with {len(sampled)} synthetic pts '
                                  f'(gidx {synth_start}..'
                                  f'{synth_start + len(sampled) - 1}), '
                                  f'{len(attach_chain) - 1} new edges; '
                                  f'refreshing topo_mst')
                            if auto:
                                return ('search', None)
                            try:
                                _topo_mst_viz(use_cached_adj=True)
                            except Exception as _ex:
                                print(f'  [trunk-probe] re-render '
                                      f'warn: {_ex}')
                            # CYAN ball = the disconnected fragment's
                            # tip (the OTHER MST's diameter endpoint)
                            pcl = ps.register_point_cloud(
                                'topo_branch_loose_leaf',
                                L_iso_pos.astype(
                                    np.float32).reshape(1, 3))
                            pcl.set_radius(3.20, relative=False)
                            pcl.set_color((0.00, 1.00, 0.70))
                            # RED ball = the stem attach point (backed
                            # off toward J from P_near)
                            pcs = ps.register_point_cloud(
                                'topo_branch_stem_attach',
                                P_attach_pos.astype(
                                    np.float32).reshape(1, 3))
                            pcs.set_radius(3.20, relative=False)
                            pcs.set_color((1.00, 0.20, 0.20))
                            # yellow sampled points along the bridge
                            pca = ps.register_point_cloud(
                                'topo_branch_attach_pts',
                                sampled.astype(np.float32))
                            pca.set_radius(1.60, relative=False)
                            pca.set_color((1.00, 0.95, 0.10))
                            # yellow link = the smooth polyline through
                            # L_iso → sampled → P_attach (matches the
                            # actual stitched curve, not a straight chord)
                            link_nodes = np.vstack(
                                [L_iso_pos[None, :], sampled,
                                 P_attach_pos[None, :]]).astype(np.float32)
                            link_edges = np.asarray(
                                [(i, i + 1)
                                 for i in range(len(link_nodes) - 1)],
                                np.int32)
                            cnl = ps.register_curve_network(
                                'topo_branch_attach_link',
                                link_nodes, link_edges)
                            cnl.set_radius(1.20, relative=False)
                            cnl.set_color((1.00, 0.95, 0.10))
                else:
                    # SEPARATE case (next real junction found on the
                    # trunk → classic >-----<).  Never mutates; in batch
                    # mode it returns the split-apart point sets so the
                    # master can build the 2^n possibility layers.
                    _sep_payload = None
                    # During the deterministic phases (apply_action is a
                    # search/loop/cut string) we only need the LABEL —
                    # skip the expensive uncapped full-arm BFS + PCA that
                    # build the payload (the master collects payloads in
                    # a dedicated final pass with apply_action=None).
                    if auto and apply_action is not None:
                        return ('separate', None)
                    # reconstruct J → ... → J2 chain via parent dict
                    chain: list[int] = [next_jct]
                    cur = next_jct
                    while t_parent[cur] >= 0:
                        cur = t_parent[cur]
                        chain.append(cur)
                    chain.reverse()              # J → ... → J2
                    print(f'  [trunk-probe] next real junction = '
                          f'gidx{next_jct} at {t_depth[next_jct]} '
                          f'hops on branch[{trunk_local}]; shared '
                          f'trunk = {len(chain)} nodes')
                    chain_pos   = src[np.asarray(chain)]
                    chain_edges = np.asarray(
                        [(i, i + 1) for i in range(len(chain) - 1)],
                        np.int32)
                    if len(chain_edges) > 0:
                        cn_st = ps.register_curve_network(
                            'topo_branch_shared_trunk',
                            chain_pos.astype(np.float32),
                            chain_edges)
                        cn_st.set_radius(1.30, relative=False)
                        cn_st.set_color((0.00, 1.00, 0.70))  # cyan-green
                    pcn = ps.register_point_cloud(
                        'topo_branch_next_jct',
                        src[next_jct].astype(np.float32).reshape(1, 3))
                    pcn.set_radius(3.20, relative=False)
                    pcn.set_color((0.00, 1.00, 0.70))
                    pcn.add_scalar_quantity(
                        'global_idx',
                        np.asarray([float(next_jct)], np.float64),
                        enabled=False)
                    # ── Arms.  From each of J's 2 non-trunk MST
                    # neighbours (and J2's 2 non-trunk neighbours), BFS
                    # the subtree with the respective junction blocked.
                    # Render the union (plus J↔arm-start / J2↔arm-start
                    # attach edges) as a SINGLE orange curve_network so
                    # the full >-----< shows as cyan-green trunk +
                    # orange arms.
                    chain_set = {int(x) for x in chain}
                    # arm cap is 1/10 of MAX_DEPTH — arms are usually
                    # much shorter than the trunk that connects two
                    # junctions, so 100 hops is plenty to show the
                    # local >-----< structure without spilling into
                    # neighbouring CC topology via a long noisy arm.
                    ARM_MAX_DEPTH = MAX_DEPTH // 10
                    def _arm_sub(start_n: int,
                                 block_jct: int,
                                 cap: int = ARM_MAX_DEPTH
                                 ) -> tuple[list[int], bool]:
                        """BFS-collect arm subtree from `start_n` with
                        `block_jct` blocked, capped at `cap` hops from the
                        junction (start_n itself is at depth 1; default cap
                        = ARM_MAX_DEPTH for the viz/pairing, a smaller cap
                        for the 'D' delete).  Returns (nodes, cap_hit) —
                        cap_hit True iff a node at depth `cap` was
                        truncated."""
                        out  = [int(start_n)]
                        seen = {int(block_jct), int(start_n)}
                        # (node, depth-from-junction); start_n = 1
                        queue = [(int(start_n), 1)]
                        qi = 0
                        cap_hit = False
                        while qi < len(queue):
                            u, d = queue[qi]; qi += 1
                            if d >= cap:
                                cap_hit = True
                                continue       # don't expand past cap
                            for w in global_adj.get(u, ()):
                                iw = int(w)
                                if iw in seen:
                                    continue
                                seen.add(iw)
                                out.append(iw)
                                queue.append((iw, d + 1))
                        return out, cap_hit
                    _NREAL = int(len(s['pts']))   # gidx ≥ this == synthetic

                    def _anchor_break_to_real(b: int, exclude: set) -> int:
                        """Map a break node to a REAL gidx: a synthetic
                        search/loop node is wiped by the D-delete recompute,
                        so walk that arm away from the junction to the first
                        real node and break THERE instead."""
                        if int(b) < _NREAL:
                            return int(b)
                        seen = set(exclude); seen.add(int(b))
                        frontier = [int(b)]; qi = 0
                        while qi < len(frontier):
                            u = frontier[qi]; qi += 1
                            for w in global_adj.get(u, ()):
                                iw = int(w)
                                if iw in seen:
                                    continue
                                seen.add(iw)
                                if iw < _NREAL:
                                    return iw
                                frontier.append(iw)
                        return int(b)
                    arm_nodes: set[int] = set()
                    del_arm_nodes: set[int] = set()   # 'D' delete: shorter cap
                    arm_breaks: list[dict] = []       # per-arm DELETE break info
                    arm_subs:  list[list[int]] = []   # per-arm node lists
                    arm_starts: list[tuple[int, int]] = []  # (start_ne, J_or_J2)
                    arm_count = 0
                    for sJ in (int(start_J), int(next_jct)):
                        for w in global_adj.get(sJ, ()):
                            iw = int(w)
                            if iw in chain_set:
                                continue                  # on trunk
                            sub, cap_hit = _arm_sub(iw, sJ)
                            arm_count += 1
                            arm_nodes |= set(sub)
                            arm_subs.append(sub)
                            arm_starts.append((iw, int(sJ)))
                            # delete-only: first _TOPO_DELETE_ARM_HOPS hops
                            _dsub, _ = _arm_sub(iw, sJ,
                                                cap=_TOPO_DELETE_ARM_HOPS)
                            del_arm_nodes |= set(_dsub)
                            _bsub, _ = _arm_sub(iw, sJ,
                                                cap=_TOPO_DELETE_ARM_HOPS + 1)
                            _brk0 = sorted(set(_bsub) - set(_dsub))
                            _excl = set(_dsub) | {int(sJ)}
                            _brk = sorted({_anchor_break_to_real(b, _excl)
                                           for b in _brk0})
                            _bpos = [[float(c) for c in src[int(g)]]
                                     for g in _brk if 0 <= int(g) < len(src)]
                            arm_breaks.append({
                                'arm':        arm_count - 1,   # 0..3
                                'junction':   int(sJ),
                                'side':       ('J0' if sJ == int(start_J)
                                               else 'J1'),
                                'arm_start':  int(iw),
                                'break_gidx': [int(x) for x in _brk],
                                'break_pos':  _bpos,
                                'arm_len':    int(len(sub)),
                            })
                            tag = ('  [TRUNCATED at ARM_MAX_DEPTH='
                                   f'{ARM_MAX_DEPTH}]'
                                   if cap_hit else '')
                            print(f'  [trunk-probe] arm[{arm_count}] '
                                  f'(off gidx{sJ}, ne=gidx{iw}): '
                                  f'{len(sub)} nodes{tag}')
                    if arm_nodes:
                        viz_nodes = sorted(
                            arm_nodes | {int(start_J), int(next_jct)})
                        viz_g2l = {g: i for i, g in enumerate(viz_nodes)}
                        viz_pos = src[np.asarray(viz_nodes)]
                        chain_e_set: set[tuple[int, int]] = set()
                        for ci in range(len(chain) - 1):
                            a, b = int(chain[ci]), int(chain[ci + 1])
                            chain_e_set.add((min(a, b), max(a, b)))
                        seen_e: set[tuple[int, int]] = set()
                        arm_edges_loc: list[tuple[int, int]] = []
                        for u in viz_nodes:
                            for v in global_adj.get(u, ()):
                                iv = int(v)
                                if iv not in viz_g2l:
                                    continue
                                a, b = (u, iv) if u < iv else (iv, u)
                                if (a, b) in seen_e:
                                    continue
                                if (a, b) in chain_e_set:
                                    continue       # belongs to trunk
                                seen_e.add((a, b))
                                arm_edges_loc.append(
                                    (viz_g2l[a], viz_g2l[b]))
                        if arm_edges_loc:
                            cn_arm = ps.register_curve_network(
                                'topo_branch_arms',
                                viz_pos.astype(np.float32),
                                np.asarray(arm_edges_loc, np.int32))
                            cn_arm.set_radius(1.10, relative=False)
                            cn_arm.set_color((1.0, 0.55, 0.10))  # fallback
                            # 4 DISTINCT colours, one per arm (arm0..3), so
                            # the >-----< arms are visually separable; the
                            # two shared junctions stay gray.
                            _arm_pal = np.array([
                                [0.90, 0.20, 0.20],   # arm0 red
                                [0.20, 0.55, 1.00],   # arm1 blue
                                [1.00, 0.85, 0.10],   # arm2 yellow
                                [0.20, 0.85, 0.55],   # arm3 teal
                            ], np.float32)
                            _node_col = np.tile(
                                np.array([0.80, 0.80, 0.80], np.float32),
                                (len(viz_nodes), 1))      # junctions gray
                            for _ai, _sub in enumerate(arm_subs):
                                _c = (_arm_pal[_ai] if _ai < 4
                                      else np.array([0.6, 0.6, 0.6], np.float32))
                                for _g in _sub:
                                    _li = viz_g2l.get(int(_g))
                                    if _li is not None:
                                        _node_col[_li] = _c
                            cn_arm.add_color_quantity(
                                'arm', _node_col, defined_on='nodes',
                                enabled=True)
                            print(f'  [trunk-probe] >-----< arms: '
                                  f'{arm_count} arms / '
                                  f'{len(arm_nodes)} nodes / '
                                  f'{len(arm_edges_loc)} edges '
                                  f'(arm0=red arm1=blue arm2=yellow '
                                  f'arm3=teal, junctions gray)')
                    # ── Yarn-pairing viz: both candidate topologies are
                    # offered, the user decides visually.
                    # Pairing A:  arm[0]+arm[2] = yarn-α, arm[1]+arm[3] = yarn-β
                    # Pairing B:  arm[0]+arm[3] = yarn-α, arm[1]+arm[2] = yarn-β
                    # (arm[0..1] are J's non-trunk neighbours, arm[2..3]
                    #  are J2's; B is the "cross" of A's J2-side pair.)
                    # Same yarn = same colour; the trunk stays NEUTRAL
                    # gray because yarn identity inside it is exactly
                    # the ambiguity that cannot be resolved here.
                    for _nm in ('topo_branch_pairing_0',
                                'topo_branch_pairing_1',
                                'topo_branch_pairing_2',
                                'topo_branch_pairing_3',
                                'topo_branch_pairing_A',
                                'topo_branch_pairing_B'):
                        if ps.has_point_cloud(_nm):
                            ps.remove_point_cloud(_nm)
                    if len(arm_subs) == 4:
                        def _full_arm(start_n: int,
                                      block_jct: int) -> list[int]:
                            """BFS full arm subtree (NO cap)."""
                            out  = [int(start_n)]
                            seen = {int(block_jct), int(start_n)}
                            stk  = [int(start_n)]
                            while stk:
                                u = stk.pop()
                                for w in global_adj.get(u, ()):
                                    iw = int(w)
                                    if iw in seen:
                                        continue
                                    seen.add(iw)
                                    out.append(iw)
                                    stk.append(iw)
                            return out
                        full_arms = [
                            _full_arm(start_ne, block_jct)
                            for (start_ne, block_jct) in arm_starts]
                        n_arms_pts = [len(a) for a in full_arms]

                        # ── Trunk split-apart via PCA-2 displacement on
                        # MST nodes (not fitted curves): PCA on the 4 arm
                        # means gives the lateral axis; each trunk node is
                        # projected onto it and split at the median, using
                        # PTS_ORIG because it still holds the pre-MS lateral
                        # separation that src has collapsed.  The two halves
                        # are then displaced ±pca2 × DISPLACE_VOX so the one
                        # chain reads as two parallel ones (red = up, cyan =
                        # down, same in both pairings; only the arm colours
                        # differ between A/B).
                        arm_means_capped = np.stack(
                            [src[np.asarray(a)].mean(0)
                             for a in arm_subs])
                        center_pca = arm_means_capped.mean(0)
                        centered = arm_means_capped - center_pca
                        _u, _sv, vt = np.linalg.svd(
                            centered, full_matrices=False)
                        pca2 = vt[1]
                        # Sign convention: PCA-2 has positive Y so
                        # "up" is consistent across clicks.
                        if pca2[1] < 0:
                            pca2 = -pca2
                        pos_o = np.asarray(pts_orig, np.float64)
                        trunk_arr = np.asarray(chain, np.int64)
                        # pts_orig only spans the N original gidx.  After
                        # a search-stitch the trunk can include SYNTHETIC
                        # gidx ≥ N (no pre-MS position) — fall back to
                        # their `src` position for those so the median
                        # projection doesn't index out of bounds.
                        n_orig_g = len(pos_o)
                        trunk_o = np.empty((len(trunk_arr), 3), np.float64)
                        m_real = trunk_arr < n_orig_g
                        trunk_o[m_real] = pos_o[trunk_arr[m_real]]
                        if (~m_real).any():
                            trunk_o[~m_real] = src[trunk_arr[~m_real]]
                        trunk_proj = (trunk_o - center_pca) @ pca2
                        thr_proj = float(np.median(trunk_proj))
                        is_up = trunk_proj > thr_proj
                        DISPLACE_VOX = _TOPO_PAIR_DISPLACE_VOX

                        # Arms displace toward their colour side with a
                        # SMOOTHSTEP RAMP: full DISPLACE_VOX at the junction
                        # (matching the trunk, so the gap is bridged) decaying
                        # to 0 by RAMP_VOX, which keeps the far arm continuous
                        # with the un-displaced cloud.  The signs are PER
                        # PAIRING (A: arm[0,2] up; B: arm[0,3] up), so
                        # toggling shows which pairing lines up with the split.
                        RAMP_VOX = max(DISPLACE_VOX * 5.0, 1e-6)
                        # `gidx` = every node of this >-----< (arm0..arm3
                        # then the shared trunk), used by the pairing
                        # ACCEPT to rewire exactly these nodes.
                        _sep_gidx = np.concatenate(
                            [np.asarray(a, np.int64) for a in full_arms]
                            + [trunk_arr.astype(np.int64)])
                        _sep_payload = {
                            'J':     int(start_J),
                            'J2':    int(next_jct),
                            'gidx':  _sep_gidx,
                            # The highlighted >-----<: 4 capped arm subtrees
                            # + shared trunk + both junctions ('D' delete).
                            'viz_gidx': np.asarray(
                                sorted(arm_nodes | chain_set), np.int64),
                            # What 'D' delete actually removes (shorter cap).
                            'del_gidx': np.asarray(
                                sorted(del_arm_nodes | chain_set), np.int64),
                            # Per-arm break points for downstream linking.
                            'arm_breaks': arm_breaks,
                        }
                        if auto:
                            return ('separate', _sep_payload)

                        # ── 4 CANDIDATES = the arm up/down arrangements.  At
                        # EACH junction the two arms are MUTUALLY EXCLUSIVE
                        # (one up, one down — never the same way).  Left
                        # junction has 2 states (up-down / down-up), right junction 2
                        # states → 2×2 = 4.  The middle TRUNK is NOT touched
                        # (kept at its real position, drawn gray).  ← / →
                        # cycle the 4, ENTER accepts the shown one.
                        palette3 = np.stack([
                            np.array([1.00, 0.25, 0.30]),   # 0 = up   (red)
                            np.array([0.10, 0.70, 0.95]),   # 1 = down (cyan)
                            np.array([0.62, 0.62, 0.66])])  # 2 = trunk(gray)
                        _left = [i for i in range(len(arm_starts))
                                 if int(arm_starts[i][1]) == int(start_J)]
                        _right = [i for i in range(len(arm_starts))
                                  if int(arm_starts[i][1]) == int(next_jct)]
                        s_node = np.where(is_up, 1, -1).astype(np.int64)
                        _arm_w = []        # per-arm ramp (1 @junction → 0 far)
                        for ai, arm_g in enumerate(full_arms):
                            _b = src[np.asarray(arm_g, np.int64)].astype(
                                np.float64)
                            _jp = src[int(arm_starts[ai][1])]
                            _uu = np.clip(np.linalg.norm(_b - _jp, axis=1)
                                          / RAMP_VOX, 0.0, 1.0)
                            _arm_w.append(1.0 - (3.0 * _uu**2 - 2.0 * _uu**3))
                        _base_all = np.vstack(
                            [src[np.asarray(a, np.int64)] for a in full_arms]
                            + [src[trunk_arr]]).astype(np.float64)
                        _ntrunk0 = int(sum(n_arms_pts))     # trunk rows start
                        # Shared trunk → TWO display strands.  NO ramp: just
                        # move the two ENDPOINTS to ±DISPLACE_VOX (the up/down
                        # arm sides) and LINEARLY INTERPOLATE the middle, so
                        # each strand is a straight line J→J2 at its side.
                        # Split the trunk's points EVENLY (even idx → up/red,
                        # odd idx → down/cyan) so each strand spans J→J2.
                        _J0 = src[int(start_J)].astype(np.float64)
                        _J1 = src[int(next_jct)].astype(np.float64)
                        _nT = len(trunk_arr)
                        _red_t = np.arange(_nT)[0::2]   # up strand point rows
                        _cyan_t = np.arange(_nT)[1::2]  # down strand point rows

                        def _lerp_line(a0, a1, n):
                            if n <= 0:
                                return np.zeros((0, 3), np.float64)
                            if n == 1:
                                return a0[None, :].copy()
                            return (a0[None, :]
                                    + np.linspace(0.0, 1.0, n)[:, None]
                                    * (a1 - a0)[None, :])
                        _red_line = _lerp_line(_J0 + DISPLACE_VOX * pca2,
                                               _J1 + DISPLACE_VOX * pca2,
                                               len(_red_t))
                        _cyan_line = _lerp_line(_J0 - DISPLACE_VOX * pca2,
                                                _J1 - DISPLACE_VOX * pca2,
                                                len(_cyan_t))
                        _trunk_disp_pos = np.empty((_nT, 3), np.float64)
                        _trunk_disp_pos[_red_t] = _red_line
                        _trunk_disp_pos[_cyan_t] = _cyan_line
                        _cid_trunk = np.empty(_nT, np.int64)
                        _cid_trunk[_red_t] = 0
                        _cid_trunk[_cyan_t] = 1
                        _combos = [(0, 0), (0, 1), (1, 0), (1, 1)]  # (L, R)
                        _clabels = []
                        s['topo_pairing_combos'] = []
                        for ck, (ls, rs) in enumerate(_combos):
                            # the two arms at each junction go OPPOSITE ways
                            sign = [0] * len(full_arms)
                            if len(_left) == 2:
                                sign[_left[0]] = 1 if ls == 0 else -1
                                sign[_left[1]] = -1 if ls == 0 else 1
                            if len(_right) == 2:
                                sign[_right[0]] = 1 if rs == 0 else -1
                                sign[_right[1]] = -1 if rs == 0 else 1
                            # arm displacement ±pca2 (ramped)
                            disp_d = np.zeros(len(_base_all))
                            _off = 0
                            for ai in range(len(full_arms)):
                                _m = n_arms_pts[ai]
                                disp_d[_off:_off + _m] = (
                                    sign[ai] * DISPLACE_VOX * _arm_w[ai])
                                _off += _m
                            # DISPLAY: arms = real+disp (up red / down cyan);
                            # trunk rows = the two straight interpolated
                            # strands (even→red up, odd→cyan down), point
                            # count split evenly.
                            pts_disp = (_base_all
                                        + disp_d[:, None] * pca2[None, :]
                                        ).astype(np.float32)
                            pts_disp[_ntrunk0:] = _trunk_disp_pos.astype(
                                np.float32)
                            cid_d = np.concatenate(
                                [np.full(n_arms_pts[ai],
                                         0 if sign[ai] > 0 else 1)
                                 for ai in range(len(full_arms))]
                                + [_cid_trunk]).astype(np.int64)
                            # ACCEPT geometry: same arms + trunk split by is_up
                            # so a re-MST cleanly separates two strands (up
                            # arms + up-trunk = α, the rest = β).
                            disp_a = disp_d.copy()
                            disp_a[_ntrunk0:] = s_node * DISPLACE_VOX
                            pts_acc = (_base_all
                                       + disp_a[:, None] * pca2[None, :]
                                       ).astype(np.float32)
                            cid_a = np.concatenate(
                                [np.full(n_arms_pts[ai],
                                         0 if sign[ai] > 0 else 1)
                                 for ai in range(len(full_arms))]
                                + [np.where(s_node > 0, 0, 1)]).astype(np.int64)
                            _lbl = (f'L({"up-down" if ls == 0 else "down-up"}) '
                                    f'R({"up-down" if rs == 0 else "down-up"})')
                            _clabels.append(_lbl)
                            s['topo_pairing_combos'].append((cid_a, pts_acc))
                            _pck = ps.register_point_cloud(
                                f'topo_branch_pairing_{ck}', pts_disp)
                            _pck.set_radius(1.50, relative=False)
                            _pck.add_color_quantity(
                                'arm_updown', palette3[cid_d], enabled=True)
                            try:
                                _pck.set_enabled(ck == 0)
                            except Exception:
                                pass
                        state['topo_branch_pairing_shown'] = 0
                        state['topo_pairing_payload'] = _sep_payload
                        state['topo_pairing_labels'] = _clabels
                        print('  [trunk-probe] 4 candidates (arm up/down, '
                              'trunk untouched/gray): '
                              + '  '.join(f'{i}={_l}'
                                          for i, _l in enumerate(_clabels))
                              + '.  red=up cyan=down — ←/→ cycle, '
                                'ENTER accepts, D deletes the >-----<.')
                    return ('separate', _sep_payload)
            else:
                L     = deepest_node
                L_pos = src[L]
                print(f'  [arm-connect] shorter-arm deepest leaf: '
                      f'gidx{L} at depth {deepest_depth}, |L−J|='
                      f'{np.linalg.norm(L_pos - pos_J):.2f}vox, '
                      f'pos={np.round(L_pos, 2)}')
                # blue ball at the leaf
                pcl = ps.register_point_cloud(
                    'topo_branch_bridge_leaf',
                    L_pos.astype(np.float32).reshape(1, 3))
                pcl.set_radius(3.00, relative=False)
                pcl.set_color((0.10, 0.40, 1.00))    # bright blue
                pcl.add_scalar_quantity(
                    'global_idx',
                    np.asarray([float(L)], np.float64),
                    enabled=False)
                # ── Bridge search restricted to long-arm subtree.
                # Radius = UI's `curve_radius` (NOT hardcoded 5).
                # Pick ONLY the single nearest qualifying point; if
                # none, abort the loop search entirely.
                RADIUS = float(s.get('curve_radius', 4.0))
                bridge_tree = cKDTree(src)
                nbrs = bridge_tree.query_ball_point(L_pos, r=RADIUS)
                bridge_cands = [int(n) for n in nbrs
                                if int(n) in long_sub
                                and int(n) != L]
                if ps.has_curve_network('topo_branch_loop'):
                    ps.remove_curve_network('topo_branch_loop')
                if not bridge_cands:
                    # Classified as CUT.  In batch mode only delete during
                    # the 'cut' phase; otherwise just report the class.
                    if auto and apply_action != 'cut':
                        return ('cut', None)
                    print(f'  [arm-connect] NO point on branch'
                          f'[{long_li}] subtree within curve_radius'
                          f'={RADIUS:.2f}vox of leaf; AUTO-STITCH: '
                          f'delete short arm (branch[{short_li}]) '
                          f'from topo_mst.')
                    if not auto:
                        _topo_stitch_snapshot(
                            f'short-arm stitch @ J=gidx{int(start_J)}')
                    # short_sub already covers the WHOLE short-arm
                    # subtree (every node reachable from ne_short
                    # with J blocked, including sub-junction twigs
                    # along the chain).  Clear every MST edge that
                    # touches it — including the boundary edge
                    # J ↔ ne_short — so twig-leaves don't linger as
                    # ghost green leaf markers after the re-render.
                    n_rm = 0
                    for u in short_sub:
                        if u not in global_adj:
                            continue
                        for v in list(global_adj[u]):
                            global_adj[u].discard(v)
                            if v in global_adj:
                                global_adj[v].discard(u)
                            n_rm += 1   # may double-count interior
                    print(f'  [arm-connect] cleared ~{n_rm} adj '
                          f'entries across {len(short_sub)} nodes of '
                          f'branch[{short_li}] subtree '
                          f'(J=gidx{start_J} should drop from deg 3 '
                          f'→ 2 → no longer a red junction; twig '
                          f'leaves on the short arm also gone)')
                    if auto:
                        return ('cut', None)
                    # Refresh topo_mst viz from the MODIFIED adj.
                    # use_cached_adj=True → keeps our stitch alive
                    # (a fresh rebuild would put the edges back).
                    try:
                        _topo_mst_viz(use_cached_adj=True)
                    except Exception as _ex:
                        print(f'  [arm-connect] re-render warn: {_ex}')
                    # The clicked red junction no longer exists (deg
                    # 3 → 2).  Drop EVERY viz this click produced —
                    # the user shouldn't see leftovers of a ghost
                    # junction.
                    _clear_topo_branch_click_viz()
                    return ('cut', None)
                else:
                    bridge_cands.sort(
                        key=lambda n: float(np.linalg.norm(
                            src[n] - L_pos)))
                    Y      = bridge_cands[0]
                    Y_dist = float(np.linalg.norm(src[Y] - L_pos))
                    print(f'  [arm-connect] nearest bridge: gidx{Y} '
                          f'at {Y_dist:.2f}vox  (chosen from '
                          f'{len(bridge_cands)} cands in '
                          f'branch[{long_li}] within {RADIUS:.2f}vox)')
                    # Reconstruct short-arm path J→L from parent_bfs.
                    short_path: list[int] = []
                    cur = L
                    while cur != -1:
                        short_path.append(cur)
                        cur = parent_bfs[cur]
                    short_path.reverse()       # [J, ne_short, …, L]
                    # BFS along LONG arm (J blocked) for Y's parent
                    # chain → unique MST path Y→J in long subtree.
                    long_parent: dict[int, int] = {
                        int(start_J): -1, long_ne: int(start_J)}
                    long_queue = [long_ne]
                    qi2 = 0
                    while qi2 < len(long_queue):
                        u2 = long_queue[qi2]; qi2 += 1
                        for w2 in global_adj.get(u2, ()):
                            iw2 = int(w2)
                            if (iw2 not in long_parent
                                    and iw2 != int(start_J)):
                                long_parent[iw2] = u2
                                long_queue.append(iw2)
                    if Y not in long_parent:
                        print('  [arm-connect] WARN: Y not reachable '
                              'via long-arm BFS; ABORT loop draw')
                    else:
                        y_path_rev: list[int] = []
                        cur = Y
                        while cur != -1:
                            y_path_rev.append(cur)
                            cur = long_parent[cur]
                        # y_path_rev = [Y, …, long_ne, J]
                        # loop = J→short→L  +  Y→long→J
                        loop = short_path + y_path_rev
                        print(f'  [arm-connect] loop = {len(loop)} '
                              f'nodes  (short arm {len(short_path)} '
                              f'+ long arm {len(y_path_rev)}); '
                              f'J@idx0  L@idx{len(short_path)-1}  '
                              f'Y@idx{len(short_path)}  '
                              f'J@idx{len(loop)-1}')

                        # ── Find ANOTHER junction along the loop.
                        # For every loop node (J excluded), inspect
                        # its MST neighbours.  Any neighbour NOT on
                        # the loop is an "off-branch"; its subtree
                        # size (BFS in global_adj, loop blocked) tells
                        # us how big the dangling structure is.  The
                        # biggest off-branch is the most likely
                        # second junction of the >-----< topology
                        # (= trunk attachment from the OTHER side).
                        for _nm in ('topo_branch_other_jct',
                                    'topo_branch_other_trunk'):
                            if ps.has_point_cloud(_nm):
                                ps.remove_point_cloud(_nm)
                        loop_set = set(loop) | {int(start_J)}
                        SUBTREE_CAP  = 10000
                        MIN_OFF_SIZE = 3   # filter twigs (size ≤ 2)
                        off_cands: list[tuple[int, int, int]] = []
                        n_twig = 0
                        for u_loop in loop_set:
                            if u_loop == int(start_J):
                                continue       # skip the clicked J
                            for v_nb in global_adj.get(u_loop, ()):
                                iv_nb = int(v_nb)
                                if iv_nb in loop_set:
                                    continue
                                # BFS off-branch size (loop blocked)
                                stk = [iv_nb]
                                seen_o = set(loop_set)
                                seen_o.add(iv_nb)
                                sz = 0
                                while stk and sz < SUBTREE_CAP:
                                    x = stk.pop()
                                    sz += 1
                                    for w_nb in global_adj.get(x, ()):
                                        iw_nb = int(w_nb)
                                        if iw_nb not in seen_o:
                                            seen_o.add(iw_nb)
                                            stk.append(iw_nb)
                                if sz >= MIN_OFF_SIZE:
                                    off_cands.append(
                                        (int(u_loop), iv_nb, sz))
                                else:
                                    n_twig += 1
                        if not off_cands:
                            print(f'  [arm-connect] no off-branch '
                                  f'with subtree size ≥ '
                                  f'{MIN_OFF_SIZE} ({n_twig} twig(s) '
                                  f'filtered); no second junction '
                                  f'detected')
                        else:
                            off_cands.sort(key=lambda t: -t[2])
                            print(f'  [arm-connect] {len(off_cands)} '
                                  f'off-branch(es) ≥ {MIN_OFF_SIZE} '
                                  f'pts along the loop ({n_twig} '
                                  f'twig(s) filtered); top 5:')
                            for u_loop, v_nb, sz in off_cands[:5]:
                                cs = '≥' if sz >= SUBTREE_CAP else '='
                                print(f'      gidx{u_loop} (on loop) '
                                      f'→ off via gidx{v_nb}  '
                                      f'subtree size {cs}{sz}')
                            o_jct, o_first, o_size = off_cands[0]
                            cs = ('≥' if o_size >= SUBTREE_CAP
                                  else '=')
                            print(f'  [arm-connect] → OTHER junction '
                                  f'candidate = gidx{o_jct} '
                                  f'(biggest off-subtree {cs}{o_size}'
                                  f' via gidx{o_first})')
                            # Two-tone the loop along o_jct: half-A
                            # J→o_jct (pink) | half-B o_jct→J (blue).
                            # Uses edge color quantity so the boundary
                            # at o_jct is a hard cut, not a gradient.
                            try:
                                idx_oj = loop.index(int(o_jct))
                            except ValueError:
                                idx_oj = -1
                            if 0 < idx_oj < len(loop) - 1:
                                half_a_idx = loop[:idx_oj + 1]
                                half_b_idx = loop[idx_oj:][::-1]


                                # ── Centerline computation (ACTIVE).
                                # Resample each half by arc-length to N
                                # equally-spaced points, then take the
                                # pointwise midpoint → geometric medial
                                # axis between the two yarns of the
                                # merged section.  Used by the
                                # centerline-snapped auto-stitch below.
                                n_res = max(2, min(len(half_a_idx),
                                                   len(half_b_idx)))

                                def _arclen_resample(P, n):
                                    if len(P) < 2 or n < 2:
                                        return (np.tile(P[0], (n, 1))
                                                if len(P) > 0
                                                else np.zeros((n, 3)))
                                    diffs    = np.diff(P, axis=0)
                                    seg_lens = np.linalg.norm(
                                        diffs, axis=1)
                                    cum = np.concatenate(
                                        [[0.0], np.cumsum(seg_lens)])
                                    total = cum[-1]
                                    if total < 1e-9:
                                        return np.tile(P[0], (n, 1))
                                    t_t = np.linspace(0.0, total, n)
                                    out = np.zeros((n, 3),
                                                   dtype=P.dtype)
                                    for ii, t in enumerate(t_t):
                                        seg = int(np.searchsorted(
                                            cum, t, side='right')) - 1
                                        seg = max(0, min(seg,
                                                         len(P) - 2))
                                        d = cum[seg + 1] - cum[seg]
                                        a = (0.0 if d < 1e-9
                                             else (t - cum[seg]) / d)
                                        out[ii] = (P[seg] * (1 - a)
                                                   + P[seg + 1] * a)
                                    return out

                                Ha = src[np.asarray(half_a_idx)
                                         ].astype(np.float64)
                                Hb = src[np.asarray(half_b_idx)
                                         ].astype(np.float64)
                                Ha_r = _arclen_resample(Ha, n_res)
                                Hb_r = _arclen_resample(Hb, n_res)
                                centerline = (Ha_r + Hb_r) * 0.5


                                # ── AUTO-STITCH (loop, CENTERLINE-AS-NEW-PTS).
                                # Both halves are DELETED from the
                                # MST; the centerline is added as a
                                # NEW chain of synthetic gidx (≥ N,
                                # backed by `state['topo_mst_extra_pts']`).
                                # state['pts'] is NOT touched.
                                # Classified as LOOP (o_jct found, valid
                                # split).  In batch mode only stitch
                                # during the 'loop' phase.
                                if auto and apply_action != 'loop':
                                    return ('loop', None)
                                if not auto:
                                    _topo_stitch_snapshot(
                                        f'loop stitch (centerline-as-new-pts) '
                                        f'@ J=gidx{int(start_J)} via '
                                        f'o_jct=gidx{int(o_jct)}')
                                loop_pool_set = (
                                    set(int(g) for g in half_a_idx)
                                    | set(int(g) for g in half_b_idx))
                                loop_region = (loop_pool_set
                                               | {int(start_J),
                                                  int(o_jct)})
                                # Append centerline interior points to
                                # extras (skip first/last = J/o_jct
                                # positions, already in state['pts']).
                                n_orig_st = len(state['pts'])
                                existing_extras = s.get(
                                    'topo_mst_extra_pts')
                                if (existing_extras is None
                                        or len(existing_extras) == 0):
                                    existing_extras = np.zeros(
                                        (0, 3), dtype=np.float64)
                                new_ctr_pts = centerline[1:-1].astype(
                                    np.float64)
                                n_exist = len(existing_extras)
                                synth_gidx_start = (n_orig_st + n_exist)
                                synth_gidx_list = list(range(
                                    synth_gidx_start,
                                    synth_gidx_start + len(new_ctr_pts)))
                                s['topo_mst_extra_pts'] = np.vstack(
                                    [existing_extras, new_ctr_pts])
                                # New chain: J → synth_0 → ... → synth_k
                                # → o_jct.
                                chain = ([int(start_J)] + synth_gidx_list
                                         + [int(o_jct)])
                                # Save external attachments at J and
                                # o_jct (the only chain nodes that
                                # currently have any adj — synthetic
                                # gidx are fresh).
                                ext_for_chain: dict[int, set[int]] = {}
                                for u_ch in (int(start_J), int(o_jct)):
                                    if u_ch not in global_adj:
                                        continue
                                    ext = {int(v) for v in
                                           global_adj[u_ch]
                                           if int(v) not in loop_region}
                                    if ext:
                                        ext_for_chain[u_ch] = ext
                                # Clear EVERY edge of every loop_region
                                # node (both directions) — both halves
                                # gone.
                                n_rm = 0
                                for u in loop_region:
                                    if u not in global_adj:
                                        continue
                                    for v in list(global_adj[u]):
                                        global_adj[u].discard(v)
                                        if v in global_adj:
                                            global_adj[v].discard(u)
                                        n_rm += 1
                                # Add new chain edges (synthetic gidx
                                # do NOT exist in global_adj yet —
                                # setdefault creates them).
                                n_add = 0
                                for ic in range(len(chain) - 1):
                                    uc = int(chain[ic])
                                    vc = int(chain[ic + 1])
                                    global_adj.setdefault(
                                        uc, set()).add(vc)
                                    global_adj.setdefault(
                                        vc, set()).add(uc)
                                    n_add += 1
                                # Re-add external attachments at J +
                                # o_jct (trunk side, off-branch side).
                                n_ext = 0
                                for u_ch, exts in ext_for_chain.items():
                                    for v in exts:
                                        global_adj.setdefault(
                                            u_ch, set()).add(v)
                                        global_adj.setdefault(
                                            v, set()).add(u_ch)
                                        n_ext += 1
                                print(f'  [arm-connect] AUTO-STITCH '
                                      f'(loop, CENTERLINE-AS-NEW-PTS): '
                                      f'{len(loop_region)} loop-region '
                                      f'nodes deleted (cleared ~{n_rm} '
                                      f'edges); centerline added as '
                                      f'{len(new_ctr_pts)} synthetic '
                                      f'pts (gidx {synth_gidx_start}..'
                                      f'{synth_gidx_start + len(new_ctr_pts) - 1}) '
                                      f'forming a {len(chain)}-node '
                                      f'chain (added {n_add} edges + '
                                      f'{n_ext} preserved external '
                                      f'attachments at J/o_jct)')
                                if auto:
                                    return ('loop', None)
                                try:
                                    _topo_mst_viz(use_cached_adj=True)
                                except Exception as _ex:
                                    print(f'  [arm-connect] re-render '
                                          f'warn: {_ex}')
                                _clear_topo_branch_click_viz()
                                return ('loop', None)
    else:
        print(f'    [skip] pairwise angle analysis needs exactly 3 '
              f'branches (got {len(closest_pts)})')
    # No decision leaf reached (deg≠3, degenerate vectors, loop with no
    # valid o_jct, etc.) → nothing to do.
    return ('skip', None)


def _mst_edges_for_positions(positions: np.ndarray, radius: float,
                             min_pts: int):
    """Re-derive the proximity-graph MST on `positions` (N,3).  Returns
    `(edges, seg_labels)`: edges is an (E,2) int array of LOCAL-index
    MST edges (one MST per segment); seg_labels is the
    `_compute_segments` label per node (radius-proximity component id,
    -1 = dropped <min_pts) so the viz can colour by SEGMENT.  Used by the
    possibility viz: after the split-apart MOVE pulls the merged stem
    into two separated strands, a fresh MST on the moved coords connects
    within-strand (no zig-zag / no threads)."""
    seg_labels, K = _compute_segments(positions, radius, min_pts)
    edges: list = []
    for sid in range(int(K)):
        idx = np.where(seg_labels == sid)[0]
        if len(idx) < 4:
            continue
        adj, _M = _cc_mst_adj(positions[idx], radius=radius)
        if adj is None:
            continue
        for i_loc, neigh in adj.items():
            gi = int(idx[i_loc])
            for n_loc in neigh:
                gj = int(idx[int(n_loc)])
                if gi < gj:
                    edges.append((gi, gj))
    edges_arr = (np.asarray(edges, np.int32) if edges
                 else np.zeros((0, 2), np.int32))
    return edges_arr, seg_labels


def _topo_stitch_snapshot(label: str) -> None:
    """Snapshot `state['topo_mst_global_adj']` AND
    `state['topo_mst_extra_pts']` BEFORE a stitch mutation so the
    operation can be undone.  `label` is a short human-readable
    description shown by the undo log.  Stack capped at 10 entries
    (oldest dropped) to bound memory."""
    s = state
    adj = s.get('topo_mst_global_adj')
    if adj is None:
        return
    stack = s.setdefault('topo_mst_undo_stack', [])
    # shallow-copy the dict but deep-copy each set value (sets are
    # mutated in place by stitches, so we need our own copies)
    adj_snap = {int(k): set(v) for k, v in adj.items()}
    extras = s.get('topo_mst_extra_pts')
    extras_snap = (extras.copy()
                   if extras is not None and len(extras) > 0
                   else None)
    override = s.get('topo_mst_pos_override')
    override_snap = (dict(override) if override else None)
    deleted = s.get('topo_mst_deleted')
    deleted_snap = (set(deleted) if deleted else None)
    stack.append((label, adj_snap, extras_snap, override_snap,
                  deleted_snap))
    if len(stack) > 10:
        del stack[:len(stack) - 10]


def _topo_stitch_undo() -> None:
    """Pop the most-recent snapshot off the undo stack and restore
    `state['topo_mst_global_adj']` from it.  Re-renders the topo_mst
    viz so the restored MST shows up immediately.  No-op if the
    stack is empty."""
    s = state
    stack = s.get('topo_mst_undo_stack', [])
    if not stack:
        print('  [stitch-undo] nothing to undo (empty stack)')
        return
    item = stack.pop()
    override_snap = None
    deleted_snap = None
    if len(item) == 5:
        (label, adj_snap, extras_snap, override_snap,
         deleted_snap) = item
    elif len(item) == 4:
        label, adj_snap, extras_snap, override_snap = item
    elif len(item) == 3:
        label, adj_snap, extras_snap = item
    else:
        # legacy 2-tuple snapshot (pre-extras)
        label, adj_snap = item
        extras_snap = None
    s['topo_mst_global_adj'] = adj_snap
    s['topo_mst_extra_pts'] = extras_snap
    s['topo_mst_pos_override'] = override_snap
    s['topo_mst_deleted'] = set(deleted_snap) if deleted_snap else set()
    print(f'  [stitch-undo] restored snapshot: {label}  '
          f'(stack now has {len(stack)} entry(ies))')
    try:
        _topo_mst_viz(use_cached_adj=True)
    except Exception as _ex:
        print(f'  [stitch-undo] re-render warn: {_ex}')


_TOPO_DELETE_ARM_HOPS = 50     # D-delete: arm hops removed per side
_TOPO_MST_CLICKDEL_VOX = 3.0   # hardcoded radius for Mode A (vox)
_TOPO_MST_CONNECT_STEP = 2.0   # vox between synthetic bridge pts (Connect 1↔2 / Sketch-bridge)


def _topo_mst_clickdel_clear() -> None:
    """Drop BOTH `topo_mst` picks (yellow + cyan markers + state)."""
    s = state
    s['topo_mst_pick_node']  = -1
    s['topo_mst_pick_gidx']  = -1
    s['topo_mst_pick_deg']   = -1
    s['topo_mst_pick2_node'] = -1
    s['topo_mst_pick2_gidx'] = -1
    s['topo_mst_pick2_deg']  = -1
    s['topo_mst_last_clickdel'] = -1
    for nm in ('topo_mst_pick', 'topo_mst_pick2'):
        if ps.has_point_cloud(nm):
            ps.remove_point_cloud(nm)


def _report_new_junctions(jct_before: set) -> None:
    """After a Connect / Sketch-bridge re-viz, compare the real
    junctions (the red `topo_mst_junctions`: deg≥3 with every branch
    > jct_min, stored in `topo_mst_jct_info`) against the snapshot
    `jct_before` (gidx set taken BEFORE the connect).  Print the
    NEWLY-created junctions, numbered.  Junctions that DISAPPEARED
    are ignored."""
    s = state
    info_after = s.get('topo_mst_jct_info') or []
    after = {int(d['gidx']): d for d in info_after}
    new_g = sorted(g for g in after if g not in jct_before)
    if not new_g:
        print(f'  [jct-diff] no NEW junctions  '
              f'(before={len(jct_before)} → after={len(after)})')
        return
    adj = s.get('topo_mst_global_adj') or {}
    # RED junctions to compare against = pre-Connect junctions that are
    # STILL junctions after the connect (after ∩ jct_before) — i.e. the
    # ones still drawn red (the newly created ones are excluded).
    red_g = [g for g in after if g in jct_before and after[g].get('pos')]
    red_pos = (np.asarray([after[g]['pos'] for g in red_g], np.float64)
               if red_g else np.zeros((0, 3), np.float64))
    print(f'  [jct-diff] {len(new_g)} NEW junction(s) after Connect  '
          f'(before={len(jct_before)} → after={len(after)}; '
          f'{len(red_g)} red JCT to compare):')
    near_info = {}   # g -> (nearest_red_gidx, dist) — for the midpoint-cut rule
    for k, g in enumerate(new_g, 1):
        d = after[g]
        p = d.get('pos')
        deg = len(adj.get(g, ()))
        nb = len(d.get('branches', []))
        # nearest pre-Connect (red) junction to this new one + dist
        if p is not None and len(red_pos):
            _dd = np.linalg.norm(red_pos - np.asarray(p, np.float64),
                                 axis=1)
            _j = int(np.argmin(_dd))
            _ndist = float(_dd[_j])
            near_info[g] = (int(red_g[_j]), _ndist)
            near_str = (f'  → nearest red JCT gidx={red_g[_j]} '
                        f'dist={_ndist:.1f} vox')
        else:
            _ndist = float('nan')
            near_str = '  → nearest red JCT: none'
        print(f'      #{k}: gidx={g}  deg={deg}  branches={nb}  '
              f'pos={[round(float(x), 1) for x in p] if p else "?"}'
              f'{near_str}')
    # ── Auto-run the junction CLICK pipeline on each new junction ────
    # Same as clicking the red cap: _topo_branch_explore in its
    # default mode → branch analysis + viz + auto-APPLY the classified
    # action (cut / loop / search; see the COLLAPSE rule), snapshotting per
    # junction (each Undo-able via "Undo stitch").
    # Turn off with state['connect_auto_process_new_jct'] = False.
    if not s.get('connect_auto_process_new_jct', True):
        return
    # MIDPOINT-CUT rule: if a new junction's nearest red
    # junction is within `midcut_vox`, BFS the MST path between them and
    # BREAK it at its arc-length midpoint edge BEFORE the click — so the
    # click analyses the new region already detached from that nearby
    # red junction.  Threshold: state['connect_newjct_midcut_vox'].
    midcut_vox = float(s.get('connect_newjct_midcut_vox', 100.0))
    src_pts = np.asarray(s['pts'], np.float64)
    _ex = s.get('topo_mst_extra_pts')
    src = (np.vstack([src_pts, np.asarray(_ex, np.float64)])
           if _ex is not None and len(_ex) > 0 else src_pts)

    def _bfs_path_adj(_adj, a0, b0):
        a0, b0 = int(a0), int(b0)
        if a0 == b0:
            return [a0]
        par = {a0: -1}; q = [a0]; h = 0
        while h < len(q):
            u = q[h]; h += 1
            if u == b0:
                break
            for v in _adj.get(u, ()):
                iv = int(v)
                if iv not in par:
                    par[iv] = u; q.append(iv)
        if b0 not in par:
            return None
        out = [b0]; c = b0
        while c != a0:
            c = par[c]; out.append(c)
        out.reverse(); return out

    # Dedup the midpoint-cut by target red junction: if several new
    # junctions share the SAME nearest red (within threshold), only the
    # CLOSEST of them gets the cut (the others skip the cut but still
    # get clicked).  cut_winner[red] = (new_gidx, dist).
    cut_winner: dict = {}
    for _g, (_rg, _d) in near_info.items():
        if _d <= midcut_vox and (_rg not in cut_winner
                                 or _d < cut_winner[_rg][1]):
            cut_winner[_rg] = (_g, _d)
    winners = {wg for (wg, _wd) in cut_winner.values()}

    # If NO new junction is within `midcut_vox` of a red JCT,
    # skip the WHOLE auto-process (no midpoint-cuts, no clicks).
    if not cut_winner:
        print(f'  [jct-diff] no new junction within {midcut_vox:.0f}vox of a '
              f'red JCT → skip auto-process (no cuts, no clicks)')
        return

    # COLLAPSE rule: a collapse (>-----<, classified 'separate') is an
    # undecided connection with two pairings, so the auto-process leaves
    # it alone.  BEFORE any midpoint-cut, classify every new junction
    # (auto mode with a non-phase apply_action returns the label without
    # mutating) and skip the WHOLE auto-process if any is a collapse.
    # The classifier still draws click overlays, so clear those.
    import io
    import contextlib
    collapse_g: list = []
    for g in new_g:
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                _act, _ = _topo_branch_explore(
                    int(g), n_per_branch=100, auto=True,
                    apply_action='classify')
        except Exception as _ex:
            print(f'  [jct-diff] classify gidx={g} warn: {_ex}')
            continue
        if _act == 'separate':
            collapse_g.append(int(g))
    if collapse_g:
        _clear_topo_branch_click_viz()
        print(f'  [jct-diff] new junction(s) gidx {collapse_g} classify as '
              f'collapse (>-----<) → skip auto-process (no cuts, no clicks)')
        return

    involved_reds: list = []   # red JCTs whose path got midpoint-cut
    print(f'  [jct-diff] auto-running click pipeline on '
          f'{len(new_g)} new junction(s)…')
    for k, g in enumerate(new_g, 1):
        print(f'  [jct-diff] ── auto-click #{k}: gidx={g} ──')
        nr = near_info.get(g)
        if nr is not None and nr[1] <= midcut_vox and g not in winners:
            _rg, _d = nr
            _wg, _wd = cut_winner[_rg]
            print(f'  [jct-diff]    dist {_d:.1f}≤{midcut_vox:.0f}vox but a '
                  f'CLOSER new junction (gidx{_wg} @ {_wd:.1f}vox) shares red '
                  f'gidx{_rg} — skip midpoint-cut for this one')
        elif nr is not None and nr[1] <= midcut_vox:
            red_gidx, dist = nr
            cur_adj = s.get('topo_mst_global_adj') or {}
            path = _bfs_path_adj(cur_adj, g, red_gidx)
            if (path and len(path) >= 2
                    and all(0 <= int(n) < len(src) for n in path)):
                pp = src[np.asarray(path, np.int64)]
                seg = np.linalg.norm(np.diff(pp, axis=0), axis=1)
                cum = np.concatenate([[0.0], np.cumsum(seg)])
                tot = float(cum[-1])
                # edge straddling the 50%-arc-length point
                e = int(np.searchsorted(cum, tot * 0.5)) - 1
                e = max(0, min(e, len(path) - 2))
                u_cut, v_cut = int(path[e]), int(path[e + 1])
                _topo_stitch_snapshot(
                    f'midpoint-cut new gidx{g} ↔ red gidx{red_gidx} '
                    f'(dist={dist:.1f}vox, cut gidx{u_cut}–gidx{v_cut})')
                if u_cut in cur_adj:
                    cur_adj[u_cut].discard(v_cut)
                if v_cut in cur_adj:
                    cur_adj[v_cut].discard(u_cut)
                print(f'  [jct-diff]    dist {dist:.1f}≤{midcut_vox:.0f}vox '
                      f'→ broke path to red gidx{red_gidx} at midpoint: '
                      f'cut edge gidx{u_cut}–gidx{v_cut} '
                      f'(path {len(path)} nodes, arclen {tot:.1f}vox)')
                involved_reds.append(int(red_gidx))
            elif path is None:
                print(f'  [jct-diff]    dist {dist:.1f}≤{midcut_vox:.0f}vox '
                      f'but no MST path to red gidx{red_gidx} — skip cut')
        try:
            _topo_branch_explore(int(g), n_per_branch=100)
        except Exception as _ex:
            print(f'  [jct-diff] auto-click #{k} (gidx={g}) warn: {_ex}')
    # Also auto-click the RED junctions that got pulled into a
    # midpoint-cut — severing the path changed their local structure,
    # so they should be re-processed too (dedup'd: one cut per red).
    for k, rg in enumerate(involved_reds, 1):
        print(f'  [jct-diff] ── auto-click involved RED #{k}: gidx={rg} ──')
        try:
            _topo_branch_explore(int(rg), n_per_branch=100)
        except Exception as _ex:
            print(f'  [jct-diff] auto-click red (gidx={rg}) warn: {_ex}')


def _topo_mst_connect_apply() -> None:
    """Connect pick #1 ↔ pick #2 in the MST by sampling intermediate
    synthetic points along the straight Euclidean line between them.
    Step size = _TOPO_MST_CONNECT_STEP (vox).  New points are
    appended to `topo_mst_extra_pts` (gidx ≥ N) and chained in the
    cached `topo_mst_global_adj`.  Refresh via `use_cached_adj=True`
    so prior edits stay alive.  Snapshot-undoable."""
    s = state
    g1 = int(s.get('topo_mst_pick_gidx', -1))
    g2 = int(s.get('topo_mst_pick2_gidx', -1))
    adj = s.get('topo_mst_global_adj')
    if g1 < 0 or g2 < 0 or g1 == g2 or adj is None:
        print('  [topo-mst-connect] need TWO distinct picks + Topo MST '
              'viz cache'); return
    pts = np.asarray(s['pts'], np.float64)
    N_pts = len(pts)
    ex = s.get('topo_mst_extra_pts')
    ex_arr = (np.zeros((0, 3), np.float64) if ex is None
              else np.asarray(ex, np.float64))
    src = np.vstack([pts, ex_arr]) if len(ex_arr) else pts
    if not (0 <= g1 < len(src) and 0 <= g2 < len(src)):
        print(f'  [topo-mst-connect] gidx out of range (g1={g1}, '
              f'g2={g2}, src={len(src)})'); return
    p1 = src[g1]; p2 = src[g2]
    d = float(np.linalg.norm(p2 - p1))
    step = _TOPO_MST_CONNECT_STEP
    n_steps = max(1, int(round(d / step)))
    if n_steps >= 2:
        ts = np.arange(1, n_steps, dtype=np.float64) / float(n_steps)
        new_pts = p1[None, :] + ts[:, None] * (p2 - p1)[None, :]
    else:
        new_pts = np.zeros((0, 3), np.float64)
    _topo_stitch_snapshot(
        f'connect gidx{g1}↔gidx{g2} (dist={d:.2f}, +{len(new_pts)} pts)')
    new_start_gidx = N_pts + len(ex_arr)
    new_ex_arr = (np.vstack([ex_arr, new_pts])
                  if len(new_pts) else ex_arr)
    s['topo_mst_extra_pts'] = (None if len(new_ex_arr) == 0
                               else new_ex_arr)
    chain = ([g1]
             + [new_start_gidx + i for i in range(len(new_pts))]
             + [g2])
    n_edge = 0
    for a, b in zip(chain[:-1], chain[1:]):
        ia, ib = int(a), int(b)
        adj.setdefault(ia, set()).add(ib)
        adj.setdefault(ib, set()).add(ia)
        n_edge += 1
    print(f'  [topo-mst-connect] gidx{g1}↔gidx{g2}: dist={d:.2f} vox, '
          f'step={step:.2f} → +{len(new_pts)} synth pts (gidx '
          f'{new_start_gidx}..{new_start_gidx + max(0, len(new_pts) - 1)}), '
          f'{n_edge} new chain edges.'
          + '  Refreshing topo_mst… Undo via "Undo stitch".')
    _repair_stats_bump('A', g1, g2, new_start_gidx, len(new_pts))
    _topo_mst_clickdel_clear()
    jct_before = {int(d['gidx']) for d in (s.get('topo_mst_jct_info') or [])}
    try:
        _topo_mst_viz(use_cached_adj=True)
        _report_new_junctions(jct_before)
    except Exception as _ex:
        print(f'  [topo-mst-connect] refresh warn: {_ex}')
    # auto-checkpoint after a manual connect — refreshes
    # seg_state_latest.npz only (no timestamped backup, no console
    # spam).  Lets the user safely exit + resume.
    try:
        _save_seg_state(timestamped=False, quiet=True)
        print('  [seg-save] auto → seg_state_latest.npz')
    except Exception as _ex:
        print(f'  [seg-save] auto warn: {_ex}')


def _topo_mst_sketch_connect() -> None:
    """Like _topo_mst_connect_apply, but the bridge follows a 2D curve
    sketched by the user in a tk window overlaid on the current
    polyscope camera view.  Each drawn pixel is back-projected to 3D
    using depth = lerp(d1, d2, t) along chord-length t ∈ [0, 1] — so
    the curve passes through p1, p2 in 3D and looks like the sketch
    from this camera angle.  Resampled at `_TOPO_MST_CONNECT_STEP`."""
    s = state
    g1 = int(s.get('topo_mst_pick_gidx', -1))
    g2 = int(s.get('topo_mst_pick2_gidx', -1))
    adj = s.get('topo_mst_global_adj')
    if g1 < 0 or g2 < 0 or g1 == g2 or adj is None:
        print('  [sketch-bridge] need TWO distinct picks + Topo MST viz '
              'cache'); return
    pts = np.asarray(s['pts'], np.float64)
    N_pts = len(pts)
    ex = s.get('topo_mst_extra_pts')
    ex_arr = (np.zeros((0, 3), np.float64) if ex is None
              else np.asarray(ex, np.float64))
    src = np.vstack([pts, ex_arr]) if len(ex_arr) else pts
    if not (0 <= g1 < len(src) and 0 <= g2 < len(src)):
        print('  [sketch-bridge] gidx out of range'); return
    p1, p2 = src[g1], src[g2]
    # Sketch only supports perspective (depth-lerp formula assumes pinhole).
    proj_mode = str(ps.get_view_projection_mode()).lower()
    if 'perspective' not in proj_mode:
        print(f'  [sketch-bridge] view mode is {proj_mode} — switch to '
              f'perspective and retry.'); return
    try:
        img = ps.screenshot_to_buffer(transparent_bg=False,
                                      include_UI=False)
    except Exception as exn:
        print(f'  [sketch-bridge] screenshot failed: {exn}'); return
    H_img, W_img = img.shape[:2]
    cp = ps.get_view_camera_parameters()
    V = np.asarray(cp.get_view_mat(), np.float64)
    fovy = np.deg2rad(float(cp.get_fov_vertical_deg()))
    aspect = float(W_img) / float(H_img)
    tan_half = float(np.tan(fovy / 2.0))

    def _world_to_screen(pw):
        pv = V @ np.r_[pw, 1.0]
        d = -pv[2]
        if d <= 1e-9:
            return None, None
        ndc_x = pv[0] / (aspect * d * tan_half)
        ndc_y = pv[1] / (d * tan_half)
        sx = (ndc_x + 1.0) * 0.5 * W_img
        sy = (1.0 - ndc_y) * 0.5 * H_img
        return (sx, sy), d

    s1, d1 = _world_to_screen(p1)
    s2, d2 = _world_to_screen(p2)
    if s1 is None or s2 is None:
        print('  [sketch-bridge] endpoint(s) behind camera — abort'); return

    # ── Tangent hint ────────────────────────────────────────────────
    # Walk back N=25 MST steps from each pick, project the trail, and
    # use the screen-space tangents (one per pick) to seed the Bezier
    # control point P1.  Without this, the default ⟂-midpoint init
    # often suggests a control point on the WRONG side and the user
    # has to drag it 180° before they can refine.  The trails are also
    # drawn on the sketch pad in each pick's colour so the user sees
    # the local MST direction context.
    def _bfs_path(_adj, src_n, dst_n):
        src_n, dst_n = int(src_n), int(dst_n)
        if src_n == dst_n:
            return [src_n]
        parent = {src_n: -1}
        q = [src_n]; head = 0
        while head < len(q):
            u = q[head]; head += 1
            if u == dst_n:
                break
            for v in _adj.get(int(u), ()):
                iv = int(v)
                if iv in parent:
                    continue
                parent[iv] = u; q.append(iv)
        if dst_n not in parent:
            return None
        path = [dst_n]; c = dst_n
        while c != src_n:
            c = parent[c]; path.append(c)
        path.reverse(); return path

    old_path = _bfs_path(adj, g1, g2)
    # If g1↔g2 already share a component, walking back must NOT take
    # the first step toward the other pick — that direction is what
    # the new sketch will replace, so it'd give a backwards tangent.
    excl1 = (int(old_path[1])  if old_path and len(old_path) >= 2 else None)
    excl2 = (int(old_path[-2]) if old_path and len(old_path) >= 2 else None)
    N_BACK = 25

    def _walk_back(start, exclude, this_s, other_s, n=N_BACK):
        """Greedy walk along the MST from `start`, avoiding `exclude`
        on the first step.  At deg≥3 junctions follow the most-aligned
        neighbour.  At the start (no direction yet) pick the branch
        whose 2D screen direction points most AWAY from the other pick
        (`other_s`), so the trail captures the strand as it extends
        away from the bridge — tie-broken (near-⟂ or unprojectable) by
        the longest straight chain.  Returns [(gidx, xyz), …]."""
        start = int(start)
        if start not in adj:
            return [(start, src[start])]
        d_axis = None
        if this_s is not None and other_s is not None:
            _da = (np.asarray(other_s, np.float64)
                   - np.asarray(this_s, np.float64))
            _dn = float(np.linalg.norm(_da))
            if _dn > 1e-9:
                d_axis = _da / _dn

        def _chain_len(first, frm):
            p, c, L, guard = frm, first, 1, 40
            while L < 30 and guard > 0:
                guard -= 1
                nn = [int(v) for v in adj.get(c, ()) if int(v) != p]
                nn = [v for v in nn if 0 <= v < len(src)]
                if len(nn) != 1:
                    break
                p, c = c, nn[0]; L += 1
            return L

        trail = [(start, src[start])]
        prev, cur, cur_dir = -1, start, None
        for step in range(n):
            nbrs = [int(v) for v in adj.get(cur, ()) if int(v) != prev]
            if step == 0 and exclude is not None:
                nbrs = [v for v in nbrs if v != exclude]
            nbrs = [v for v in nbrs if 0 <= v < len(src)]
            if not nbrs:
                break
            if len(nbrs) == 1:
                nxt = nbrs[0]
            elif cur_dir is None:
                # step 0: branch pointing most AWAY from the other pick
                # in 2D screen space; ties (≈⟂, |Δ|≤0.15) → longest chain.
                scored = []
                for n0 in nbrs:
                    sp, _dd = _world_to_screen(src[n0])
                    sc = None
                    if sp is not None and d_axis is not None:
                        dv = (np.asarray(sp, np.float64)
                              - np.asarray(this_s, np.float64))
                        dn = float(np.linalg.norm(dv))
                        if dn > 1e-9:
                            sc = float(np.dot(dv / dn, d_axis))
                    scored.append((n0, sc))
                valid = [(n0, sc) for (n0, sc) in scored if sc is not None]
                if valid:
                    min_sc = min(sc for (_n, sc) in valid)
                    cand = [n0 for (n0, sc) in valid
                            if sc <= min_sc + 0.15]
                    nxt = max(cand, key=lambda nb: _chain_len(nb, cur))
                else:
                    nxt = max(nbrs, key=lambda nb: _chain_len(nb, cur))
            else:
                best_n, best_dot = nbrs[0], -2.0
                for n0 in nbrs:
                    dv = src[n0] - src[cur]
                    nrm = float(np.linalg.norm(dv))
                    if nrm < 1e-9:
                        continue
                    dot = float(np.dot(dv / nrm, cur_dir))
                    if dot > best_dot:
                        best_dot, best_n = dot, n0
                nxt = best_n
            dv = src[nxt] - src[cur]
            nrm = float(np.linalg.norm(dv))
            if nrm > 1e-9:
                cur_dir = dv / nrm
            prev, cur = cur, nxt
            trail.append((cur, src[cur]))
        return trail

    def _project_trail(trail):
        out = []
        for _gx, _pw in trail:
            sp, _dd = _world_to_screen(_pw)
            if sp is None:
                break
            out.append((float(sp[0]), float(sp[1])))
        return out

    trail1_screen = _project_trail(_walk_back(g1, excl1, s1, s2))
    trail2_screen = _project_trail(_walk_back(g2, excl2, s2, s1))

    def _tangent_at_end(trail_sc, anchor):
        # Outgoing tangent at the pick = anchor − last back-point.
        if len(trail_sc) < 2:
            return None
        far = np.asarray(trail_sc[-1], np.float64)
        a   = np.asarray(anchor,        np.float64)
        t   = a - far
        n   = float(np.linalg.norm(t))
        return (t / n) if n > 1e-9 else None

    t1 = _tangent_at_end(trail1_screen, s1)
    t2 = _tangent_at_end(trail2_screen, s2)
    bezier_ctrl_suggest = None
    _hint_mode = 'n/a'
    if t1 is not None and t2 is not None:
        P0 = np.asarray(s1, np.float64)
        P2 = np.asarray(s2, np.float64)
        chord = float(np.linalg.norm(P2 - P0))
        # Intersect the tangent LINE through P0 (dir t1) with the one
        # through P2 (dir t2).  A quadratic Bézier's endpoint tangents are
        # parallel to (P1−P0) and (P2−P1), so that intersection is the
        # unique control point tangent to BOTH lines.  A negative a / b only
        # means the intersection sits behind the arrow, which is still a
        # valid tangent-hugging curve, so accept either sign and bail only
        # when it is absurdly far (≈parallel).
        A   = np.column_stack([t1, -t2])
        rhs = P2 - P0
        det = float(np.linalg.det(A))
        if abs(det) > 1e-6:
            ab = np.linalg.solve(A, rhs)
            a, b = float(ab[0]), float(ab[1])
            if abs(a) < 4 * chord and abs(b) < 4 * chord:
                bezier_ctrl_suggest = (float(P0[0] + a * t1[0]),
                                       float(P0[1] + a * t1[1]))
                _hint_mode = f'line-intersect (a={a:.0f}, b={b:.0f})'
        if bezier_ctrl_suggest is None:
            # ≈parallel tangents (no usable line intersection) — average
            # each tangent's 0.4·chord projection.  Respects tangent
            # direction; just can't match both lines simultaneously.
            alpha = max(0.4 * chord, 30.0)
            p1a = P0 + alpha * t1
            p1b = P2 + alpha * t2
            avg = (p1a + p1b) * 0.5
            bezier_ctrl_suggest = (float(avg[0]), float(avg[1]))
            _hint_mode = f'fallback-avg (α={alpha:.0f})'
        print(f'  [sketch-bridge] tangent hint: trails '
              f'{len(trail1_screen)}/{len(trail2_screen)} pts → '
              f'P1 = ({bezier_ctrl_suggest[0]:.0f}, '
              f'{bezier_ctrl_suggest[1]:.0f})  '
              f'[det={det:.2g}, {_hint_mode}]')
    else:
        print(f'  [sketch-bridge] tangent hint: trails '
              f'{len(trail1_screen)}/{len(trail2_screen)} pts (too '
              f'short for tangent — using default ⟂-midpoint init)')

    user_poly = _tk_sketch_pad(img, s1, s2,
                               trail1_screen=trail1_screen,
                               trail2_screen=trail2_screen,
                               bezier_ctrl_init=bezier_ctrl_suggest,
                               tangent1=t1, tangent2=t2)
    if user_poly is None or len(user_poly) < 1:
        print('  [sketch-bridge] cancelled / empty'); return

    # Auto-orient: the user may draw from s2 → s1 instead of s1 → s2.
    # Reverse the polyline if its first point is closer to s2 than s1,
    # so the endpoint-anchor blend below never produces a Z-shape.
    _up = np.asarray(user_poly, np.float64)
    _d_start_to_1 = np.linalg.norm(_up[0]  - np.asarray(s1))
    _d_start_to_2 = np.linalg.norm(_up[0]  - np.asarray(s2))
    if _d_start_to_2 < _d_start_to_1:
        _up = _up[::-1].copy()
        print(f'  [sketch-bridge] user drew #2→#1 '
              f'(start dist: to#1={_d_start_to_1:.0f}px, '
              f'to#2={_d_start_to_2:.0f}px) — auto-reversed')

    # ── 2D smoothing: kill mouse jitter with a 1D Gaussian in index
    # space.  Reflect-pad so the endpoints aren't pulled inward.
    def _gauss1d(x, sigma):
        n = len(x)
        if n < 3 or sigma < 0.5:
            return x.astype(np.float64).copy()
        radius = min(int(3 * sigma + 0.5), n - 1)
        if radius < 1:
            return x.astype(np.float64).copy()
        k = np.arange(-radius, radius + 1, dtype=np.float64)
        w = np.exp(-0.5 * (k / sigma) ** 2); w /= w.sum()
        pad = np.concatenate([x[radius:0:-1], x, x[-2:-radius-2:-1]])
        return np.convolve(pad, w, mode='valid').astype(np.float64)

    if len(_up) >= 3:
        sigma = max(2.0, min(8.0, 0.04 * len(_up)))   # ~4% of pts
        _up = np.column_stack([_gauss1d(_up[:, 0], sigma),
                               _gauss1d(_up[:, 1], sigma)])
    else:
        sigma = 0.0

    # ── Endpoint anchor via linear shift blend ──────────────────────
    # Instead of hard-prepending s1 / appending s2 (which creates a
    # corner at the junction), translate the smoothed curve so its
    # first point lands at s1 and its last lands at s2.  The shift is
    # lerp((s1 - q0), (s2 - qN), t) along chord-length, which preserves
    # the curve's original tangent at each end → kink-free junction.
    _s1a = np.asarray(s1, np.float64)
    _s2a = np.asarray(s2, np.float64)
    if len(_up) >= 2:
        seg = np.linalg.norm(np.diff(_up, axis=0), axis=1)
        chord = np.concatenate([[0.0], np.cumsum(seg)])
        tot = float(chord[-1])
        ts = (chord / tot) if tot > 1e-9 else np.zeros(len(_up))
        shift = ((1.0 - ts)[:, None] * (_s1a - _up[0])[None, :]
                 + ts[:, None]      * (_s2a - _up[-1])[None, :])
        poly = _up + shift
    else:
        poly = np.vstack([_s1a, _s2a])
    poly[0]  = _s1a   # kill float drift
    poly[-1] = _s2a
    print(f'  [sketch-bridge] 2D smooth: σ={sigma:.1f} idx → '
          f'{len(poly)} pts (linear-shift anchor at endpoints)')
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    total = float(seg.sum())
    if total < 1e-9:
        print('  [sketch-bridge] zero-length polyline'); return
    t = np.concatenate([[0.0], np.cumsum(seg) / total])      # (n,)
    depths = d1 + t * (d2 - d1)
    # Unproject all polyline pixels at once.
    V_inv = np.linalg.inv(V)
    ndc_x = 2.0 * poly[:, 0] / W_img - 1.0
    ndc_y = 1.0 - 2.0 * poly[:, 1] / H_img
    half_h = depths * tan_half
    half_w = aspect * half_h
    view_h = np.column_stack([ndc_x * half_w,
                              ndc_y * half_h,
                              -depths,
                              np.ones_like(depths)])
    world_h = view_h @ V_inv.T
    world_pts = world_h[:, :3]
    world_pts[0]  = p1                     # kill numeric drift
    world_pts[-1] = p2
    # Arc-length resample by step (interior pts only — endpoints are the picks)
    diffs = np.linalg.norm(np.diff(world_pts, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(diffs)])
    total_world = float(arc[-1])
    step = _TOPO_MST_CONNECT_STEP
    n_steps = max(1, int(round(total_world / step)))
    if n_steps >= 2:
        target_arcs = (np.arange(1, n_steps)
                       * (total_world / n_steps))
        interp_pts = np.column_stack([
            np.interp(target_arcs, arc, world_pts[:, k])
            for k in range(3)])
    else:
        interp_pts = np.zeros((0, 3), np.float64)
    # ── Same-MST path replacement ───────────────────────────────────
    # If g1 and g2 already share a component, the new chain would close a
    # cycle, so BFS the existing path and drop its EDGES (not the nodes —
    # side branches stay attached).  Interior nodes left at degree 0 are
    # pruned by min_pts in _topo_mst_viz.
    # EXCEPTION: if ≥2 red junctions lie ON that path, KEEP every edge
    # between the first and last of them.  That sub-path is a real
    # junction↔junction strand, not a dead stub; dropping only the two end
    # stubs still breaks the cycle.
    _path_n = len(old_path) if old_path else 0
    keep_lo = keep_hi = None
    if _path_n >= 2:
        _jct_set = {int(dd['gidx'])
                    for dd in (s.get('topo_mst_jct_info') or [])}
        _on_path = [i for i in range(_path_n)
                    if int(old_path[i]) in _jct_set]
        if len(_on_path) >= 2:
            keep_lo, keep_hi = _on_path[0], _on_path[-1]
    rm_edge_idx = [m for m in range(_path_n - 1)
                   if not (keep_lo is not None and keep_lo <= m < keep_hi)]
    # Degenerate: preserving would leave NO edge to cut (the only
    # junctions are the picks themselves) → can't keep the span without
    # a cycle, so fall back to cutting the whole path.
    if _path_n >= 2 and not rm_edge_idx:
        rm_edge_idx = list(range(_path_n - 1))
        keep_lo = keep_hi = None
    n_rm_edges   = len(rm_edge_idx)
    n_keep_edges = ((_path_n - 1) - n_rm_edges) if _path_n >= 2 else 0
    # Snapshot BEFORE any mutation so Undo restores edges + extras.
    _topo_stitch_snapshot(
        f'sketch-connect gidx{g1}↔gidx{g2} '
        f'({n_steps} steps, +{len(interp_pts)} pts'
        + (f', −{n_rm_edges} old edges' if n_rm_edges else '')
        + (f', keep {n_keep_edges} inter-JCT edges' if n_keep_edges else '')
        + ')')
    replace_msg = ''
    if n_rm_edges:
        for m in rm_edge_idx:
            iu, iv = int(old_path[m]), int(old_path[m + 1])
            if iu in adj:
                adj[iu].discard(iv)
            if iv in adj:
                adj[iv].discard(iu)
        replace_msg = (f' [same MST → REPLACING old path: −{n_rm_edges} '
                       f'edge(s)')
        if n_keep_edges:
            replace_msg += (f', KEPT {n_keep_edges}-edge inter-JCT strand '
                            f'gidx{int(old_path[keep_lo])}↔'
                            f'gidx{int(old_path[keep_hi])}')
        replace_msg += ']'
    new_start_gidx = N_pts + len(ex_arr)
    new_ex_arr = (np.vstack([ex_arr, interp_pts])
                  if len(interp_pts) else ex_arr)
    s['topo_mst_extra_pts'] = (None if len(new_ex_arr) == 0
                               else new_ex_arr)
    chain = ([g1]
             + [new_start_gidx + i for i in range(len(interp_pts))]
             + [g2])
    for a, b in zip(chain[:-1], chain[1:]):
        ia, ib = int(a), int(b)
        adj.setdefault(ia, set()).add(ib)
        adj.setdefault(ib, set()).add(ia)
    print(f'  [sketch-bridge] gidx{g1}↔gidx{g2}  drew {len(user_poly)} '
          f'sketch pts → arc-length world {total_world:.2f} vox, '
          f'step={step:.2f} → +{len(interp_pts)} synth bridge pts.'
          f'{replace_msg}'
          + '  Refresh via cached adj.  Undo via "Undo stitch".')
    _repair_stats_bump('B', g1, g2, new_start_gidx, len(interp_pts))
    _topo_mst_clickdel_clear()
    jct_before = {int(d['gidx']) for d in (s.get('topo_mst_jct_info') or [])}
    try:
        _topo_mst_viz(use_cached_adj=True)
        _report_new_junctions(jct_before)
    except Exception as exn:
        print(f'  [sketch-bridge] refresh warn: {exn}')
    # auto-checkpoint after a manual sketch — refreshes
    # seg_state_latest.npz only (no timestamped backup).
    try:
        _save_seg_state(timestamped=False, quiet=True)
        print('  [seg-save] auto → seg_state_latest.npz')
    except Exception as exn:
        print(f'  [seg-save] auto warn: {exn}')


def _tk_sketch_pad(img: np.ndarray,
                   p1_screen, p2_screen,
                   trail1_screen=None, trail2_screen=None,
                   bezier_ctrl_init=None,
                   tangent1=None, tangent2=None):
    """Pop up a modal tk window with `img` as background and the two pick
    endpoints highlighted (yellow=#1, cyan=#2).

    `trail1_screen` / `trail2_screen` are optional MST walk-back trails in
    image-pixel space, drawn dashed in the matching pick colour so the
    local MST direction is visible at each endpoint.  `bezier_ctrl_init`
    overrides the default ⟂-midpoint control point.

    Two modes (top-bar button or `M`): FREEHAND draws a polyline on LEFT
    drag; BEZIER drags the orange control point of a quadratic with P0=#1,
    P2=#2.

    Controls:
        LEFT drag       — draw (freehand) / move control point (bezier)
        WHEEL           — zoom around cursor (1× … 5×, IN only)
        RIGHT drag      — pan (only when zoomed > 1×)
        Confirm / ENTER — accept and return the curve
        Cancel / ESC    — abort (also window close)
        Mode / M        — switch freehand ↔ bezier
        C               — clear the current draft
        0 / R           — reset view to 1× / origin

    Returns list[(sx, sy)] in IMAGE-pixel coords, or None on cancel."""
    try:
        import tkinter as tk
        from tkinter import ttk
        from PIL import Image, ImageTk
    except ImportError as exn:
        print(f'  [sketch-pad] tkinter / Pillow missing: {exn}')
        return None
    H, W = img.shape[:2]
    max_side = 1400
    scale0 = min(1.0, max_side / float(max(W, H)))   # fit-to-window
    Wv = int(round(W * scale0)); Hv = int(round(H * scale0))

    rgba = img if (img.ndim == 3 and img.shape[2] == 4) else np.concatenate(
        [img.reshape(H, W, -1)[:, :, :3],
         255 * np.ones((H, W, 1), np.uint8)], axis=2)
    pil_full = Image.fromarray(rgba)   # cached source for fast re-resize

    root = tk.Tk()
    root.title('Sketch bridge — LEFT draw / drag · WHEEL zoom · '
               'RIGHT pan · Confirm/Cancel buttons or ENTER/ESC · M=mode')

    # ── Top bar ─────────────────────────────────────────────────────
    Z_MIN, Z_MAX = 1.0, 5.0
    top = tk.Frame(root)
    top.pack(side=tk.TOP, fill=tk.X)
    mode = ['bezier']              # default = bezier (M to toggle)

    def _on_mode_toggle(_e=None):
        new_m = 'bezier' if mode[0] == 'freehand' else 'freehand'
        _set_mode(new_m)

    def _accept(_e=None):
        # finalize: in bezier mode, sample the curve into poly_img
        if mode[0] == 'bezier':
            curve = _sample_bezier()
            poly_img.clear()
            poly_img.extend(curve)
        if len(poly_img) >= 2:
            accepted[0] = True
            root.destroy()
        else:
            # ignore confirm on empty (give visual feedback?)
            pass

    def _cancel(_e=None):
        accepted[0] = False
        root.destroy()

    _UI_FONT_BTN = ('Sans', 14, 'bold')
    _UI_FONT_LBL = ('Sans', 13)
    _UI_FONT_DIM = ('Sans', 11)
    _UI_PADX, _UI_PADY = 6, 6
    mode_var = tk.StringVar(value='Mode: bezier  (M)')
    btn_mode = tk.Button(top, textvariable=mode_var, width=18,
                         font=_UI_FONT_BTN, padx=8, pady=4,
                         command=_on_mode_toggle)
    btn_mode.pack(side=tk.LEFT, padx=_UI_PADX, pady=_UI_PADY)
    tk.Button(top, text='Confirm  [Enter]', width=16,
              bg='#3a7', fg='white',
              font=_UI_FONT_BTN, padx=8, pady=4,
              command=_accept).pack(side=tk.LEFT,
                                    padx=_UI_PADX, pady=_UI_PADY)
    tk.Button(top, text='Cancel  [Esc]', width=14,
              bg='#a33', fg='white',
              font=_UI_FONT_BTN, padx=8, pady=4,
              command=_cancel).pack(side=tk.LEFT,
                                    padx=_UI_PADX, pady=_UI_PADY)
    tk.Frame(top, width=30).pack(side=tk.LEFT)  # spacer
    zoom_var = tk.StringVar(value='Zoom: 1.00×')
    tk.Label(top, textvariable=zoom_var, font=_UI_FONT_LBL,
             width=12, anchor='w').pack(side=tk.LEFT, padx=4)
    # ttk styled progress bar — bump its height too.
    _pb_style = ttk.Style()
    try:
        _pb_style.theme_use(_pb_style.theme_use())
        _pb_style.configure('Sketch.Horizontal.TProgressbar',
                            thickness=22)
    except Exception:
        pass
    zoom_bar = ttk.Progressbar(
        top, orient='horizontal', mode='determinate',
        maximum=Z_MAX, length=260,
        style='Sketch.Horizontal.TProgressbar')
    zoom_bar['value'] = Z_MIN
    zoom_bar.pack(side=tk.LEFT, padx=4, pady=_UI_PADY)
    tk.Label(top, text=f'(min {Z_MIN:.0f}× · max {Z_MAX:.0f}×)',
             font=_UI_FONT_DIM, fg='#888').pack(side=tk.LEFT, padx=4)

    canvas = tk.Canvas(root, width=Wv, height=Hv, bg='black',
                       highlightthickness=0)
    canvas.pack()

    # ── View / draw state (image-pixel space) ───────────────────────
    zoom    = [1.0]
    off_x   = [0.0]
    off_y   = [0.0]
    cached  = {'zoom': -1.0, 'photo': None}

    poly_img: list = []        # FREEHAND polyline (image coords)
    line_ids: list = []        # canvas line item ids
    drawing = [False]
    accepted = [False]
    pan_anchor: list = [None]
    # Bezier mode state — quadratic; P0=#1, P2=#2, P1=draggable.
    bezier_ctrl: list = [None]    # (ix, iy) image coords for P1
    bezier_drag = [False]
    bezier_ids: list = []

    def _ts():
        return scale0 * zoom[0]

    def _img_to_canvas(px, py):
        t = _ts()
        return (px * t + off_x[0], py * t + off_y[0])

    def _canvas_to_img(cx, cy):
        t = _ts()
        if t < 1e-9:
            return (0.0, 0.0)
        return ((cx - off_x[0]) / t, (cy - off_y[0]) / t)

    def _get_photo():
        if abs(cached['zoom'] - zoom[0]) > 1e-6 or cached['photo'] is None:
            t = _ts()
            Wz = max(1, int(round(W * t)))
            Hz = max(1, int(round(H * t)))
            pil_z = pil_full.resize((Wz, Hz), Image.BILINEAR)
            cached['photo'] = ImageTk.PhotoImage(pil_z)
            cached['zoom']  = zoom[0]
        return cached['photo']

    def _sample_bezier(n: int = 80):
        """Sample the current quadratic Bezier into `n` IMAGE-pixel
        points P0→P2 via P1.  Used both for redraw and for the final
        polyline returned on confirm."""
        if bezier_ctrl[0] is None:
            return []
        P0 = np.asarray(p1_screen, np.float64)
        P2 = np.asarray(p2_screen, np.float64)
        P1 = np.asarray(bezier_ctrl[0], np.float64)
        ts = np.linspace(0.0, 1.0, n)
        one = 1.0 - ts
        pts = ((one * one)[:, None] * P0
               + (2.0 * one * ts)[:, None] * P1
               + (ts * ts)[:, None] * P2)
        return [(float(x), float(y)) for (x, y) in pts]

    def _redraw_all():
        canvas.delete('all')
        line_ids.clear()
        bezier_ids.clear()
        photo = _get_photo()
        canvas.create_image(off_x[0], off_y[0],
                            anchor=tk.NW, image=photo)
        canvas._photo_keep = photo
        # ── MST walk-back trails (tangent hint) ─────────────────────
        # Dashed polyline + small dots in the matching pick colour, so
        # the local MST direction at each endpoint is visible while
        # the user adjusts the Bezier control point.
        for trail_sc, fill in ((trail1_screen, 'yellow'),
                                (trail2_screen, 'cyan')):
            if trail_sc and len(trail_sc) >= 2:
                cpts = [_img_to_canvas(px, py) for (px, py) in trail_sc]
                for i in range(len(cpts) - 1):
                    x0, y0 = cpts[i]; x1, y1 = cpts[i + 1]
                    canvas.create_line(x0, y0, x1, y1,
                                       fill=fill, width=2,
                                       dash=(5, 3))
                rd = 4
                for (cx, cy) in cpts:
                    canvas.create_oval(cx - rd, cy - rd,
                                       cx + rd, cy + rd,
                                       fill=fill, outline='black',
                                       width=1)
        r = 11
        for psc, fill, lbl in ((p1_screen, 'yellow', '#1'),
                               (p2_screen, 'cyan',   '#2')):
            cx, cy = _img_to_canvas(psc[0], psc[1])
            canvas.create_oval(cx-r, cy-r, cx+r, cy+r,
                               fill=fill, outline='black', width=2)
            canvas.create_text(cx + r + 6, cy, anchor=tk.W,
                               text=lbl, fill=fill,
                               font=('Sans', 16, 'bold'))
        # ── Tangent-hint arrows ─────────────────────────────────────
        # Solid arrow from each pick along its computed continuation
        # tangent (the direction the bridge should leave that pick) —
        # drawn black-under-colour so it reads on any background.
        TANG_LEN = 130.0   # image px
        for psc, tang, fill in ((p1_screen, tangent1, 'yellow'),
                                (p2_screen, tangent2, 'cyan')):
            if tang is None:
                continue
            ex = psc[0] + TANG_LEN * float(tang[0])
            ey = psc[1] + TANG_LEN * float(tang[1])
            x0, y0 = _img_to_canvas(psc[0], psc[1])
            x1, y1 = _img_to_canvas(ex, ey)
            canvas.create_line(x0, y0, x1, y1, fill='black', width=7,
                               arrow=tk.LAST, arrowshape=(20, 24, 8))
            canvas.create_line(x0, y0, x1, y1, fill=fill, width=3,
                               arrow=tk.LAST, arrowshape=(18, 22, 6))
        if mode[0] == 'freehand':
            if len(poly_img) >= 2:
                cpts = [_img_to_canvas(px, py) for (px, py) in poly_img]
                for i in range(len(cpts) - 1):
                    x0, y0 = cpts[i]; x1, y1 = cpts[i + 1]
                    line_ids.append(canvas.create_line(
                        x0, y0, x1, y1, fill='magenta', width=3))
        else:
            # bezier: draw the curve + control polygon + draggable handle
            curve = _sample_bezier()
            if len(curve) >= 2:
                cpts = [_img_to_canvas(x, y) for (x, y) in curve]
                for i in range(len(cpts) - 1):
                    x0, y0 = cpts[i]; x1, y1 = cpts[i + 1]
                    bezier_ids.append(canvas.create_line(
                        x0, y0, x1, y1, fill='magenta', width=3))
            if bezier_ctrl[0] is not None:
                # dashed guides P0—P1, P1—P2 to make the handle obvious
                for ep in (p1_screen, p2_screen):
                    ex, ey = _img_to_canvas(ep[0], ep[1])
                    cx, cy = _img_to_canvas(*bezier_ctrl[0])
                    bezier_ids.append(canvas.create_line(
                        ex, ey, cx, cy,
                        fill='#888', width=1, dash=(4, 3)))
                cx, cy = _img_to_canvas(*bezier_ctrl[0])
                R = 14
                bezier_ids.append(canvas.create_oval(
                    cx-R, cy-R, cx+R, cy+R,
                    fill='orange', outline='red', width=3))
                bezier_ids.append(canvas.create_text(
                    cx + R + 6, cy, anchor=tk.W,
                    text='ctrl (drag)', fill='orange',
                    font=('Sans', 15, 'bold')))

    def _init_bezier_ctrl():
        # Tangent-hint init (from MST walk-back) takes priority; only
        # fall back to the perpendicular-midpoint heuristic when no
        # hint was provided or the tangents were degenerate.
        if bezier_ctrl_init is not None:
            bezier_ctrl[0] = (float(bezier_ctrl_init[0]),
                              float(bezier_ctrl_init[1]))
            return
        P0 = np.asarray(p1_screen, np.float64)
        P2 = np.asarray(p2_screen, np.float64)
        mid = (P0 + P2) * 0.5
        d = P2 - P0
        n = float(np.linalg.norm(d))
        if n > 1e-9:
            perp = np.array([-d[1], d[0]]) / n * (n * 0.3)
        else:
            perp = np.array([0.0, 60.0])
        bezier_ctrl[0] = (float(mid[0] + perp[0]),
                          float(mid[1] + perp[1]))

    def _set_mode(new_m: str):
        if new_m == mode[0]:
            return
        mode[0] = new_m
        if new_m == 'bezier' and bezier_ctrl[0] is None:
            _init_bezier_ctrl()
        mode_var.set(f'Mode: {new_m}  (M)')
        # drawing state from the other mode is left intact so user can
        # switch back and forth without losing work.
        drawing[0] = False
        bezier_drag[0] = False
        _redraw_all()

    def _press(e):
        if mode[0] == 'freehand':
            for lid in line_ids:
                canvas.delete(lid)
            line_ids.clear()
            poly_img.clear()
            drawing[0] = True
            ix, iy = _canvas_to_img(e.x, e.y)
            poly_img.append((ix, iy))
        else:
            ix, iy = _canvas_to_img(e.x, e.y)
            bezier_ctrl[0] = (ix, iy)
            bezier_drag[0] = True
            _redraw_all()

    def _motion(e):
        if mode[0] == 'freehand':
            if not drawing[0]:
                return
            ix, iy = _canvas_to_img(e.x, e.y)
            prev_ix, prev_iy = poly_img[-1]
            cx0, cy0 = _img_to_canvas(prev_ix, prev_iy)
            cx1, cy1 = _img_to_canvas(ix, iy)
            line_ids.append(canvas.create_line(
                cx0, cy0, cx1, cy1, fill='magenta', width=3))
            poly_img.append((ix, iy))
        else:
            if not bezier_drag[0]:
                return
            ix, iy = _canvas_to_img(e.x, e.y)
            bezier_ctrl[0] = (ix, iy)
            _redraw_all()    # bezier curve is global → full redraw

    def _release(_e):
        drawing[0] = False
        bezier_drag[0] = False

    def _clamp_offset():
        t = _ts()
        Wz, Hz = W * t, H * t
        if Wz <= Wv:
            off_x[0] = (Wv - Wz) * 0.5
        else:
            off_x[0] = min(0.0, max(Wv - Wz, off_x[0]))
        if Hz <= Hv:
            off_y[0] = (Hv - Hz) * 0.5
        else:
            off_y[0] = min(0.0, max(Hv - Hz, off_y[0]))

    def _update_zoom_widget():
        zoom_var.set(f'Zoom: {zoom[0]:.2f}×')
        zoom_bar['value'] = max(Z_MIN, min(Z_MAX, zoom[0]))

    def _zoom_at(e, factor):
        z_old = zoom[0]
        z_new = max(Z_MIN, min(Z_MAX, z_old * factor))
        if abs(z_new - z_old) < 1e-6:
            return
        ix, iy = _canvas_to_img(e.x, e.y)
        zoom[0] = z_new
        t = _ts()
        off_x[0] = e.x - ix * t
        off_y[0] = e.y - iy * t
        _clamp_offset()
        _update_zoom_widget()
        _redraw_all()

    def _on_wheel(e):
        d = getattr(e, 'delta', 0)
        if d > 0:
            _zoom_at(e, 1.2)
        elif d < 0:
            _zoom_at(e, 1 / 1.2)

    def _on_button4(e): _zoom_at(e, 1.2)
    def _on_button5(e): _zoom_at(e, 1 / 1.2)

    def _pan_press(e):
        if zoom[0] <= Z_MIN + 1e-6:
            return
        pan_anchor[0] = (e.x, e.y, off_x[0], off_y[0])

    def _pan_motion(e):
        if pan_anchor[0] is None:
            return
        ex0, ey0, ox0, oy0 = pan_anchor[0]
        off_x[0] = ox0 + (e.x - ex0)
        off_y[0] = oy0 + (e.y - ey0)
        _clamp_offset()
        _redraw_all()

    def _pan_release(_e):
        pan_anchor[0] = None

    def _reset(_e=None):
        zoom[0] = 1.0
        off_x[0] = 0.0
        off_y[0] = 0.0
        _clamp_offset()
        _update_zoom_widget()
        _redraw_all()

    def _clear(_e=None):
        if mode[0] == 'freehand':
            poly_img.clear()
            for lid in line_ids:
                canvas.delete(lid)
            line_ids.clear()
        else:
            _init_bezier_ctrl()
            _redraw_all()

    canvas.bind('<ButtonPress-1>',    _press)
    canvas.bind('<B1-Motion>',        _motion)
    canvas.bind('<ButtonRelease-1>',  _release)
    canvas.bind('<ButtonPress-3>',    _pan_press)
    canvas.bind('<B3-Motion>',        _pan_motion)
    canvas.bind('<ButtonRelease-3>',  _pan_release)
    canvas.bind('<MouseWheel>',       _on_wheel)
    canvas.bind('<Button-4>',         _on_button4)
    canvas.bind('<Button-5>',         _on_button5)
    root.bind('<Return>',             _accept)
    root.bind('<Escape>',             _cancel)
    root.bind('<c>',                  _clear)
    root.bind('<C>',                  _clear)
    root.bind('<0>',                  _reset)
    root.bind('<r>',                  _reset)
    root.bind('<R>',                  _reset)
    root.bind('<m>',                  _on_mode_toggle)
    root.bind('<M>',                  _on_mode_toggle)
    root.protocol('WM_DELETE_WINDOW', _cancel)

    _clamp_offset()
    _update_zoom_widget()
    if mode[0] == 'bezier' and bezier_ctrl[0] is None:
        _init_bezier_ctrl()
    _redraw_all()
    root.mainloop()

    if not accepted[0] or len(poly_img) < 2:
        return None
    return [(float(x), float(y)) for (x, y) in poly_img]


def _topo_mst_clickdel_apply(mode: str) -> None:
    """Apply one of three deletes to the currently-picked `topo_mst`
    node.  No-op if nothing is picked or the MST adj cache is missing.
    Snapshots so 'Undo stitch' rolls it back.

        A: bury every node within ±R-vox MST-walk distance from pick
        B: bury the SHORTER side (clicked node kept)
        C: bury the ENTIRE connected MST component the pick belongs to"""
    s = state
    P = s.get('pts')
    if P is None:
        return
    N_pts = len(P)
    pn = int(s.get('topo_mst_pick_node', -1))
    g0 = int(s.get('topo_mst_pick_gidx', -1))
    adj = s.get('topo_mst_global_adj')
    if pn < 0 or g0 < 0 or adj is None:
        print('  [topo-mst-click-del] no pick / no MST cache')
        return
    rm: set = set()
    if mode == 'A':
        # Vox-radius BFS along the MST: include every node whose
        # CUMULATIVE edge-length distance from g0 (walked along the
        # tree) is ≤ _TOPO_MST_CLICKDEL_VOX.  Uses positions from the
        # pts ∪ extras stack so synthetic centerline gidx work too.
        R = float(_TOPO_MST_CLICKDEL_VOX)
        pts = np.asarray(P, np.float64)
        ex = s.get('topo_mst_extra_pts')
        ex_arr = (np.zeros((0, 3), np.float64) if ex is None
                  else np.asarray(ex, np.float64))
        src = (np.vstack([pts, ex_arr]) if len(ex_arr) else pts)
        if not (0 <= g0 < len(src)):
            print(f'  [topo-mst-click-del A] gidx={g0} out of src '
                  f'range (len={len(src)})'); return
        dist = {g0: 0.0}
        q = [g0]; head = 0
        max_d_seen = 0.0
        while head < len(q):
            u = q[head]; head += 1
            du = dist[u]
            for v in adj.get(int(u), ()):
                iv = int(v)
                if iv in dist:
                    continue
                if not (0 <= iv < len(src)):
                    continue
                dv = du + float(np.linalg.norm(src[iv] - src[u]))
                if dv <= R:
                    dist[iv] = dv
                    if dv > max_d_seen:
                        max_d_seen = dv
                    q.append(iv)
        rm = {int(g) for g in dist if 0 <= int(g) < N_pts}
        label = (f'A click-del topo_mst #{pn} gidx={g0} '
                 f'±{R:.1f} vox ({len(rm)} nodes)')
        print(f'  [topo-mst-click-del A] node #{pn} (gidx={g0}) '
              f'±{R:.1f}-vox neighbourhood (MST-walk dist) → '
              f'buried {len(rm)} node(s)  (max reached '
              f'{max_d_seen:.2f} vox)')
    elif mode == 'B':
        neigh = [int(v) for v in adj.get(int(g0), ())]
        if not neigh:
            print(f'  [topo-mst-click-del B] node #{pn} (gidx={g0}) '
                  f'is isolated; nothing to delete')
            return
        subtrees = []
        for n0 in neigh:
            seen = {g0, n0}
            q = [n0]; head = 0
            while head < len(q):
                u = q[head]; head += 1
                for v in adj.get(int(u), ()):
                    v = int(v)
                    if v in seen:
                        continue
                    seen.add(v); q.append(v)
            sub = seen - {g0}
            subtrees.append((len(sub), n0, sub))
        subtrees.sort(key=lambda x: x[0])
        sizes = [t[0] for t in subtrees]
        sz, n0, sub = subtrees[0]
        rm = {int(g) for g in sub if 0 <= int(g) < N_pts}
        label = (f'B click-del topo_mst #{pn} gidx={g0} short side '
                 f'({sz} nodes, dir→gidx{n0})')
        print(f'  [topo-mst-click-del B] node #{pn} (gidx={g0})  '
              f'deg={len(neigh)}  side sizes={sizes}  → shortest dir '
              f'gidx{n0} ({sz} nodes) buried; clicked node kept')
    elif mode == 'C':
        # ── Mode C: bury the ENTIRE connected MST component ─────
        # BFS from the picked gidx across all adj edges → one
        # connected component (the whole isolated yarn fragment).
        # `rm` includes synthetic gidx (≥ N_pts) too so the adj-
        # disconnect loop strips their edges as well; the
        # topo_mst_deleted set silently filters them at burial
        # time (only real cloud gidx ever get buried in seg_src).
        cc = {int(g0)}
        q = [int(g0)]; head = 0
        while head < len(q):
            u = q[head]; head += 1
            for v in adj.get(int(u), ()):
                iv = int(v)
                if iv in cc:
                    continue
                cc.add(iv); q.append(iv)
        rm = set(int(g) for g in cc)
        n_real = sum(1 for g in cc if 0 <= int(g) < N_pts)
        label = (f'C click-del topo_mst #{pn} gidx={g0} whole '
                 f'component ({len(cc)} nodes, {n_real} real)')
        print(f'  [topo-mst-click-del C] node #{pn} (gidx={g0}) → '
              f'whole connected MST component: {len(cc)} node(s) '
              f'({n_real} real + {len(cc) - n_real} synthetic)')
    else:
        print(f'  [topo-mst-click-del] unknown mode {mode!r}')
        return
    if not rm:
        print('  [topo-mst-click-del] resolved to 0 cloud pts; skip')
        _topo_mst_clickdel_clear()
        return
    # Snapshot BEFORE any mutation so 'Undo stitch' restores the adj edits
    # AND the deleted-set entry.  Then disconnect rm in the LIVE cached adj
    # (in-place, like arm-cut / loop / accept, so earlier adj-only edits
    # survive instead of being proximity-reconnected by a rebuild), union rm
    # into topo_mst_deleted so the next full rebuild keeps them buried, and
    # refresh with use_cached_adj=True.
    _topo_stitch_snapshot(label)
    adj_live = s.get('topo_mst_global_adj') or {}
    n_cut = 0
    for g in rm:
        ig = int(g)
        if ig not in adj_live:
            continue
        for nb in list(adj_live[ig]):
            inb = int(nb)
            if inb in adj_live:
                adj_live[inb].discard(ig)
            n_cut += 1
        adj_live[ig] = set()
    _del = s.setdefault('topo_mst_deleted', set())
    _del |= rm
    print(f'    (cleared ~{n_cut} adj entries; total deleted='
          f'{len(_del)}; refreshing from cached adj — prior edits '
          f'preserved.  Undo via "Undo stitch".)')
    _topo_mst_clickdel_clear()
    try:
        _topo_mst_viz(use_cached_adj=True)
    except Exception as _ex:
        print(f'  [topo-mst-click-del] refresh warn: {_ex}')


def _topo_mst_keep_largest_comp() -> None:
    """Keep ONLY the largest connected MST component; bury every node in
    all other components.  Largest = most nodes.  No pick needed.
    Snapshot-undoable ("Undo stitch").  Same in-place adj edit + bury
    pattern as the click-deletes, so prior edits stay alive."""
    s = state
    P = s.get('pts')
    adj = s.get('topo_mst_global_adj')
    if P is None or adj is None:
        print('  [keep-largest] no MST cache — run "Topo MST viz" first')
        return
    N_pts = len(P)
    # ── flood every connected component over the cached adj ──
    comp_of: dict = {}
    comps: list = []
    for seed in list(adj.keys()):
        s0 = int(seed)
        if s0 in comp_of:
            continue
        idx = len(comps)
        cc = {s0}; comp_of[s0] = idx
        q = [s0]; head = 0
        while head < len(q):
            u = q[head]; head += 1
            for v in adj.get(int(u), ()):
                iv = int(v)
                if iv not in comp_of:
                    comp_of[iv] = idx; cc.add(iv); q.append(iv)
        comps.append(cc)
    if not comps:
        print('  [keep-largest] empty MST adj — nothing to do')
        return
    sizes = [len(c) for c in comps]
    n_keep = 1                                     # keep only the largest comp
    order = sorted(range(len(comps)), key=lambda i: sizes[i], reverse=True)
    keep_idxs = set(order[:n_keep])
    keep: set = set()
    for i in keep_idxs:
        keep |= comps[i]
    if len(comps) <= n_keep:
        print(f'  [keep-largest] only {len(comps)} component(s) ≤ N={n_keep} '
              f'— nothing to delete')
        return
    rm: set = set()
    for i, c in enumerate(comps):
        if i not in keep_idxs:
            rm |= c
    n_real_keep = sum(1 for g in keep if 0 <= int(g) < N_pts)
    n_real_rm   = sum(1 for g in rm   if 0 <= int(g) < N_pts)
    top = sorted(sizes, reverse=True)[:6]
    label = (f'keep-{n_keep}-largest MST comp(s) ({len(keep_idxs)} kept, '
             f'{len(keep)} nodes / {n_real_keep} real); buried '
             f'{len(comps) - len(keep_idxs)} other comp(s), {len(rm)} '
             f'nodes / {n_real_rm} real')
    _topo_stitch_snapshot(label)
    adj_live = s.get('topo_mst_global_adj') or {}
    n_cut = 0
    for g in rm:
        ig = int(g)
        if ig not in adj_live:
            continue
        for nb in list(adj_live[ig]):
            inb = int(nb)
            if inb in adj_live:
                adj_live[inb].discard(ig)
            n_cut += 1
        adj_live[ig] = set()
    _del = s.setdefault('topo_mst_deleted', set())
    _del |= rm
    print(f'  [keep-largest] {len(comps)} components (top sizes {top}); '
          f'kept {len(keep_idxs)} largest ({len(keep)} nodes, {n_real_keep} '
          f'real); buried {len(rm)} nodes ({n_real_rm} real) across '
          f'{len(comps) - len(keep_idxs)} comp(s); ~{n_cut} adj entries '
          f'cleared.  Undo via "Undo stitch".')
    _topo_mst_clickdel_clear()
    try:
        _topo_mst_viz(use_cached_adj=True)
    except Exception as _ex:
        print(f'  [keep-largest] refresh warn: {_ex}')


def _clear_curves() -> None:
    global _curve_data
    if ps.has_curve_network('curves'):
        ps.remove_curve_network('curves')
    _curve_data = []
    state['curve_seg_labels'] = None
    state['curve_K']          = 0
    state['curve_sub_to_cc']  = {}
    state['curve_node_sub']   = None
    state['curve_node_local'] = None
    state['curve_edges']      = None
    _clear_curve_selection()
    _maybe_show_orig_viz()


def _clear_curve_selection() -> None:
    if ps.has_point_cloud('curve_selection'):
        ps.remove_point_cloud('curve_selection')
    if ps.has_curve_network('curve_sel'):
        ps.remove_curve_network('curve_sel')
    state['curve_selected_cc'] = -1
    state['curve_selected_idx'] = None


def _clear_all_selection() -> None:
    """Right-click: drop EVERY selection — picked sticks/cyan AND the
    clicked-curve highlight (curve_sel)."""
    s = state
    _clear_curve_selection()                 # curve_sel net + cloud
    s['curve_last_clicked_sub'] = -1
    s['curve_last_clicked_node'] = -1
    # reset red/green click change-gates so a fresh click prints again
    s['topo_mst_last_jct']  = -1
    s['topo_mst_last_leaf'] = -1
    # drop the per-branch explorer + r=5vox sphere + closest-to-sphere
    # markers + before-merge halo + arm-connect leaf/bridge markers
    # (all red-ball click viz)
    _clear_topo_branch_click_viz()
    print('  [select] right-click → ALL selections cleared '
          '(curve, topo click viz)')


def _delete_curve() -> None:
    """Delete the last-clicked sub-curve: drop it from _curve_data,
    discard its CC's points (label −1 so a re-fit won't bring
    it back), rebuild the 'curves' network, clear stale selection /
    endpoint-merge recs."""
    global _curve_data
    s = state
    sub = int(s.get('curve_last_clicked_sub', -1))
    if sub < 0:
        print('  [del] click a curve first'); return
    entry = next((i for i, (sid, _) in enumerate(_curve_data)
                  if sid == sub), None)
    if entry is None:
        print(f'  [del] sub {sub} not in _curve_data — re-click'); return
    _curve_data.pop(entry)
    sub_to_cc = dict(s.get('curve_sub_to_cc') or {})
    cc = sub_to_cc.pop(sub, -1)
    labels = s.get('curve_seg_labels')
    n_pts = 0
    if labels is not None and cc is not None and cc >= 0:
        labels = np.asarray(labels).copy()
        m = labels == cc
        n_pts = int(m.sum())
        labels[m] = -1                       # discard those points
        s['curve_seg_labels'] = labels
    s['curve_sub_to_cc'] = sub_to_cc
    Ksub = max([sid for sid, _ in _curve_data], default=-1) + 1
    _register_curves(_curve_data, max(Ksub, 1))
    s['curve_last_clicked_sub'] = -1
    s['curve_last_clicked_node'] = -1
    _clear_curve_selection()
    print(f'  [del] removed sub{sub} (CC {cc}, {n_pts} pts '
          f'discarded); {len(_curve_data)} curves left.')


def _update_viz() -> None:
    pc.update_point_positions(state['pts'])
    _dir_nodes_new = np.vstack(
        [state['pts'], state['pts'] + DIR_LEN * state['dirs']]).astype(np.float32)
    ps.get_curve_network('direction_lines').update_node_positions(_dir_nodes_new)
    _refresh_datapt_colors(state['dirs'])
    _refresh_picked(state['picked'])
    # Refresh inspect viz so bw ball + kNN highlight follow the new state
    _ipi = int(state.get('inspect_picked_idx', -1))
    if _ipi >= 0:
        _inspect_picked(_ipi)


def _sec(label: str, opn: bool = False) -> bool:
    """Collapsible UI section header. opn=True → expanded on first use."""
    if opn:
        psim.SetNextItemOpen(True, psim.ImGuiCond_FirstUseEver)
    return psim.CollapsingHeader(label)


def _firebrick_button(label: str) -> bool:
    """Button in firebrick — marks the "run this stage" action of each
    section (Run, Build Topology MST, Fit curves).  The
    click result is taken before the pops so the three pushes are always
    matched, whatever the button returns."""
    psim.PushStyleColor(psim.ImGuiCol_Button,        (0.70, 0.13, 0.13, 1.0))
    psim.PushStyleColor(psim.ImGuiCol_ButtonHovered, (0.80, 0.20, 0.20, 1.0))
    psim.PushStyleColor(psim.ImGuiCol_ButtonActive,  (0.55, 0.10, 0.10, 1.0))
    clicked = psim.Button(label)
    psim.PopStyleColor()
    psim.PopStyleColor()
    psim.PopStyleColor()
    return clicked


def callback():
    s = state

    # ── topo_mst viz radius (TOP-LEVEL, always visible) ───────────────────
    # polyscope's OWN built-in radius slider (in the topo_mst structure panel)
    # is a RELATIVE slider hard-capped at 0.1 (debug confirmed: dragging it
    # only goes 0→0.1, snapping thin) and there's no API to widen it.  So we
    # drive the radius ourselves: an ABSOLUTE-vox slider (wide, logarithmic),
    # RE-ASSERTED every frame so it overrides the built-in slider.  USE THIS
    # one (the built-in will just revert).
    if ps.has_curve_network('topo_mst'):
        _vr_chg, _vr_val = psim.SliderFloat(
            'topo_mst viz radius (vox)  [USE THIS — built-in slider is capped]',
            float(s.get('topo_mst_viz_radius', 0.90)), 0.1, 100.0,
            format='%.2f', flags=psim.ImGuiSliderFlags_Logarithmic)
        if _vr_chg:
            s['topo_mst_viz_radius'] = float(_vr_val)
        _cn_mst = ps.get_curve_network('topo_mst')
        _cn_mst.set_radius(float(s.get('topo_mst_viz_radius', 0.90)),
                           relative=False)

    # Click pick:
    #  - `points`           → inspect (bw ellipsoid + kNN highlight)
    try:
        if ps.have_selection():
            sel = ps.get_selection()
            if sel.is_hit:
                struct = sel.structure_name
                picked = int(sel.local_index)
                if struct == 'points' and 0 <= picked < N:
                    if picked != s['picked']:
                        _refresh_picked(picked)
                        print(f'\n=== picked idx {picked} ===')
                    s['inspect_picked_idx'] = picked
                    _inspect_picked(picked)
                elif struct == 'curves':
                    # Single merged network: map the picked node/edge back
                    # to (sub_idx, local node) via the stored maps, then
                    # let the current MODE act (idle/select/split/pair).
                    nsub = s.get('curve_node_sub')
                    nloc = s.get('curve_node_local')
                    cedg = s.get('curve_edges')
                    if nsub is not None and len(nsub):
                        sd = getattr(sel, 'structure_data', {}) or {}
                        et = str(sd.get('element_type', 'node')).lower()
                        eidx = int(sd.get('index', sel.local_index))
                        t_edge = float(sd.get('t_edge', 0.0))
                        if et == 'edge' and cedg is not None \
                                and 0 <= eidx < len(cedg):
                            gnode = int(cedg[eidx][1 if t_edge >= 0.5
                                                   else 0])
                        else:
                            gnode = eidx
                        if 0 <= gnode < len(nsub):
                            sub_idx  = int(nsub[gnode])
                            cut_node = int(nloc[gnode])
                            # CC is DISPLAY-ONLY (safe fallback to the
                            # sub id).  Do NOT gate the click on it —
                            # an empty/None curve_sub_to_cc (e.g. after
                            # Load curves) must NOT make curves unclick-
                            # able / un-deletable.
                            _ccv = (s.get('curve_sub_to_cc')
                                    or {}).get(sub_idx, None)
                            cc_sid = (int(_ccv) if isinstance(
                                _ccv, (int, np.integer)) and _ccv >= 0
                                else sub_idx)
                            if (s.get('curve_last_clicked_sub')
                                    != sub_idx
                                    or s.get('curve_last_clicked_node')
                                    != cut_node):
                                s['curve_last_clicked_sub']  = sub_idx
                                s['curve_last_clicked_node'] = cut_node
                                print(f'  [curve click] sub={sub_idx} '
                                      f'CC={cc_sid} @ node {cut_node}  '
                                      f'(→ Delete clicked curve)')
                                # highlight the selected curve (white,
                                # thicker, on top of the 'curves' net)
                                _ssp = next(
                                    (np.asarray(sp, np.float32)
                                     for _sd, sp in _curve_data
                                     if int(_sd) == sub_idx), None)
                                if ps.has_curve_network('curve_sel'):
                                    ps.remove_curve_network('curve_sel')
                                if _ssp is not None and len(_ssp) >= 2:
                                    _se = np.column_stack(
                                        [np.arange(len(_ssp) - 1),
                                         np.arange(1, len(_ssp))]
                                    ).astype(np.int32)
                                    _cs = ps.register_curve_network(
                                        'curve_sel', _ssp, _se)
                                    _cs.set_color((1.0, 1.0, 1.0))
                                    _cs.set_radius(
                                        _CURVE_RADIUS * _SCALE * 2.4,
                                        relative=False)
                                    _cs.set_enabled(True)
                elif struct == 'topo_mst_junctions':
                    # change-gate: polyscope's have_selection() is
                    # sticky and fires every frame.  Only print on
                    # actual click change.
                    if picked != int(s.get('topo_mst_last_jct', -1)):
                        s['topo_mst_last_jct'] = picked
                        info = s.get('topo_mst_jct_info') or []
                        if 0 <= picked < len(info):
                            e = info[picked]
                            brs = e.get('branches', [])
                            cap_str = ''
                            if any(b.get('capped') for b in brs):
                                cap_str = '  (≥ counts hit cap_full=10000)'
                            print(f'\n  [click junction #{picked}]  '
                                  f'global_idx={e["gidx"]}  '
                                  f'cc_sid={e["sid"]}  deg={len(brs)}'
                                  f'{cap_str}')
                            for k, b in enumerate(brs):
                                sign = '≥' if b['capped'] else '='
                                print(f'    branch[{k}]: subtree size '
                                      f'{sign}{b["cnt"]} pts   '
                                      f'(first MST neigh global_idx='
                                      f'{b["first_gidx"]})')
                            # Run the click pipeline (arm/trunk analysis,
                            # delete / merge / separate / search).
                            _topo_branch_explore(int(e['gidx']),
                                                  n_per_branch=100)
                elif struct == 'topo_mst_leaves':
                    if picked != int(s.get('topo_mst_last_leaf', -1)):
                        s['topo_mst_last_leaf'] = picked
                        info = s.get('topo_mst_leaf_info') or []
                        if 0 <= picked < len(info):
                            e = info[picked]
                            print(f'\n  [click leaf #{picked}]  '
                                  f'global_idx={e["gidx"]}  '
                                  f'cc_sid={e["sid"]}  MST neighbour '
                                  f'global_idx={e["neigh_gidx"]}')
                elif struct == 'topo_mst':
                    # SELECT a node on the pink `topo_mst` curve_network.
                    # Two slots (FIFO of 2):
                    #   #1 = yellow ball (Delete-A / Delete-B act on it)
                    #   #2 = cyan ball   (used with #1 for Connect 1↔2)
                    # Edge picks resolve to the closer endpoint via t_edge.
                    ng = s.get('topo_mst_node_gidx')
                    adj = s.get('topo_mst_global_adj')
                    if ng is None or len(ng) == 0 or adj is None:
                        pass    # need 'Topo MST viz' first; silent
                    else:
                        sd = getattr(sel, 'structure_data', {}) or {}
                        et = str(sd.get('element_type', 'node')).lower()
                        eidx = int(sd.get('index', sel.local_index))
                        t_edge = float(sd.get('t_edge', 0.0))
                        ce = s.get('topo_mst_curve_edges')
                        if (et == 'edge' and ce is not None
                                and 0 <= eidx < len(ce)):
                            node_idx = int(ce[eidx][1 if t_edge >= 0.5
                                                    else 0])
                        else:
                            node_idx = eidx
                        pn1 = int(s.get('topo_mst_pick_node', -1))
                        pn2 = int(s.get('topo_mst_pick2_node', -1))
                        if (0 <= node_idx < len(ng)
                                and node_idx != pn1
                                and node_idx != pn2):
                            g0 = int(ng[node_idx])
                            neigh = [int(v) for v
                                     in adj.get(int(g0), ())]
                            sizes = []
                            for n0 in neigh:
                                seen = {g0, n0}
                                q = [n0]; head = 0
                                while head < len(q):
                                    u = q[head]; head += 1
                                    for v in adj.get(int(u), ()):
                                        v = int(v)
                                        if v in seen:
                                            continue
                                        seen.add(v); q.append(v)
                                sizes.append(len(seen) - 1)
                            # FIFO of 2: empty→#1, #1-filled→#2, both
                            # filled→evict #1, shift #2→#1, new→#2.
                            if pn1 < 0:
                                slot = 1
                            elif pn2 < 0:
                                slot = 2
                            else:
                                # rotate: #2 → #1
                                s['topo_mst_pick_node'] = pn2
                                s['topo_mst_pick_gidx'] = int(
                                    s.get('topo_mst_pick2_gidx', -1))
                                s['topo_mst_pick_deg']  = int(
                                    s.get('topo_mst_pick2_deg', -1))
                                slot = 2
                            if slot == 1:
                                s['topo_mst_pick_node'] = node_idx
                                s['topo_mst_pick_gidx'] = g0
                                s['topo_mst_pick_deg']  = len(neigh)
                            else:
                                s['topo_mst_pick2_node'] = node_idx
                                s['topo_mst_pick2_gidx'] = g0
                                s['topo_mst_pick2_deg']  = len(neigh)
                            s['topo_mst_last_clickdel'] = -1
                            # marker balls — yellow for #1, cyan for #2.
                            # Re-register BOTH so a rotation refreshes
                            # the yellow position to the new #1.
                            try:
                                nx = s.get('topo_mst_curve_nodes')
                                _pn1n = int(s.get('topo_mst_pick_node',
                                                   -1))
                                _pn2n = int(s.get('topo_mst_pick2_node',
                                                   -1))
                                for _nm in ('topo_mst_pick',
                                            'topo_mst_pick2'):
                                    if ps.has_point_cloud(_nm):
                                        ps.remove_point_cloud(_nm)
                                if (nx is not None
                                        and 0 <= _pn1n < len(nx)):
                                    pc = ps.register_point_cloud(
                                        'topo_mst_pick',
                                        np.asarray(nx[_pn1n],
                                                   np.float32)[None, :])
                                    pc.set_color((1.0, 0.95, 0.10))
                                    pc.set_radius(3.0, relative=False)
                                    pc.set_enabled(True)
                                if (nx is not None
                                        and 0 <= _pn2n < len(nx)):
                                    pc2 = ps.register_point_cloud(
                                        'topo_mst_pick2',
                                        np.asarray(nx[_pn2n],
                                                   np.float32)[None, :])
                                    pc2.set_color((0.10, 0.95, 1.0))
                                    pc2.set_radius(3.0, relative=False)
                                    pc2.set_enabled(True)
                            except Exception:
                                pass
                            print(f'  [topo-mst pick #{slot}] node '
                                  f'#{node_idx} (gidx={g0})  '
                                  f'deg={len(neigh)}  side sizes='
                                  f'{sizes}  → "Delete A/B" act on #1, '
                                  f'"Connect 1↔2" needs both')
    except Exception as _e:
        print(f'  [click error] {_e}')

    # A / B = Connect 1↔2 (straight / sketch);  ←/→/Enter = pairing preview.
    try:
        try:
            _wk = bool(psim.GetIO().WantCaptureKeyboard)
        except Exception:
            _wk = False
        # A / B : Connect 1↔2 straight / sketch-bridge — mirror the
        # "Straight Connect" / "Sketch Connect" buttons.  Only when two MST
        # picks exist and not typing in an input box.
        if not _wk and (int(s.get('topo_mst_pick_gidx', -1)) >= 0
                        and int(s.get('topo_mst_pick2_gidx', -1)) >= 0):
            if psim.IsKeyPressed(psim.ImGuiKey_A, repeat=False):
                _topo_mst_connect_apply()
            if psim.IsKeyPressed(psim.ImGuiKey_B, repeat=False):
                _topo_mst_sketch_connect()
        # K : keep only the largest MST component (no pick needed)
        if not _wk and psim.IsKeyPressed(psim.ImGuiKey_K, repeat=False):
            _topo_mst_keep_largest_comp()
        # ← / → : switch the junction-pairing preview (A / B) when it exists
        # (skip while typing in an input box so arrows still move the cursor)
        if (not _wk and ps.has_point_cloud('topo_branch_pairing_0')):
            _cur = int(s.get('topo_branch_pairing_shown', 0))
            if psim.IsKeyPressed(psim.ImGuiKey_LeftArrow, repeat=False):
                _topo_pairing_show((_cur - 1) % 4)
            elif psim.IsKeyPressed(psim.ImGuiKey_RightArrow, repeat=False):
                _topo_pairing_show((_cur + 1) % 4)
            elif psim.IsKeyPressed(psim.ImGuiKey_Enter, repeat=False):
                _topo_pairing_accept()
            elif psim.IsKeyPressed(psim.ImGuiKey_D, repeat=False):
                # D = DELETE the highlighted >-----< instead of re-pairing.
                _topo_pairing_delete()
        # RIGHT-CLICK (a click, NOT a right-drag pan) → clear all
        if psim.IsMouseClicked(psim.ImGuiMouseButton_Right):
            _mp = psim.GetMousePos()
            state['_rc0'] = ((_mp.x, _mp.y) if hasattr(_mp, 'x')
                             else (float(_mp[0]), float(_mp[1])))
        if psim.IsMouseReleased(psim.ImGuiMouseButton_Right):
            _p0 = state.get('_rc0')
            _mp = psim.GetMousePos()
            _xy = ((_mp.x, _mp.y) if hasattr(_mp, 'x')
                   else (float(_mp[0]), float(_mp[1])))
            if _p0 is not None and (
                    (_xy[0] - _p0[0]) ** 2
                    + (_xy[1] - _p0[1]) ** 2) < 36.0:   # ≤6 px ⇒ click
                _clear_all_selection()
            state['_rc0'] = None
    except Exception as _e:
        print(f'  [walk key error] {_e}')

    if s['running'] and s['iter'] < int(s['n_iter']):
        _do_step()
        _update_viz()
        if s['iter'] >= int(s['n_iter']):
            s['running'] = False


    psim.TextUnformatted(
        f'iter = {s["iter"]} / {s["n_iter"]}    '
        f'global = {int(s.get("global_iter", 0))}    '
        f'picked = {s["picked"]}')
    psim.Separator()
    if _cupy_available:
        _, s['ms_gpu'] = psim.Checkbox(
            'ms GPU (cupy shift compute)', bool(s['ms_gpu']))
        psim.Separator()

    # ════════ Main MS loop & history ════════
    if _sec('Main MS loop & history', True):
        if psim.Button('Step'):
            _do_step()
            _update_viz()
        psim.SameLine()
        if _firebrick_button('Run'):
            s['running'] = True
        psim.SameLine()
        if psim.Button('Reset'):
            s['pts']  = pts_orig.copy()
            s['dirs'] = dirs_orig.copy()
            s['iter'] = 0
            s['global_iter'] = 0
            s['history'] = [pts_orig.copy()]
            s['running'] = False
            s['phase_bws'] = []
            s['phase_active'] = False
            s['history_full'] = [_snapshot_state()]
            s['view_iter']    = 0
            s['shown_iter']   = 0
            if ps.has_curve_network('topo_links'):
                ps.remove_curve_network('topo_links')
            s['topo_links_edges'] = None
            s['topo_links_rest']  = None
            s['topo_links_iter']  = -1
            _update_viz()
            print('  reset')

        _, s['topo_lock_on'] = psim.Checkbox(
            'topo lock (freeze streamline lattice, then preserve)',
            bool(s.get('topo_lock_on', True)))
        _, s['topo_lock_iter'] = psim.SliderInt(
            'topo lock iter (freeze topology here; preserve after)',
            int(s.get('topo_lock_iter', 5)), 1, 100)
        _, s['topo_link_cone_deg'] = psim.SliderFloat(
            'topo cone (deg; fwd/bwd half-angle; sharper=lower)',
            float(s.get('topo_link_cone_deg', 30.0)), 5.0, 89.0,
            format='%.0f')
        _tle = s.get('topo_links_edges')
        if _tle is not None:
            psim.SameLine()
            psim.TextUnformatted(
                f'  frozen @ iter {int(s.get("topo_links_iter", -1))}: '
                f'{len(_tle)} edges')

        n_hist = len(s['history_full'])
        if n_hist > 1:
            _max_view = n_hist - 1
            if s['view_iter'] > _max_view:
                s['view_iter'] = _max_view
            _, s['view_iter'] = psim.SliderInt(
                f'view iter (0..{_max_view})',
                int(s['view_iter']), 0, _max_view)
            if s['view_iter'] != s['shown_iter']:
                _render_iter(int(s['view_iter']))
                s['shown_iter'] = int(s['view_iter'])

        _, s['n_iter']   = psim.SliderInt('n_iter', int(s['n_iter']), 1, 500)
        _bw_max = 60.0 if (s['bw_start'] > 15 or s['bw_end'] > 15) else 40.0
        _, s['bw_start'] = psim.SliderFloat('bw_start',
                                            float(s['bw_start']), 0.5, _bw_max)
        _, s['bw_end']   = psim.SliderFloat('bw_end',
                                            float(s['bw_end']), 0.5, _bw_max)
        bw_now = _scheduled_bw(int(s['iter']), int(s['n_iter']),
                               float(s['bw_start']), float(s['bw_end']))
        psim.TextUnformatted(f'  → current bw = {bw_now:.3f}')
        _, s['dir_alpha'] = psim.SliderFloat(
            'dir_alpha', float(s['dir_alpha']), 0.0, 8.0)
        _, s['k_search'] = psim.SliderInt(
            'k_search (per-point kNN count)',
            int(s['k_search']), 4, 1024)
        _, s['update_dirs'] = psim.Checkbox(
            'update_dirs (asymmetry-damped per-iter dir refresh)',
            bool(s['update_dirs']))
        if bool(s['update_dirs']):
            _, s['dir_asym_gamma'] = psim.SliderFloat(
                'dir_asym_gamma (0=no damp, larger=freeze endpoints harder)',
                float(s['dir_asym_gamma']), 0.0, 2.0, format='%.3f')
        _, s['gauss_penalty_par']  = psim.SliderFloat(
            'gauss_penalty_par α∥ (axial — small=loose along d_i)',
            float(s['gauss_penalty_par']), 0.1, 16.0)
        _, s['gauss_penalty_perp'] = psim.SliderFloat(
            'gauss_penalty_perp α⊥ (lateral — large=tight ⊥d_i)',
            float(s['gauss_penalty_perp']), 0.1, 16.0)
        _, s['bw_aniso_ratio'] = psim.SliderFloat(
            'bw_aniso_ratio (σ∥ / σ⊥; 1=sphere, 1.5=prolate along d_i)',
            float(s['bw_aniso_ratio']), 0.5, 5.0)

    # ════════ Topology reconstruction ════════
    if _sec('Topology reconstruction', True):
        if _firebrick_button('Build Topology MST'):
            _topo_mst_viz()
            # Hand the view over to the MST: the dense MS cloud would
            # otherwise bury the thin overlay.  Same convention as the
            # curve-fitting stage (cf. the _hide_orig_viz call sites).
            _hide_orig_viz()
            if ps.has_curve_network('topo_mst'):
                ps.get_curve_network('topo_mst').set_enabled(True)
        # Click-delete on the pink `topo_mst` curve_network.  Click a
        # node → yellow ball = pick #1.  Click a 2nd node → cyan ball =
        # pick #2.  Further clicks rotate (oldest evicted).  Delete-A /
        # Delete-B act on pick #1 only; Connect 1↔2 needs both.
        _pn1 = int(s.get('topo_mst_pick_node', -1))
        _pg1 = int(s.get('topo_mst_pick_gidx', -1))
        _pd1 = int(s.get('topo_mst_pick_deg', -1))
        _pn2 = int(s.get('topo_mst_pick2_node', -1))
        _pg2 = int(s.get('topo_mst_pick2_gidx', -1))
        _pd2 = int(s.get('topo_mst_pick2_deg', -1))
        psim.TextUnformatted(
            '  pick #1 (yellow): '
            + ('(click a node on `topo_mst`)'
               if _pn1 < 0
               else f'node #{_pn1}  gidx={_pg1}  deg={_pd1}'))
        psim.TextUnformatted(
            '  pick #2 (cyan):   '
            + ('(click a 2nd node)'
               if _pn2 < 0
               else f'node #{_pn2}  gidx={_pg2}  deg={_pd2}'))
        _no_p1 = (_pn1 < 0)
        # ── Recolour picked MST (CC) — RGB input + colour picker ───
        _node_cc_arr = s.get('topo_mst_node_cc')
        _node_cols   = s.get('topo_mst_node_cols')
        _cc_of_pick  = -1
        if (_node_cc_arr is not None and _node_cols is not None
                and _pn1 >= 0 and _pn1 < len(_node_cc_arr)):
            _cc_of_pick = int(_node_cc_arr[_pn1])
        if _cc_of_pick >= 0:
            _pal_dict   = s.setdefault('topo_mst_cc_palette', {})
            _base_pal   = s.get('topo_mst_base_cc_palette') or {}
            _cur_col    = list(_pal_dict.get(
                _cc_of_pick,
                _base_pal.get(_cc_of_pick,
                              tuple(float(c) for c in _node_cols[_pn1]))))
            _ch_rgb, _new_rgb = psim.InputFloat3(
                f'RGB input  (cc={_cc_of_pick})', _cur_col)
            _ch_pick, _new_pick = psim.ColorEdit3(
                f'RGB picker (cc={_cc_of_pick})', _cur_col)
            _new_col = (_new_rgb if _ch_rgb
                        else (_new_pick if _ch_pick else None))
            if _new_col is not None:
                _new_col = [max(0.0, min(1.0, float(c))) for c in _new_col]
                _pal_dict[_cc_of_pick] = list(_new_col)
                _mask = (np.asarray(_node_cc_arr) == _cc_of_pick)
                if _mask.any():
                    arr = np.asarray(_node_cols, np.float32).copy()
                    arr[_mask] = np.asarray(_new_col, np.float32)
                    s['topo_mst_node_cols'] = arr
                    if ps.has_curve_network('topo_mst'):
                        ps.get_curve_network('topo_mst').add_color_quantity(
                            'cc', arr.astype(np.float64),
                            defined_on='nodes', enabled=True)
            if psim.Button(f'Reset cc={_cc_of_pick} to default colour'):
                _pal_dict.pop(_cc_of_pick, None)
                _base = _base_pal.get(_cc_of_pick)
                if _base is not None:
                    _mask = (np.asarray(_node_cc_arr) == _cc_of_pick)
                    arr = np.asarray(_node_cols, np.float32).copy()
                    arr[_mask] = np.asarray(_base, np.float32)
                    s['topo_mst_node_cols'] = arr
                    if ps.has_curve_network('topo_mst'):
                        ps.get_curve_network('topo_mst').add_color_quantity(
                            'cc', arr.astype(np.float64),
                            defined_on='nodes', enabled=True)
        psim.BeginDisabled(_no_p1)
        if psim.Button('Clear picks'):
            _topo_mst_clickdel_clear()
        if psim.Button(f'[Delete A] Delete picked1 ±{_TOPO_MST_CLICKDEL_VOX:.0f} vox)'):
            _topo_mst_clickdel_apply('A')
        if psim.Button('[Delete B] Delete short side of picked1'):
            _topo_mst_clickdel_apply('B')
        if psim.Button('[Delete C] Delete whole MST component'):
            _topo_mst_clickdel_apply('C')

        psim.EndDisabled()
        if psim.Button('Keep largest MST comp, delete rest  [key K]'):
            _topo_mst_keep_largest_comp()
        _no_two = (_pn1 < 0 or _pn2 < 0)
        psim.BeginDisabled(_no_two)
        if psim.Button('Straight Connect  [key A]'):
            _topo_mst_connect_apply()
        psim.SameLine()
        if psim.Button('Sketch Connect  [key B]'):
            _topo_mst_sketch_connect()
        psim.EndDisabled()
        # Undo is gated by the STACK, not by the picks: a Connect clears
        # both picks, which would otherwise grey this button out at
        # exactly the moment you want to revert the Connect just made.
        _stack_len = len(state.get('topo_mst_undo_stack', []))
        psim.SameLine()
        psim.BeginDisabled(_stack_len == 0)
        if psim.Button(f'Undo stitch ({_stack_len})'):
            _topo_stitch_undo()
        psim.EndDisabled()
        _, s['connect_auto_process_new_jct'] = psim.Checkbox(
            'junction auto-process',
            bool(s.get('connect_auto_process_new_jct', True)))


    # ════════ Curve fitting ════════
    if _sec('Curve fitting', True):
        _, s['curve_radius']  = psim.SliderFloat(
            'curve radius (vox; KDTree proximity edge)',
            float(s['curve_radius']), 1.0, 20.0)
        _, s['curve_min_pts'] = psim.SliderInt(
            'curve min_pts (drop CCs smaller than this)',
            int(s['curve_min_pts']), 2, 200)
        if _firebrick_button('Fit curves'):
            _fit_curves_topo()
        psim.SameLine()
        if psim.Button('Clear curves'):
            _clear_curves()
        psim.SameLine()
        if psim.Button('Delete clicked curve'):
            _delete_curve()
        _last_sub = int(s.get('curve_last_clicked_sub', -1))
        if _last_sub >= 0:
            psim.SameLine()
            psim.TextUnformatted(f'  clicked: sub={_last_sub}')
        _K = int(s.get('curve_K', 0))
        if _K > 0:
            psim.TextUnformatted(
                f'  fitted {_K} CCs ({len(_curve_data)} sub-curves after '
                f'splitting); click a curve to select it for Delete')

    # ════════ Collapse solving ════════
    if _sec('Collapse solving', True):
        _sols = s.get('collapse_solutions') or []
        if _firebrick_button('Solve Collapse'):
            _solve_collapse()
        _nsol = len(_sols)
        psim.TextUnformatted(
            '  (no solutions yet - Fit curves, then Solve Collapse)'
            if _nsol == 0 else
            f'  solution {int(s.get("collapse_shown", 0)) + 1} / {_nsol}'
            f'   curves: {len(_curve_data)}')
        if _nsol:
            _dl = s.get('collapse_dets') or []
            _ci = int(s.get('collapse_shown', 0))
            _dv = _dl[_ci] if 0 <= _ci < len(_dl) else None
            psim.TextUnformatted(
                '  knot: n/a (needs a single strand)' if _dv is None else
                f'  knot: {_dv[2]} crossings   log10|det| = {_dv[1]:.2f}')
        psim.BeginDisabled(_nsol == 0)
        if psim.Button('< Prev'):
            _collapse_show(int(s.get('collapse_shown', 0)) - 1)
        psim.SameLine()
        if psim.Button('Next >'):
            _collapse_show(int(s.get('collapse_shown', 0)) + 1)
        psim.SameLine()
        if psim.Button('Accept'):
            _collapse_accept()
        psim.EndDisabled()

    # ════════ Save / Load ════════
    if _sec('Save / Load', True):
        _pfx_chg, _pfx_new = psim.InputText('save_postfix (segs/curves only; not ms_points)', _SAVE_POSTFIX)
        if _pfx_chg:
            _set_save_postfix(_pfx_new)
            print(f'  [save_postfix] segs{_SAVE_POSTFIX}/  curves{_SAVE_POSTFIX}/')
        # Save the CURRENT mean-shifted points + updated directions, in the
        # input npz format (points + directions/dirs [+ energy/linearity]).
        if psim.Button('Save current points (→ ms_points/*.npz)'):
            _save_current_points(timestamped=True)
        if psim.Button(f'Save seg state (→ segs{_SAVE_POSTFIX}/seg_state_*.npz)'):
            _save_seg_state()
        if psim.Button(f'Save yarn (→ curves{_SAVE_POSTFIX}/yarn_latest.npz + yarn_<ts>.npz)'):
            _save_yarn()
        if psim.Button('Load yarn (curves/yarn_latest.npz → state)'):
            _load_yarn()
        if psim.Button('Save camera (→ segs/camera_*.json)'):
            _save_camera_state()
        if psim.Button('Load camera (← newest segs/camera_*.json)'):
            _load_camera_state()



if args.load_seg is not None:
    _load_seg_state(args.load_seg or None)

if args.load_yarn is not None:
    _load_yarn(args.load_yarn or None)

if args.batch_fit:
    print(f'[batch-fit] {_NPZ_STEM}: Fit curves (topo MST, cached adj)')
    _fit_curves_topo()
    print(f'[batch-fit] {_NPZ_STEM}: Save yarn '
          f'(postfix={args.save_postfix!r})')
    _save_yarn('batch_fit', backup=True)
    print(f'[batch-fit] {_NPZ_STEM}: done')
    raise SystemExit(0)

if args.ablation:
    # Override MS state for an ablation variant (applied live, read per step).
    _ABLATIONS = {
        'full':           {},
        'wo_aniso':       {'bw_aniso_ratio': 1.0, 'gauss_penalty_perp': 1.0,
                           'dir_alpha': 0.0},
        'wo_dir_update':  {'update_dirs': False},
        'wo_dir_asym':    {'dir_asym_gamma': 0.0},
        'wo_topo':        {'topo_lock_on': False},
        'wo_anneal_bw30': {'bw_start': 30.0, 'bw_end': 30.0,
                           'topo_lock_on': False},
        'wo_anneal_bw10': {'bw_start': 10.0, 'bw_end': 10.0},
    }
    if args.ablation not in _ABLATIONS:
        raise SystemExit(f'[ablation] unknown variant {args.ablation!r}; '
                         f'choose from {list(_ABLATIONS)}')
    for _k, _v in _ABLATIONS[args.ablation].items():
        state[_k] = _v
    print(f'[ablation] {args.ablation}: ' +
          (', '.join(f'{k}={v}' for k, v in _ABLATIONS[args.ablation].items())
           or '(no overrides = full)'))

if args.batch or args.batch_topo_save:
    # Headless equivalent of clicking "Run" (default MS, n_iter iters) then
    # "Save current points" (→ output/<stem>/ms_points/).  With
    # --batch_topo_save also clicks "Topo MST" + "Save seg state" so the
    # full unprocessed (no-edit) seg lands under segs<save_postfix>/.
    print(f'[batch] {_NPZ_STEM}: Run MS (n_iter={int(state["n_iter"])})')
    while int(state['iter']) < int(state['n_iter']):
        _do_step()
    _save_current_points()
    if args.batch_topo_save:
        print(f'[batch] {_NPZ_STEM}: Topo MST')
        _topo_mst_viz()
        print(f'[batch] {_NPZ_STEM}: Save seg state '
              f'(postfix={args.save_postfix!r})')
        _save_seg_state(timestamped=True)
        print(f'[batch] {_NPZ_STEM}: Fit curves (topo MST)')
        _fit_curves_topo()
        print(f'[batch] {_NPZ_STEM}: Save yarn '
              f'(postfix={args.save_postfix!r})')
        _save_yarn('batch_topo', backup=True)
    print(f'[batch] {_NPZ_STEM}: done')
    raise SystemExit(0)

ps.set_user_callback(callback)
print('\nClick a point to inspect (bw ellipsoid + kNN). '
      'Step / Run to advance MS; scrub the view-iter slider for history.')
ps.show()
