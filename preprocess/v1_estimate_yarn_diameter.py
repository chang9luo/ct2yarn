#!/usr/bin/env python3
"""Estimate yarn diameter from denoised CT volumes via 2D distance-transform + skeleton.

For each `.nrrd` under --input-dir (or a single --file), samples N slices along
each of the 3 axes, computes per-slice 2D distance transform of the binary mask
(volume > 0), runs a 2D skeleton, and reads distance values at skeleton voxels.
Each value equals the local yarn radius. Pool across all slices → median = R.

Outputs a rich summary table, per-file diagnostic PNGs under
`<input-dir>/diameter_figs/`, and a suggested (SIGMA_GABOR, FREQ, KERNEL_SIZE)
triple ready to paste into v2_gabor_pointcloud.py's CONFIG.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nrrd
import numpy as np
import yaml
from rich.console import Console
from rich.table import Table
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize


DEFAULT_INPUT_DIR = Path("data/processed")
N_SLICES_PER_AXIS = 30     # 30 positions × 3 axes = 90 slices per volume
FREQ_RATIO_DEFAULT = 1.4   # FREQ = 1 / (FREQ_RATIO * diameter), matches v2_gabor_pointcloud.py


@dataclass
class DiameterEstimate:
    name: str = ""
    n_skel: int = 0
    radius_median: float = 0.0
    radius_p25: float = 0.0
    radius_p75: float = 0.0
    diameter_median: float = 0.0
    suggested_sigma: int = 0
    suggested_freq: float = 0.0
    suggested_kernel: int = 0


def slice_radii(slice_mask: np.ndarray) -> np.ndarray:
    """Distance-transform value at every skeleton voxel of a 2D binary slice."""
    if slice_mask.sum() < 10:
        return np.empty(0, dtype=np.float32)
    dist = distance_transform_edt(slice_mask)
    sk = skeletonize(slice_mask)
    return dist[sk].astype(np.float32)


def gather_radii(mask: np.ndarray, n_per_axis: int) -> np.ndarray:
    """Sample n positions along each axis (× 3 axes), pool all skeleton radii."""
    bag = []
    for axis, n in enumerate(mask.shape):
        for p in np.linspace(0.1, 0.9, n_per_axis):
            i = int(p * n)
            if axis == 0:
                s = mask[i, :, :]
            elif axis == 1:
                s = mask[:, i, :]
            else:
                s = mask[:, :, i]
            bag.append(slice_radii(s))
    return np.concatenate(bag) if bag else np.empty(0, dtype=np.float32)


def finalize_estimate(radii: np.ndarray, freq_ratio: float) -> DiameterEstimate:
    if radii.size == 0:
        return DiameterEstimate()
    r25, r50, r75 = np.percentile(radii, [25, 50, 75])
    diameter = 2.0 * float(r50)
    sigma = max(1, int(round(r50)))
    freq = 1.0 / (freq_ratio * diameter) if diameter > 0 else 0.0
    kernel = 2 * int(round(3 * sigma)) + 1
    return DiameterEstimate(
        n_skel=int(radii.size),
        radius_median=float(r50),
        radius_p25=float(r25),
        radius_p75=float(r75),
        diameter_median=diameter,
        suggested_sigma=sigma,
        suggested_freq=round(freq, 4),
        suggested_kernel=kernel,
    )


def estimate_for_volume(data: np.ndarray, freq_ratio: float,
                         samples_per_axis: int
                         ) -> tuple[DiameterEstimate, np.ndarray]:
    """Return (estimate, pooled radii)."""
    mask = data > 0
    radii = gather_radii(mask, n_per_axis=samples_per_axis)
    return finalize_estimate(radii, freq_ratio), radii


def make_inspection_figure(mask_slice: np.ndarray, radii: np.ndarray, title: str):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(mask_slice, cmap="gray", origin="lower")
    axes[0].set_title("Binary mask"); axes[0].axis("off")

    dist = distance_transform_edt(mask_slice)
    sk = skeletonize(mask_slice)

    axes[1].imshow(dist, cmap="viridis", origin="lower")
    axes[1].set_title("Distance transform"); axes[1].axis("off")

    rgb = np.stack([(mask_slice * 80).astype(np.uint8)] * 3, axis=-1)
    ys, xs = np.where(sk)
    rgb[ys, xs] = [255, 60, 60]
    axes[2].imshow(rgb, origin="lower")
    axes[2].set_title(f"Mask + skeleton ({sk.sum()} pts)"); axes[2].axis("off")

    if radii.size:
        axes[3].hist(radii, bins=50, color="steelblue", alpha=0.85)
        m = float(np.median(radii))
        axes[3].axvline(m, color="red", lw=2, label=f"median r = {m:.1f}")
        axes[3].set_xlabel("Skeleton radius (vox)"); axes[3].set_ylabel("count")
        axes[3].legend(); axes[3].grid(alpha=0.3)
        axes[3].set_title("Radii distribution")
    else:
        axes[3].text(0.5, 0.5, "no skeleton points", ha="center", va="center")
        axes[3].axis("off")

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return fig


def gather_inputs(input_dir: Path) -> list[Path]:
    if not input_dir.is_dir():
        return []
    return sorted({f.resolve() for f in input_dir.rglob("*.nrrd") if f.is_file()})


def write_per_file_yaml(params_dir: Path, base_name: str, est: DiameterEstimate,
                        vol_shape: tuple, samples_per_axis: int,
                        freq_ratio: float) -> None:
    """Per-volume yarn-diameter YAML readable by v2_gabor_pointcloud.py."""
    params_dir.mkdir(parents=True, exist_ok=True)
    doc = {
        "file": est.name,
        "volume_shape": [int(vol_shape[0]), int(vol_shape[1]), int(vol_shape[2])],
        "samples_per_axis": int(samples_per_axis),
        "n_slices_sampled": int(samples_per_axis * 3),
        "freq_ratio": float(freq_ratio),
        "skeleton_points": int(est.n_skel),
        "radius": {
            "p25": round(est.radius_p25, 3),
            "median": round(est.radius_median, 3),
            "p75": round(est.radius_p75, 3),
        },
        "diameter_median": round(est.diameter_median, 3),
        "gabor": {
            "sigma": int(est.suggested_sigma),
            "freq": float(est.suggested_freq),
            "kernel_size": int(est.suggested_kernel),
        },
    }
    with (params_dir / f"{base_name}.yaml").open("w") as fh:
        yaml.safe_dump(doc, fh, sort_keys=False)


def write_batch_summary_yaml(params_dir: Path, estimates: list[DiameterEstimate],
                              base_dir: Path, samples_per_axis: int, freq_ratio: float,
                              global_diam: float, global_sigma: int,
                              global_freq: float, global_kernel: int) -> None:
    """Batch summary YAML written to <params_dir>/batch_summary.yaml."""
    params_dir.mkdir(parents=True, exist_ok=True)
    files_doc = []
    for e in estimates:
        if e.n_skel == 0:
            files_doc.append({"file": e.name, "status": "empty"})
            continue
        files_doc.append({
            "file": e.name,
            "skeleton_points": int(e.n_skel),
            "radius_iqr": [round(e.radius_p25, 2), round(e.radius_p75, 2)],
            "diameter_median": round(e.diameter_median, 2),
            "gabor": {
                "sigma": int(e.suggested_sigma),
                "freq": float(e.suggested_freq),
                "kernel_size": int(e.suggested_kernel),
            },
        })
    doc = {
        "input_dir": str(base_dir),
        "samples_per_axis": int(samples_per_axis),
        "freq_ratio": float(freq_ratio),
        "files": files_doc,
    }
    if global_diam > 0:
        doc["batch"] = {
            "median_diameter": round(global_diam, 2),
            "suggested_v2_CONFIG": {
                "SIGMA_GABOR": int(global_sigma),
                "FREQ": float(global_freq),
                "KERNEL_SIZE": int(global_kernel),
            },
        }
    with (params_dir / "batch_summary.yaml").open("w") as fh:
        yaml.safe_dump(doc, fh, sort_keys=False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR,
                    help=f"Directory of .nrrd files (default: {DEFAULT_INPUT_DIR})")
    ap.add_argument("--file", type=Path, default=None,
                    help="Single .nrrd to inspect (overrides --input-dir)")
    ap.add_argument("--freq-ratio", type=float, default=FREQ_RATIO_DEFAULT,
                    help=f"λ/diameter ratio for FREQ suggestion (default {FREQ_RATIO_DEFAULT})")
    ap.add_argument("--samples-per-axis", type=int, default=N_SLICES_PER_AXIS,
                    help=f"Slice positions per axis (default {N_SLICES_PER_AXIS}, × 3 axes)")
    ap.add_argument("--no-figs", action="store_true",
                    help="Skip writing per-file PNG diagnostics")
    args = ap.parse_args()

    console = Console()

    if args.file is not None:
        files = [args.file.resolve()]
        if not args.file.exists():
            console.print(f"[red]File not found:[/red] {args.file}")
            return 1
    else:
        files = gather_inputs(args.input_dir)
        if not files:
            console.print(f"[yellow]No .nrrd files under[/yellow] {args.input_dir}")
            return 0

    console.rule(f"[bold cyan]Yarn diameter estimation[/]  ({len(files)} volume(s))")

    base_dir = args.file.parent if args.file else args.input_dir
    figs_dir = base_dir / "diameter_figs"
    params_dir = base_dir / "params"
    if not args.no_figs:
        figs_dir.mkdir(parents=True, exist_ok=True)
        params_dir.mkdir(parents=True, exist_ok=True)

    name_w = max((len(f.name) for f in files), default=20)

    estimates: list[DiameterEstimate] = []
    for f in files:
        try:
            data, _ = nrrd.read(str(f))
            est, pooled = estimate_for_volume(
                data, freq_ratio=args.freq_ratio,
                samples_per_axis=args.samples_per_axis,
            )
            est.name = f.name
            estimates.append(est)
            if est.n_skel == 0:
                console.print(f"[yellow]{f.name:<{name_w}}[/]  no skeleton points (mask empty?)")
                continue
            iqr = f"{est.radius_p25:>4.1f}–{est.radius_p75:>4.1f}"
            console.print(
                f"[cyan]{f.name:<{name_w}}[/]  "
                f"diameter ≈ [bold]{est.diameter_median:>5.1f}[/] vox  "
                f"radius IQR {iqr:>11}  "
                f"n_skel = {est.n_skel:>10,}  "
                f"σ = {est.suggested_sigma:>2}  "
                f"freq = {est.suggested_freq:.4f}"
            )

            if not args.no_figs:
                D = data.shape[0]
                mid = D // 2
                mask_slice = data[mid] > 0
                fig = make_inspection_figure(
                    mask_slice, pooled,
                    title=(f"{f.name}  XY mid (z={mid})  "
                           f"{args.samples_per_axis}×3 slices  "
                           f"n_skel={pooled.size:,}  "
                           f"pooled-median r = {est.radius_median:.1f}")
                )
                fig.savefig(figs_dir / f"{f.stem}_diameter.png", dpi=120, bbox_inches="tight")
                plt.close(fig)
                write_per_file_yaml(params_dir, f.stem, est, data.shape,
                                    samples_per_axis=args.samples_per_axis,
                                    freq_ratio=args.freq_ratio)
        except Exception as e:
            console.print(f"[red]FAIL {f.name}[/]: {e}")

    # Per-file table
    table = Table(title="Yarn diameter estimates", header_style="bold magenta")
    table.add_column("File", style="cyan", no_wrap=True)
    table.add_column("Skel pts", justify="right")
    table.add_column("Radius IQR", justify="right")
    table.add_column("Diameter", justify="right")
    table.add_column("σ", justify="right")
    table.add_column("freq", justify="right")
    table.add_column("kernel", justify="right")
    for e in estimates:
        if e.n_skel == 0:
            table.add_row(e.name, "0", "-", "-", "-", "-", "-")
        else:
            table.add_row(
                e.name, f"{e.n_skel:,}",
                f"{e.radius_p25:.1f}–{e.radius_p75:.1f}",
                f"{e.diameter_median:.1f}",
                str(e.suggested_sigma),
                f"{e.suggested_freq}",
                str(e.suggested_kernel),
            )
    console.print(table)

    # Aggregate across files
    diams = [e.diameter_median for e in estimates if e.diameter_median > 0]
    if diams:
        d_med = float(np.median(diams))
        sigma_glob = max(1, int(round(d_med / 2)))
        freq_glob = round(1.0 / (args.freq_ratio * d_med), 4)
        kernel_glob = 2 * int(round(3 * sigma_glob)) + 1
        console.print(
            f"\n[bold]Across-batch median diameter:[/] {d_med:.1f} vox  "
            f"(range {min(diams):.1f}–{max(diams):.1f}, n={len(diams)})"
        )
        console.print(
            f"[bold green]Suggested v2 CONFIG:[/]  "
            f"SIGMA_GABOR = {sigma_glob}   FREQ = {freq_glob}   "
            f"→ KERNEL_SIZE = {kernel_glob}"
        )

    if not args.no_figs:
        write_batch_summary_yaml(params_dir, estimates, base_dir,
                                  samples_per_axis=args.samples_per_axis,
                                  freq_ratio=args.freq_ratio,
                                  global_diam=d_med if diams else 0.0,
                                  global_sigma=sigma_glob if diams else 0,
                                  global_freq=freq_glob if diams else 0.0,
                                  global_kernel=kernel_glob if diams else 0)
        console.print(f"[dim]Figures → {figs_dir}  •  Params → {params_dir}[/dim]")

    return 0


if __name__ == "__main__":
    sys.exit(main())
