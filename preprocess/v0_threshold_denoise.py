#!/usr/bin/env python3
"""Otsu threshold denoising for NRRD CT volumes (Sec. 3.1 of the paper).

Batch-processes every .nrrd in an input directory (default
`data/raw`), writing the denoised volume to the matching path
under an output directory (default `data/processed`). Existing
outputs are NOT overwritten. Diagnostic figures (intensity + slices) are
saved into a `figs/` subfolder of the output directory as PNGs (never shown).

Usage:
    python preprocess/v0_threshold_denoise.py
    python preprocess/v0_threshold_denoise.py --input-dir data/raw --output-dir data/processed
    python preprocess/v0_threshold_denoise.py --no-figs
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib  # backend is set in main() (Agg unless --interactive)

import nrrd
import numpy as np
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks


@dataclass
class Result:
    name: str
    status: str  # "ok" | "skip" | "fail"
    threshold: float = 0.0
    kept_pct: float = 0.0
    peaks_found: int = 0
    out_bytes: int = 0
    seconds: float = 0.0
    note: str = ""
    quit_requested: bool = False


def compute_threshold(
    volume: np.ndarray, nonzero: np.ndarray
) -> tuple[float, int, np.ndarray, np.ndarray, np.ndarray, Optional[int], Optional[int]]:
    """Bimodal valley threshold; fall back to P95 of non-zero voxels if <2 peaks.

    Histogram-level smoothing (sigma=3 on the 256-bin counts) stabilizes peak
    detection; the volume itself is NOT smoothed, to preserve yarn boundaries.
    """
    hist, bins = np.histogram(volume.flatten(), bins=256, range=(0, 1))
    bin_centers = (bins[:-1] + bins[1:]) / 2
    hist_smooth = gaussian_filter(hist.astype(np.float64), sigma=3)
    peaks, _ = find_peaks(hist_smooth, height=hist_smooth.max() * 0.01, distance=10)

    p1_idx: Optional[int] = None
    p2_idx: Optional[int] = None
    if len(peaks) >= 2:
        peak_heights = hist_smooth[peaks]
        top2 = np.argsort(peak_heights)[-2:]
        p1_idx, p2_idx = sorted(int(peaks[i]) for i in top2)
        valley_region = hist_smooth[p1_idx:p2_idx]
        valley_idx = p1_idx + int(np.argmin(valley_region))
        threshold = float(bin_centers[valley_idx])
    else:
        threshold = float(np.percentile(nonzero, 95)) if nonzero.size else 0.0

    return threshold, len(peaks), hist, hist_smooth, bin_centers, p1_idx, p2_idx


def compute_otsu(
    volume: np.ndarray, nonzero: np.ndarray
) -> tuple[float, int, np.ndarray, np.ndarray, np.ndarray, Optional[int], Optional[int]]:
    """Otsu's between-class variance threshold over non-zero voxels.

    Returns the same tuple shape as `compute_threshold` so callers can swap.
    `peaks_found` is reported as 0 and peak indices as None for Otsu (no
    explicit peak detection happens).
    """
    hist, bins = np.histogram(volume.flatten(), bins=256, range=(0, 1))
    bin_centers = (bins[:-1] + bins[1:]) / 2
    hist_smooth = gaussian_filter(hist.astype(np.float64), sigma=3)

    if nonzero.size == 0:
        return 0.0, 0, hist, hist_smooth, bin_centers, None, None

    nz_counts, nz_edges = np.histogram(nonzero, bins=256)
    nz_centers = (nz_edges[:-1] + nz_edges[1:]) / 2
    total = nz_counts.sum()
    if total == 0:
        return 0.0, 0, hist, hist_smooth, bin_centers, None, None

    weight1 = np.cumsum(nz_counts).astype(np.float64)
    weight2 = total - weight1
    sum_counts = np.cumsum(nz_counts * nz_centers)
    sum_total = sum_counts[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        mean1 = np.where(weight1 > 0, sum_counts / weight1, 0.0)
        mean2 = np.where(weight2 > 0, (sum_total - sum_counts) / weight2, 0.0)
        var_between = weight1 * weight2 * (mean1 - mean2) ** 2

    valid = (weight1 > 0) & (weight2 > 0)
    if not valid.any():
        return 0.0, 0, hist, hist_smooth, bin_centers, None, None
    var_between = np.where(valid, var_between, -np.inf)
    idx = int(np.argmax(var_between))
    return float(nz_centers[idx]), 0, hist, hist_smooth, bin_centers, None, None


def save_figures(
    figs_dir: Path,
    stem: str,
    volume: np.ndarray,
    volume_denoised: np.ndarray,
    hist: np.ndarray,
    hist_smooth: np.ndarray,
    bin_centers: np.ndarray,
    threshold: float,
    p1_idx: Optional[int],
    p2_idx: Optional[int],
    nonzero: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt

    figs_dir.mkdir(parents=True, exist_ok=True)
    bins_edge = np.linspace(0, 1, 257)

    fig1, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes[0, 0].bar(bins_edge[:-1], hist, width=bins_edge[1] - bins_edge[0], color="blue", alpha=0.5)
    axes[0, 0].plot(bin_centers, hist_smooth, color="navy", lw=2, label="Smoothed hist")
    axes[0, 0].axvline(x=threshold, color="red", linestyle="--", lw=2, label=f"Valley={threshold:.4f}")
    if p1_idx is not None and p2_idx is not None:
        axes[0, 0].axvline(x=bin_centers[p1_idx], color="green", ls=":", lw=1.5,
                           label=f"Peak1={bin_centers[p1_idx]:.4f}")
        axes[0, 0].axvline(x=bin_centers[p2_idx], color="orange", ls=":", lw=1.5,
                           label=f"Peak2={bin_centers[p2_idx]:.4f}")
    axes[0, 0].set(xlabel="Intensity", ylabel="Frequency",
                   title="Intensity Distribution — Bimodal Valley")
    axes[0, 0].set_yscale("log")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=8)

    hist_nz, bins_nz = np.histogram(nonzero, bins=256)
    axes[0, 1].bar(bins_nz[:-1], hist_nz, width=bins_nz[1] - bins_nz[0], color="green", alpha=0.7)
    axes[0, 1].axvline(x=threshold, color="red", linestyle="--", lw=2)
    axes[0, 1].set(xlabel="Intensity", ylabel="Frequency", title="Non-zero Voxels")
    axes[0, 1].grid(True, alpha=0.3)

    cumsum = np.cumsum(hist).astype(np.float64)
    cumsum /= cumsum[-1] if cumsum[-1] else 1.0
    axes[1, 0].plot(bins_edge[:-1], cumsum, lw=2, color="purple")
    axes[1, 0].axvline(x=threshold, color="red", linestyle="--", lw=2)
    axes[1, 0].set(xlabel="Intensity", ylabel="Cumulative Probability", title="CDF")
    axes[1, 0].grid(True, alpha=0.3)

    thresholds_to_test = np.linspace(0.3, 0.6, 20)
    kept_pct = [np.sum(volume > t) / volume.size * 100 for t in thresholds_to_test]
    axes[1, 1].plot(thresholds_to_test, kept_pct, lw=2, color="blue")
    axes[1, 1].axvline(x=threshold, color="red", linestyle="--", lw=2)
    axes[1, 1].set(xlabel="Threshold", ylabel="Voxels Kept (%)", title="Threshold Effect")
    axes[1, 1].grid(True, alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(figs_dir / f"{stem}_intensity.png", dpi=120, bbox_inches="tight")

    fig2, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig2.suptitle(f"Manual Threshold (={threshold:.4f})", fontsize=16)
    z_mid = volume.shape[0] // 2
    y_mid = volume.shape[1] // 2
    x_mid = volume.shape[2] // 2
    pos = volume[volume > 0]
    vmin, vmax = (np.percentile(pos, [5, 95]) if pos.size else (0.0, 1.0))

    axes[0, 0].imshow(volume[z_mid, :, :], cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
    axes[0, 0].set_title("Before XY")
    axes[0, 1].imshow(volume[:, y_mid, :].T, cmap="gray", vmin=vmin, vmax=vmax,
                      origin="lower", aspect="auto")
    axes[0, 1].set_title("Before XZ")
    axes[0, 2].imshow(volume[:, :, x_mid].T, cmap="gray", vmin=vmin, vmax=vmax,
                      origin="lower", aspect="auto")
    axes[0, 2].set_title("Before YZ")
    axes[1, 0].imshow(volume_denoised[z_mid, :, :], cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
    axes[1, 0].set_title("After XY")
    axes[1, 1].imshow(volume_denoised[:, y_mid, :].T, cmap="gray", vmin=vmin, vmax=vmax,
                      origin="lower", aspect="auto")
    axes[1, 1].set_title("After XZ")
    axes[1, 2].imshow(volume_denoised[:, :, x_mid].T, cmap="gray", vmin=vmin, vmax=vmax,
                      origin="lower", aspect="auto")
    axes[1, 2].set_title("After YZ")
    fig2.tight_layout()
    fig2.savefig(figs_dir / f"{stem}_slices.png", dpi=120, bbox_inches="tight")

    plt.close(fig1)
    plt.close(fig2)


def interactive_threshold(
    volume: np.ndarray, initial_threshold: float, title: str
) -> tuple[float, str]:
    """Show a matplotlib window with three mid-slices + a threshold slider.

    The user slides to a value and clicks Accept (saves), Skip (no save, move
    on), Quit (stop batch), or Auto (reset to algorithm's initial value).
    Returns (chosen_threshold, action) where action ∈ {"accept", "skip", "quit"}.
    Only three central slices are updated on each slider change, so the UI
    stays responsive on large volumes.
    """
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, Slider

    z_mid = volume.shape[0] // 2
    y_mid = volume.shape[1] // 2
    x_mid = volume.shape[2] // 2
    slice_xy = volume[z_mid, :, :]
    slice_xz = volume[:, y_mid, :].T
    slice_yz = volume[:, :, x_mid].T

    pos = volume[volume > 0]
    if pos.size:
        vmin, vmax = np.percentile(pos, [5, 99])
    else:
        vmin, vmax = 0.0, 1.0
    vmax = float(max(vmax, vmin + 1e-4))

    def threshed(s: np.ndarray, t: float) -> np.ndarray:
        return np.where(s > t, s, 0)

    fig = plt.figure(figsize=(15, 8))
    fig.suptitle(f"{title}   (auto={initial_threshold:.4f})", fontsize=13)
    gs = fig.add_gridspec(2, 3, height_ratios=[3, 1], hspace=0.3)
    ax_xy = fig.add_subplot(gs[0, 0])
    ax_xz = fig.add_subplot(gs[0, 1])
    ax_yz = fig.add_subplot(gs[0, 2])
    ax_hist = fig.add_subplot(gs[1, :])

    im_xy = ax_xy.imshow(threshed(slice_xy, initial_threshold), cmap="gray",
                         vmin=vmin, vmax=vmax, origin="lower")
    ax_xy.set_title("XY"); ax_xy.axis("off")
    im_xz = ax_xz.imshow(threshed(slice_xz, initial_threshold), cmap="gray",
                         vmin=vmin, vmax=vmax, origin="lower", aspect="auto")
    ax_xz.set_title("XZ"); ax_xz.axis("off")
    im_yz = ax_yz.imshow(threshed(slice_yz, initial_threshold), cmap="gray",
                         vmin=vmin, vmax=vmax, origin="lower", aspect="auto")
    ax_yz.set_title("YZ"); ax_yz.axis("off")

    hist, bin_edges = np.histogram(volume.flatten(), bins=256, range=(0, 1))
    ax_hist.bar(bin_edges[:-1], hist, width=bin_edges[1] - bin_edges[0],
                color="steelblue", alpha=0.6)
    ax_hist.set_yscale("log")
    ax_hist.axvline(initial_threshold, color="gray", ls=":", lw=1.5, label=f"auto={initial_threshold:.4f}")
    line_current = ax_hist.axvline(initial_threshold, color="red", ls="--", lw=2,
                                    label=f"current={initial_threshold:.4f}")
    ax_hist.set_xlim(0, 1)
    ax_hist.set_xlabel("Intensity")
    ax_hist.grid(True, alpha=0.3)
    legend = ax_hist.legend(loc="upper right")

    fig.subplots_adjust(bottom=0.18)
    ax_slider = fig.add_axes([0.10, 0.07, 0.55, 0.03])
    slider = Slider(ax_slider, "Threshold", 0.0, 1.0, valinit=initial_threshold, valfmt="%.4f")

    ax_auto = fig.add_axes([0.68, 0.065, 0.07, 0.04])
    ax_accept = fig.add_axes([0.76, 0.065, 0.08, 0.04])
    ax_skip = fig.add_axes([0.85, 0.065, 0.06, 0.04])
    ax_quit = fig.add_axes([0.92, 0.065, 0.06, 0.04])
    btn_auto = Button(ax_auto, "Auto")
    btn_accept = Button(ax_accept, "Accept", color="palegreen", hovercolor="lightgreen")
    btn_skip = Button(ax_skip, "Skip", color="khaki")
    btn_quit = Button(ax_quit, "Quit", color="lightcoral")

    state = {"action": None, "threshold": initial_threshold}

    def update(_val: float) -> None:
        t = float(slider.val)
        im_xy.set_data(threshed(slice_xy, t))
        im_xz.set_data(threshed(slice_xz, t))
        im_yz.set_data(threshed(slice_yz, t))
        line_current.set_xdata([t, t])
        line_current.set_label(f"current={t:.4f}")
        legend.get_texts()[1].set_text(f"current={t:.4f}")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    btn_auto.on_clicked(lambda _e: slider.set_val(initial_threshold))

    def on_accept(_e):
        state["threshold"] = float(slider.val)
        state["action"] = "accept"
        plt.close(fig)

    def on_skip(_e):
        state["action"] = "skip"
        plt.close(fig)

    def on_quit(_e):
        state["action"] = "quit"
        plt.close(fig)

    btn_accept.on_clicked(on_accept)
    btn_skip.on_clicked(on_skip)
    btn_quit.on_clicked(on_quit)

    plt.show()  # blocks until window closes
    return state["threshold"], state["action"] or "skip"


def process_one(
    input_path: Path, output_path: Path, figs_dir: Path,
    save_figs: bool, method: str, interactive: bool,
) -> Result:
    if output_path.exists():
        return Result(input_path.name, "skip", out_bytes=output_path.stat().st_size, note="exists")

    t0 = time.monotonic()
    try:
        data, _ = nrrd.read(str(input_path))
        original_max = float(data.max())
        if original_max == 0:
            return Result(input_path.name, "fail", seconds=time.monotonic() - t0, note="empty volume")

        volume = data.astype(np.float32) / original_max
        nonzero = volume[volume > 0]

        compute = compute_otsu if method == "otsu" else compute_threshold
        threshold, n_peaks, hist, hist_smooth, bin_centers, p1, p2 = compute(volume, nonzero)

        if interactive:
            threshold, action = interactive_threshold(
                volume, threshold, title=f"{input_path.name}  [method={method}]"
            )
            if action == "skip":
                return Result(input_path.name, "skip", seconds=time.monotonic() - t0,
                              note="user-skipped")
            if action == "quit":
                return Result(input_path.name, "skip", seconds=time.monotonic() - t0,
                              note="user-quit", quit_requested=True)

        volume_denoised = np.where(volume > threshold, volume, 0)
        kept_pct = float(np.sum(volume_denoised > 0) / volume_denoised.size * 100)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        out_arr = (volume_denoised * original_max).astype(np.uint16)
        nrrd.write(str(output_path), out_arr)

        if save_figs:
            save_figures(figs_dir, output_path.stem, volume,
                         volume_denoised, hist, hist_smooth, bin_centers, threshold,
                         p1, p2, nonzero)

        return Result(
            input_path.name, "ok",
            threshold=threshold, kept_pct=kept_pct, peaks_found=n_peaks,
            out_bytes=output_path.stat().st_size, seconds=time.monotonic() - t0,
        )
    except Exception as e:
        return Result(input_path.name, "fail", seconds=time.monotonic() - t0, note=str(e)[:80])


def gather_inputs(input_dir: Path) -> list[Path]:
    """All .nrrd files under input_dir (recursive), sorted, deduped."""
    if not input_dir.is_dir():
        return []
    return sorted({f.resolve() for f in input_dir.rglob("*.nrrd") if f.is_file()})


def human_bytes(n: int) -> str:
    x = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if x < 1024 or unit == "TB":
            return f"{int(x)} B" if unit == "B" else f"{x:.1f} {unit}"
        x /= 1024
    return f"{x:.1f} TB"


DEFAULT_INPUT_DIR = Path("data/raw")
DEFAULT_OUTPUT_DIR = Path("data/processed")


def write_per_file_log(log_dir: Path, base_name: str, res: "Result",
                       method: str, interactive: bool) -> None:
    """Per-volume stats text file in <log_dir>/."""
    log_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        f"file              : {res.name}",
        f"method            : {method}",
        f"interactive       : {interactive}",
        f"status            : {res.status}",
        f"threshold         : {res.threshold:.6f}",
        f"peaks found       : {res.peaks_found}",
        f"voxels kept %     : {res.kept_pct:.3f}",
        f"output bytes      : {human_bytes(res.out_bytes)}",
        f"elapsed seconds   : {res.seconds:.1f}",
        f"note              : {res.note}",
    ]
    (log_dir / f"{base_name}_stats.txt").write_text("\n".join(lines) + "\n")


def write_batch_summary(log_dir: Path, results: list["Result"],
                        input_dir: Path, output_dir: Path,
                        method: str, interactive: bool,
                        ok: int, sk: int, fl: int,
                        total_bytes: int, total_time: float) -> None:
    """Plain-text dump of the rich summary table to <log_dir>/batch_summary.txt."""
    log_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "Threshold-denoise batch summary",
        "=" * 60,
        f"input dir         : {input_dir}",
        f"output dir        : {output_dir}",
        f"method            : {method}",
        f"interactive       : {interactive}",
        "",
        f"{'file':<32} {'status':>6} {'thresh':>10} {'kept%':>8} {'peaks':>6} "
        f"{'output':>10} {'time':>8}  note",
        "-" * 110,
    ]
    for r in results:
        thresh_s = f"{r.threshold:.4f}" if r.status == "ok" else "-"
        kept_s = f"{r.kept_pct:.2f}%" if r.status == "ok" else "-"
        peaks_s = str(r.peaks_found) if r.status == "ok" else "-"
        out_s = human_bytes(r.out_bytes) if r.out_bytes else "-"
        time_s = f"{r.seconds:.1f}s" if r.seconds else "-"
        lines.append(
            f"{r.name:<32} {r.status:>6} {thresh_s:>10} {kept_s:>8} {peaks_s:>6} "
            f"{out_s:>10} {time_s:>8}  {r.note}"
        )
    lines.append("-" * 110)
    lines.append(
        f"Totals: {ok} ok, {sk} skipped, {fl} failed  •  "
        f"wrote {human_bytes(total_bytes)} in {total_time:.1f}s"
    )
    (log_dir / "batch_summary.txt").write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR,
                    help=f"Input directory (default: {DEFAULT_INPUT_DIR})")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                    help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})")
    ap.add_argument("--method", choices=("valley", "otsu"), default="otsu",
                    help="Threshold method: 'otsu' (default, used in the paper) or 'valley' (bimodal valley)")
    ap.add_argument("--interactive", action="store_true",
                    help="Human-in-the-loop: open a slider UI per volume to pick the threshold")
    ap.add_argument("--no-figs", action="store_true", help="Skip saving PNG diagnostics")
    ap.add_argument("--sample", nargs="+", default=None, metavar="NAME",
                    help="Only process these samples, matched exactly by name (e.g. bar).")
    args = ap.parse_args()

    if not args.interactive:
        matplotlib.use("Agg")  # headless: never pop up windows

    console = Console()
    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir
    figs_dir = output_dir / "figs"
    log_dir = output_dir / "log"

    files = gather_inputs(input_dir)
    if args.sample:
        wanted = set(args.sample)
        files = [f for f in files if f.stem in wanted]
        missing = wanted - {f.stem for f in files}
        if missing:
            console.print(f"[red]No input for sample(s):[/red] {', '.join(sorted(missing))} under {input_dir}")
            return 1
    if not files:
        console.print(f"[yellow]No .nrrd files under[/yellow] {input_dir}")
        return 0

    mode_tag = f"method={args.method}" + (", interactive" if args.interactive else "")
    console.rule(
        f"[bold cyan]Threshold-denoise[/]  ({len(files)} volumes, {mode_tag})  "
        f"[dim]{input_dir} → {output_dir}[/]"
    )

    input_root = input_dir.resolve()
    results: list[Result] = []
    quit_early = False
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(bar_width=None),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=console,
        disable=args.interactive,  # rich's live display fights matplotlib's window
    ) as progress:
        overall = progress.add_task("all volumes", total=len(files))
        for f in files:
            progress.update(overall, description=f"[bold]{f.name}")
            try:
                rel = f.relative_to(input_root)
            except ValueError:
                # f is a symlink resolving outside input_dir, which is a
                # normal way to keep the volumes on another disk. Mirror it
                # flat instead of crashing.
                rel = Path(f.name)
            out_path = output_dir / rel
            if args.interactive:
                console.print(f"[bold cyan]→ {f.name}[/]")
            res = process_one(
                f, out_path, figs_dir,
                save_figs=not args.no_figs,
                method=args.method,
                interactive=args.interactive,
            )
            results.append(res)
            if not args.no_figs and res.status == "ok":
                write_per_file_log(log_dir, Path(res.name).stem, res,
                                   method=args.method, interactive=args.interactive)
            progress.advance(overall)
            if res.quit_requested:
                console.print("[red]Quit requested — stopping batch.[/red]")
                quit_early = True
                break
        progress.update(overall, description="[bold]done" if not quit_early else "[bold]aborted")

    table = Table(title="Threshold-denoise summary", header_style="bold magenta")
    table.add_column("File", style="cyan", no_wrap=True)
    table.add_column("Status", justify="center")
    table.add_column("Threshold", justify="right")
    table.add_column("Kept %", justify="right")
    table.add_column("Peaks", justify="right")
    table.add_column("Output", justify="right")
    table.add_column("Time", justify="right")
    table.add_column("Note", style="dim")

    sty = {"ok": "[green]ok[/]", "skip": "[yellow]skip[/]", "fail": "[red]fail[/]"}
    for r in results:
        table.add_row(
            r.name,
            sty[r.status],
            f"{r.threshold:.4f}" if r.status == "ok" else "-",
            f"{r.kept_pct:.2f}%" if r.status == "ok" else "-",
            str(r.peaks_found) if r.status == "ok" else "-",
            human_bytes(r.out_bytes) if r.out_bytes else "-",
            f"{r.seconds:.1f}s" if r.seconds else "-",
            r.note,
        )
    console.print(table)

    ok = sum(r.status == "ok" for r in results)
    sk = sum(r.status == "skip" for r in results)
    fl = sum(r.status == "fail" for r in results)
    total_bytes = sum(r.out_bytes for r in results if r.status == "ok")
    total_time = sum(r.seconds for r in results if r.status == "ok")
    console.print(
        f"[bold]Totals:[/bold] [green]{ok} ok[/], [yellow]{sk} skipped[/], [red]{fl} failed[/]  •  "
        f"wrote {human_bytes(total_bytes)} in {total_time:.1f}s"
    )

    if not args.no_figs:
        write_batch_summary(log_dir, results, input_dir, output_dir,
                            method=args.method, interactive=args.interactive,
                            ok=ok, sk=sk, fl=fl,
                            total_bytes=total_bytes, total_time=total_time)
        console.print(f"[dim]Batch summary → {log_dir}/batch_summary.txt[/dim]")

    return 0 if fl == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
