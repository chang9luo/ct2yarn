#!/usr/bin/env python3
"""Voxel-binning downsample for denoised Gabor pointclouds (pipeline step v4).

For each input pointcloud, partition the bbox into cubic cells of size
`DS * voxel_size` where `voxel_size = (V_bbox / N) ** (1/3)` (i.e. the mean
inter-point spacing assuming uniform distribution); keep the FIRST point of
each occupied cell. For uniformly-distributed clouds this gives ≈ N/DS³
points; for clustered 1D yarn structures the reduction is typically much
more aggressive because most cells are empty.

Default: batch every `.npz` written by v3 under data/denoised/ → write
`<input stem>_ds{DS}.npz` to data/binned/. Use `--file` for a single npz.

PNG diagnostics → <output>/figs/, text logs → <output>/log/.
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
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


DEFAULT_INPUT_DIR = Path("data/denoised")
DEFAULT_OUTPUT_DIR = Path("data/binned")
DEFAULT_DS = 4


@dataclass
class FileResult:
    name: str
    status: str = "fail"
    n_input: int = 0
    n_kept: int = 0
    voxel_size: float = 0.0
    cell_size: float = 0.0
    ds: float = 0.0
    out_bytes: int = 0
    seconds: float = 0.0
    note: str = ""


def voxel_bin_indices(points: np.ndarray, cell_size: float) -> np.ndarray:
    """Return indices of one representative point per occupied cell.

    Flatten the (i, j, k) cell key to a single int64 via per-axis spans, then np.unique → first index
    per cell. Order-stable (always keeps the input's first occurrence per cell).
    """
    keys = np.floor(points / cell_size).astype(np.int64)
    lo = keys.min(axis=0)
    span = keys.max(axis=0) - lo + 1
    flat = (
        (keys[:, 0] - lo[0]) * span[1] * span[2]
        + (keys[:, 1] - lo[1]) * span[2]
        + (keys[:, 2] - lo[2])
    )
    _, idx = np.unique(flat, return_index=True)
    return idx


def compute_voxel_size(points: np.ndarray) -> tuple[float, np.ndarray]:
    """voxel_size = (V_bbox / N) ** (1/3). Returns (voxel_size, bbox_extent)."""
    bbox = points.max(axis=0) - points.min(axis=0)
    voxel_size = float((np.prod(bbox) / max(len(points), 1)) ** (1.0 / 3.0))
    return voxel_size, bbox


def load_npz(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    src = np.load(path, allow_pickle=False)
    points = src["points"].astype(np.float32)
    if "directions" in src.files:
        directions = src["directions"].astype(np.float32)
    elif "dirs" in src.files:
        directions = src["dirs"].astype(np.float32)
    else:
        directions = np.zeros((len(points), 3), dtype=np.float32)
    extras = {k: src[k] for k in src.files if k not in ("points", "directions", "dirs")}
    return points, directions, extras


def save_bin_figs(
    figs_dir: Path, base_name: str, points_in: np.ndarray,
    points_out: np.ndarray, voxel_size: float, cell_size: float, ds: float,
) -> None:
    figs_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles = ["XY", "XZ", "YZ"]
    proj_axes = [(0, 1), (0, 2), (1, 2)]
    for ax, (a, b), title in zip(axes, proj_axes, titles):
        ax.scatter(points_in[::100, a], points_in[::100, b],
                   s=0.5, c="lightgray", label=f"input ({len(points_in):,})", alpha=0.5)
        ax.scatter(points_out[:, a], points_out[:, b],
                   s=1.5, c="crimson", label=f"binned ({len(points_out):,})", alpha=0.8)
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.legend(fontsize=8, markerscale=4)
        ax.grid(True, alpha=0.3)
    fig.suptitle(
        f"{base_name}  •  DS={ds}, voxel_size={voxel_size:.2f}, cell={cell_size:.2f} vox",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(figs_dir / f"{base_name}_bin.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def write_stats_txt(log_dir: Path, base_name: str, res: FileResult) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        f"file              : {res.name}",
        f"DS                : {res.ds}",
        f"voxel_size        : {res.voxel_size:.4f}",
        f"cell_size         : {res.cell_size:.4f}",
        f"input points      : {res.n_input}",
        f"output points     : {res.n_kept}",
        f"reduction factor  : {res.n_input / max(res.n_kept, 1):.2f}",
        f"out size          : {res.out_bytes} bytes",
        f"elapsed seconds   : {res.seconds:.3f}",
    ]
    (log_dir / f"{base_name}_stats.txt").write_text("\n".join(lines) + "\n")


def write_batch_summary(
    log_dir: Path, results: list[FileResult], input_dir: Path, output_dir: Path,
    ds: float, ok: int, sk: int, fl: int,
) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "Voxel-binning batch summary",
        "=" * 60,
        f"input dir         : {input_dir}",
        f"output dir        : {output_dir}",
        f"DS                : {ds}",
        "",
        f"{'file':<60} {'status':>6} {'n_in':>12} {'n_out':>10} "
        f"{'voxel':>7} {'cell':>7} {'reduce':>7} {'time':>7}",
        "-" * 130,
    ]
    for r in results:
        if r.status == "ok":
            red = f"{r.n_input / max(r.n_kept, 1):.1f}×"
            lines.append(
                f"{r.name:<60} {r.status:>6} {r.n_input:>12,} {r.n_kept:>10,} "
                f"{r.voxel_size:>7.2f} {r.cell_size:>7.2f} {red:>7} {f'{r.seconds:.1f}s':>7}"
            )
        else:
            lines.append(f"{r.name:<60} {r.status:>6}  {r.note}")
    lines.append("-" * 130)
    lines.append(f"Totals: {ok} ok, {sk} skipped, {fl} failed")
    (log_dir / "batch_summary.txt").write_text("\n".join(lines) + "\n")


def process_one(
    input_path: Path, output_path: Path, figs_dir: Path, log_dir: Path,
    ds: float, save_figs: bool,
) -> FileResult:
    res = FileResult(name=input_path.name, ds=ds)
    if output_path.exists():
        res.status = "skip"
        res.note = "exists"
        res.out_bytes = output_path.stat().st_size
        return res

    t0 = time.monotonic()
    try:
        points, directions, extras = load_npz(input_path)
        res.n_input = points.shape[0]
        voxel_size, _bbox = compute_voxel_size(points)
        cell_size = ds * voxel_size
        res.voxel_size = voxel_size
        res.cell_size = cell_size

        idx = voxel_bin_indices(points, cell_size)
        points_out = np.ascontiguousarray(points[idx])
        dirs_out = np.ascontiguousarray(directions[idx])
        res.n_kept = len(idx)

        if res.n_kept == 0:
            res.status = "fail"
            res.note = "0 points after binning"
            res.seconds = time.monotonic() - t0
            return res

        payload: dict = {"points": points_out, "directions": dirs_out}
        for k, v in extras.items():
            if hasattr(v, "shape") and v.ndim >= 1 and v.shape[0] == res.n_input:
                payload[k] = v[idx]
            else:
                payload[k] = v
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path, **payload)
        res.out_bytes = output_path.stat().st_size
        res.seconds = time.monotonic() - t0
        res.status = "ok"

        if save_figs:
            save_bin_figs(figs_dir, input_path.stem, points, points_out,
                          voxel_size, cell_size, ds)
            write_stats_txt(log_dir, input_path.stem, res)
    except Exception as e:
        res.status = "fail"
        res.note = str(e)[:80]
        res.seconds = time.monotonic() - t0
    return res


def gather_inputs(input_dir: Path) -> list[Path]:
    if not input_dir.is_dir():
        return []
    return sorted({f.resolve() for f in input_dir.rglob("*.npz") if f.is_file()})


def output_name(input_name: str, ds: float) -> str:
    """Strip a trailing .npz and append _ds{int(ds)}.npz."""
    stem = input_name[:-4] if input_name.endswith(".npz") else input_name
    ds_tag = str(int(ds)) if abs(ds - int(ds)) < 1e-9 else f"{ds:g}"
    return f"{stem}_ds{ds_tag}.npz"


def human_bytes(n: int) -> str:
    x = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if x < 1024 or unit == "TB":
            return f"{int(x)} B" if unit == "B" else f"{x:.1f} {unit}"
        x /= 1024
    return f"{x:.1f} TB"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR,
                    help=f"Default: {DEFAULT_INPUT_DIR}")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                    help=f"Default: {DEFAULT_OUTPUT_DIR}")
    ap.add_argument("--file", type=Path, default=None,
                    help="Process a single .npz instead of batch.")
    ap.add_argument("--ds", type=float, default=DEFAULT_DS,
                    help=f"Linear cell size multiplier × voxel_size (default {DEFAULT_DS}).")
    ap.add_argument("--no-figs", action="store_true",
                    help="Skip PNG diagnostics + per-file stats text.")
    ap.add_argument("--sample", nargs="+", default=None, metavar="NAME",
                    help="Only process these samples, matched exactly by name (e.g. bar).")
    args = ap.parse_args()

    console = Console()
    figs_dir = args.output_dir / "figs"
    log_dir = args.output_dir / "log"

    if args.file is not None:
        files = [args.file.resolve()]
        if not args.file.is_file():
            console.print(f"[red]Not found:[/red] {args.file}")
            return 1
    else:
        files = gather_inputs(args.input_dir)
        if args.sample:
            wanted = set(args.sample)
            files = [f for f in files if f.name.split("_pointcloud_")[0] in wanted]
            missing = wanted - {f.name.split("_pointcloud_")[0] for f in files}
            if missing:
                console.print(f"[red]No input for sample(s):[/red] {', '.join(sorted(missing))} under {args.input_dir}")
                return 1
        if not files:
            console.print(f"[yellow]No .npz under[/yellow] {args.input_dir}")
            return 0

    console.rule(
        f"[bold cyan]Voxel-binning[/]  ({len(files)} files, DS={args.ds})  "
        f"[dim]{args.input_dir} → {args.output_dir}[/]"
    )

    results: list[FileResult] = []
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
    ) as progress:
        overall = progress.add_task("[bold]all files", total=len(files))
        for f in files:
            progress.update(overall, description=f"[bold]{f.name[:48]}")
            out_path = args.output_dir / output_name(f.name, args.ds)
            res = process_one(f, out_path, figs_dir, log_dir, args.ds,
                              save_figs=not args.no_figs)
            results.append(res)
            progress.advance(overall)
        progress.update(overall, description="[bold]done")

    table = Table(title="Voxel-binning summary", header_style="bold magenta")
    table.add_column("File", style="cyan", no_wrap=True)
    table.add_column("Status", justify="center")
    table.add_column("n_in", justify="right")
    table.add_column("n_out", justify="right")
    table.add_column("voxel", justify="right")
    table.add_column("cell", justify="right")
    table.add_column("Δ×", justify="right")
    table.add_column("Time", justify="right")
    table.add_column("Note", style="dim")
    sty = {"ok": "[green]ok[/]", "skip": "[yellow]skip[/]", "fail": "[red]fail[/]"}
    for r in results:
        if r.status == "ok":
            red = f"{r.n_input/max(r.n_kept,1):.1f}×"
            table.add_row(
                r.name, sty[r.status],
                f"{r.n_input:,}", f"{r.n_kept:,}",
                f"{r.voxel_size:.2f}", f"{r.cell_size:.2f}",
                red, f"{r.seconds:.1f}s", r.note,
            )
        else:
            table.add_row(r.name, sty.get(r.status, r.status),
                          "-", "-", "-", "-", "-", "-", r.note)
    console.print(table)

    ok = sum(r.status == "ok" for r in results)
    sk = sum(r.status == "skip" for r in results)
    fl = sum(r.status == "fail" for r in results)
    console.print(
        f"[bold]Totals:[/bold] [green]{ok} ok[/], [yellow]{sk} skipped[/], [red]{fl} failed[/]"
    )
    if not args.no_figs:
        write_batch_summary(log_dir, results, args.input_dir, args.output_dir,
                            args.ds, ok, sk, fl)
        console.print(f"[dim]Figs → {figs_dir}  •  Log → {log_dir}[/dim]")

    return 0 if fl == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
