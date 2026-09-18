#!/usr/bin/env python3
"""United-degree denoise for Gabor pointclouds (pipeline step v3).

For each point i: united_degree(i) = fraction of its R-radius neighbors whose
tangent direction is similar (|dot(d_i, d_j)| ≥ sim_th). Points with low
united_degree (incoherent neighborhoods) are dropped — typically noise or
yarn crossings rather than continuous yarn segments. Surviving points form
cleaner yarn-centerline-candidate clouds.

Default: batch every `*_pointcloud_full_*.npz` written by v2 under data/gabor/ and
save the filtered cloud under the same file name to data/denoised/. Use `--file`
for a single npz (then launches napari for inspection).

The CONFIG defaults below are the values used for the paper results. With them
this step keeps 98.5% to 99.6% of the points on our dataset.

PNG diagnostics → <output>/figs/, text logs → <output>/log/.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")  # set in main() override if --file (napari pop-up)

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
from scipy.spatial import cKDTree

try:
    import napari

    HAS_NAPARI = True
except ImportError:
    HAS_NAPARI = False


# ============================================================
# CONFIG — adjust per-data scale if needed
# ============================================================
DEFAULT_INPUT_DIR = Path("data/gabor")      # v2 output, *_pointcloud_full_*.npz
DEFAULT_OUTPUT_DIR = Path("data/denoised")  # same file names as the input

RADIUS = 10.0              # voxel-units neighbor search radius
SIM_TH = 0.85              # |dot(d_i, d_j)| ≥ SIM_TH → considered "united"
UNITED_KEEP_TH = 0.85      # keep points whose united_degree > threshold
MIN_NEIGHBORS = 1          # if neighbor count < this, united_degree = 0
WORKERS = 32               # KDTree query workers (0 = ~1/3 cpu cores)
CHUNK_SIZE = 20_000        # anchor points per query batch
MAX_NEIGHBORS = 512        # cap neighbors per anchor (memory/compute guard)
FIG_VIZ_CAP = 50_000       # scatter-plot point cap in diagnostic PNG
# ============================================================


@dataclass
class FileResult:
    name: str
    status: str  # "ok" | "skip" | "fail"
    n_input: int = 0
    n_kept: int = 0
    keep_pct: float = 0.0
    united_mean: float = 0.0
    united_p50: float = 0.0
    united_p95: float = 0.0
    saturated_pct: float = 0.0
    out_bytes: int = 0
    seconds: float = 0.0
    note: str = ""


def normalize_rows(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.where(n > 1e-8, n, 1.0)


def compute_united_degree(
    points: np.ndarray, directions: np.ndarray,
    radius: float, sim_th: float, min_neighbors: int,
    workers: int, chunk_size: int, max_neighbors: int,
    progress: Optional[Progress] = None, task_id=None,
) -> tuple[np.ndarray, int]:
    """Return (united_per_point, saturated_count)."""
    N = points.shape[0]
    tree = cKDTree(points)
    united = np.zeros(N, dtype=np.float32)

    cpu_count = os.cpu_count() or 1
    if workers <= 0:
        workers = max(1, cpu_count // 3)
    workers = max(1, min(workers, cpu_count))
    chunk_size = max(1000, int(chunk_size))
    max_neighbors = max(1, int(max_neighbors))
    k_query = max_neighbors + 1

    starts = list(range(0, N, chunk_size))
    if progress is not None and task_id is not None:
        progress.update(task_id, total=len(starts), completed=0)

    saturated_total = 0
    for s in starts:
        e = min(s + chunk_size, N)
        p_chunk = points[s:e]

        dists, idx = tree.query(
            p_chunk, k=k_query,
            distance_upper_bound=radius, workers=workers,
        )
        if idx.ndim == 1:
            idx = idx[:, None]
            dists = dists[:, None]

        row_ids = np.arange(s, e, dtype=np.int64)[:, None]
        valid = (idx < N) & (idx != row_ids) & np.isfinite(dists)
        neighbor_count = valid.sum(axis=1)
        saturated_total += int(np.count_nonzero(neighbor_count == max_neighbors))

        idx_safe = np.where(valid, idx, 0)
        nbr_dirs = directions[idx_safe]
        sim = np.abs(np.einsum("bkc,bc->bk", nbr_dirs, directions[s:e]))
        good = (sim >= sim_th) & valid

        united_chunk = np.divide(
            good.sum(axis=1),
            neighbor_count,
            out=np.zeros(e - s, dtype=np.float32),
            where=neighbor_count >= min_neighbors,
        ).astype(np.float32)
        united[s:e] = united_chunk

        if progress is not None and task_id is not None:
            progress.update(task_id, advance=1)

    return united, saturated_total


def save_diag_figs(
    figs_dir: Path, base_name: str, points_in: np.ndarray, united: np.ndarray,
    keep_mask: np.ndarray, keep_th: float,
) -> None:
    """Histogram of united_degree + 3-axis projections (kept vs dropped)."""
    figs_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.4])

    # Row 0: histogram (spans 3 cols)
    ax_h = fig.add_subplot(gs[0, :])
    ax_h.hist(united, bins=120, color="steelblue", alpha=0.85, log=True)
    ax_h.axvline(keep_th, color="red", lw=2, label=f"keep > {keep_th}")
    n_kept = int(keep_mask.sum())
    ax_h.set_title(
        f"{base_name}   united-degree distribution   "
        f"kept {n_kept:,} / {len(united):,} ({100*n_kept/max(1,len(united)):.1f}%)",
        fontsize=11,
    )
    ax_h.set_xlabel("united_degree")
    ax_h.set_ylabel("count (log)")
    ax_h.legend()
    ax_h.grid(alpha=0.3)

    # Row 1: 3 projections
    rng = np.random.default_rng(42)
    if len(points_in) > FIG_VIZ_CAP:
        viz_idx = rng.choice(len(points_in), FIG_VIZ_CAP, replace=False)
    else:
        viz_idx = np.arange(len(points_in))
    pv = points_in[viz_idx]
    uv = united[viz_idx]
    keep_v = keep_mask[viz_idx]

    pairs = [("YX", 2, 1), ("ZX", 2, 0), ("ZY", 1, 0)]
    for col, (label, ix, iy) in enumerate(pairs):
        ax = fig.add_subplot(gs[1, col])
        ax.scatter(pv[~keep_v, ix], pv[~keep_v, iy], s=0.5, c="lightgray",
                   alpha=0.4, label="dropped")
        sc = ax.scatter(pv[keep_v, ix], pv[keep_v, iy], s=0.6,
                        c=uv[keep_v], cmap="inferno", vmin=0, vmax=1, alpha=0.9)
        ax.set_xlabel(label[0]); ax.set_ylabel(label[1])
        ax.set_title(f"projection {label}")
        ax.set_aspect("equal")
        ax.invert_yaxis()
        if col == 2:
            fig.colorbar(sc, ax=ax, fraction=0.04, label="united_degree")

    fig.tight_layout()
    fig.savefig(figs_dir / f"{base_name}_united.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def write_stats_txt(
    log_dir: Path, base_name: str, res: FileResult, params: dict,
) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        f"file              : {res.name}",
        f"input points      : {res.n_input}",
        f"kept points       : {res.n_kept}  ({res.keep_pct:.3f}%)",
        f"radius            : {params['radius']}",
        f"sim_th            : {params['sim_th']}",
        f"united_keep_th    : {params['united_keep_th']}",
        f"min_neighbors     : {params['min_neighbors']}",
        f"max_neighbors     : {params['max_neighbors']}",
        f"chunk_size        : {params['chunk_size']}",
        f"workers           : {params['workers']}",
        f"united mean       : {res.united_mean:.4f}",
        f"united median     : {res.united_p50:.4f}",
        f"united p95        : {res.united_p95:.4f}",
        f"saturated %       : {res.saturated_pct:.2f}  (points hitting max_neighbors)",
        f"output bytes      : {res.out_bytes}",
        f"elapsed seconds   : {res.seconds:.1f}",
        f"note              : {res.note}",
    ]
    (log_dir / f"{base_name}_stats.txt").write_text("\n".join(lines) + "\n")


def write_batch_summary(
    log_dir: Path, results: list[FileResult], input_dir: Path, output_dir: Path,
    params: dict,
) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    ok = sum(r.status == "ok" for r in results)
    sk = sum(r.status == "skip" for r in results)
    fl = sum(r.status == "fail" for r in results)
    lines = [
        "United-degree denoise batch summary",
        "=" * 60,
        f"input dir         : {input_dir}",
        f"output dir        : {output_dir}",
        f"radius            : {params['radius']}",
        f"sim_th            : {params['sim_th']}",
        f"united_keep_th    : {params['united_keep_th']}",
        f"min_neighbors     : {params['min_neighbors']}",
        f"max_neighbors     : {params['max_neighbors']}",
        "",
        f"{'file':<60} {'status':>6} {'input':>12} {'kept':>12} {'keep%':>7} {'time':>7}",
        "-" * 108,
    ]
    for r in results:
        lines.append(
            f"{r.name:<60} {r.status:>6} "
            f"{r.n_input:>12,} {r.n_kept:>12,} {r.keep_pct:>6.2f}% "
            f"{r.seconds:>6.1f}s"
        )
    lines += [
        "-" * 108,
        f"Totals: {ok} ok, {sk} skipped, {fl} failed",
    ]
    (log_dir / "batch_summary.txt").write_text("\n".join(lines) + "\n")


def load_npz(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return (points, directions, all_extra_keys_dict)."""
    data = np.load(path)
    if "points" not in data.files or "directions" not in data.files:
        raise KeyError("NPZ must have 'points' and 'directions'")
    points = np.asarray(data["points"], dtype=np.float32)
    directions = np.asarray(data["directions"], dtype=np.float32)
    extras = {k: data[k] for k in data.files if k not in ("points", "directions")}
    data.close()
    if points.shape[0] != directions.shape[0]:
        raise ValueError("'points' and 'directions' length mismatch")
    if points.shape[1] != 3 or directions.shape[1] != 3:
        raise ValueError("'points'/'directions' must be (N, 3)")
    return points, normalize_rows(directions), extras


def filter_and_save(
    out_path: Path,
    points: np.ndarray, directions: np.ndarray, extras: dict,
    keep_mask: np.ndarray,
) -> int:
    """Save filtered NPZ, preserving auxiliary per-point arrays (energy, linearity, ...)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_total = points.shape[0]
    payload = {"points": points[keep_mask], "directions": directions[keep_mask]}
    for k, v in extras.items():
        if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == n_total:
            payload[k] = v[keep_mask]
        else:
            payload[k] = v
    np.savez_compressed(out_path, **payload)
    return out_path.stat().st_size


def process_one(
    input_path: Path, output_path: Path, figs_dir: Path, log_dir: Path,
    params: dict, save_figs: bool, progress: Optional[Progress] = None,
) -> FileResult:
    res = FileResult(name=input_path.name, status="fail")
    if output_path.exists():
        res.status = "skip"
        res.note = "exists"
        res.out_bytes = output_path.stat().st_size
        return res

    t0 = time.monotonic()
    try:
        points, directions, extras = load_npz(input_path)
        res.n_input = points.shape[0]

        inner_task = None
        if progress is not None:
            inner_task = progress.add_task(
                f"[dim]{input_path.stem[:40]}[/]", total=1, transient=True,
            )

        united, saturated = compute_united_degree(
            points, directions,
            radius=params["radius"], sim_th=params["sim_th"],
            min_neighbors=params["min_neighbors"], workers=params["workers"],
            chunk_size=params["chunk_size"], max_neighbors=params["max_neighbors"],
            progress=progress, task_id=inner_task,
        )

        if inner_task is not None:
            progress.remove_task(inner_task)

        keep_mask = united > params["united_keep_th"]
        n_kept = int(keep_mask.sum())
        res.n_kept = n_kept
        res.keep_pct = 100.0 * n_kept / max(1, res.n_input)
        res.united_mean = float(united.mean())
        res.united_p50 = float(np.median(united))
        res.united_p95 = float(np.percentile(united, 95))
        res.saturated_pct = 100.0 * saturated / max(1, res.n_input)

        if n_kept == 0:
            res.status = "fail"
            res.note = "0 points after filter"
            res.seconds = time.monotonic() - t0
            return res

        res.out_bytes = filter_and_save(output_path, points, directions, extras, keep_mask)
        res.seconds = time.monotonic() - t0
        res.status = "ok"

        if save_figs:
            save_diag_figs(
                figs_dir, input_path.stem, points, united, keep_mask,
                keep_th=params["united_keep_th"],
            )
            write_stats_txt(log_dir, input_path.stem, res, params)
    except Exception as e:
        res.status = "fail"
        res.note = str(e)[:80]
        res.seconds = time.monotonic() - t0
    return res


def gather_inputs(input_dir: Path, prefer_full: bool) -> list[Path]:
    if not input_dir.is_dir():
        return []
    sub = sorted(input_dir.glob("*_pointcloud_*_sub_*.npz"))
    full = sorted(input_dir.glob("*_pointcloud_full_*.npz"))
    by_stem: dict[str, Path] = {}
    primary, fallback = (full, sub) if prefer_full else (sub, full)
    for f in primary:
        by_stem.setdefault(f.name.split("_pointcloud_")[0], f)
    for f in fallback:
        by_stem.setdefault(f.name.split("_pointcloud_")[0], f)
    return [by_stem[s] for s in sorted(by_stem)]


def human_bytes(n: int) -> str:
    x = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if x < 1024 or unit == "TB":
            return f"{int(x)} B" if unit == "B" else f"{x:.1f} {unit}"
        x /= 1024
    return f"{x:.1f} TB"


def launch_napari_viz(
    points_orig: np.ndarray, united: np.ndarray, keep_mask: np.ndarray,
    keep_th: float, name: str,
) -> None:
    """For --file mode: show original (gray) + kept (colored by united_degree)."""
    cmap = plt.get_cmap("inferno")
    viewer = napari.Viewer(ndisplay=3, title=f"v3 united-denoise — {name}")
    if (~keep_mask).any():
        viewer.add_points(
            points_orig[~keep_mask].astype(np.float32),
            name=f"dropped ({(~keep_mask).sum():,})",
            size=1.5, face_color="gray", opacity=0.25, visible=False,
        )
    kept_colors = cmap(np.clip(united[keep_mask], 0, 1))[:, :3]
    viewer.add_points(
        points_orig[keep_mask].astype(np.float32),
        name=f"kept (united > {keep_th})  [{keep_mask.sum():,}]",
        size=1.5, face_color=kept_colors, opacity=0.9,
    )
    print(f"\n=== napari controls ===")
    print(f"  yellow→red = high united_degree; toggle 'dropped' to see noise.")
    napari.run()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--file", type=Path, default=None,
                    help="Process a single .npz and launch napari for inspection.")
    ap.add_argument("--prefer-full", action="store_true", default=True,
                    help="Operate on _full_*.npz. v2 no longer emits _sub_*.npz so this is "
                         "the only mode that gets any input; flag kept for backwards-compat.")
    ap.add_argument("--radius", type=float, default=RADIUS)
    ap.add_argument("--sim-th", type=float, default=SIM_TH)
    ap.add_argument("--united-keep-th", type=float, default=UNITED_KEEP_TH)
    ap.add_argument("--min-neighbors", type=int, default=MIN_NEIGHBORS)
    ap.add_argument("--max-neighbors", type=int, default=MAX_NEIGHBORS)
    ap.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--no-figs", action="store_true",
                    help="Skip PNG diagnostics + per-volume stats txt.")
    ap.add_argument("--n-jobs", type=int, default=1,
                    help="Parallel file-level workers (default 1 = sequential). "
                         "Each worker still uses --workers threads in its KDTree query, "
                         "so total CPU = n_jobs × workers. With many large files and "
                         "free RAM (~5GB/file), set 2–4 to roughly halve wall time.")
    ap.add_argument("--sample", nargs="+", default=None, metavar="NAME",
                    help="Only process these samples, matched exactly by name (e.g. bar).")
    args = ap.parse_args()

    params = {
        "radius": args.radius, "sim_th": args.sim_th,
        "united_keep_th": args.united_keep_th,
        "min_neighbors": args.min_neighbors,
        "max_neighbors": args.max_neighbors,
        "chunk_size": args.chunk_size, "workers": args.workers,
    }

    console = Console()
    figs_dir = args.output_dir / "figs"
    log_dir = args.output_dir / "log"

    if args.file is not None:
        if not args.file.exists():
            console.print(f"[red]Not found:[/red] {args.file}")
            return 1
        # Single-file: napari pop-up + diagnostics
        if HAS_NAPARI:
            matplotlib.use("Qt5Agg")
        console.rule(f"[bold cyan]Single:[/] {args.file.name}")
        out_path = args.output_dir / args.file.name
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.description}"),
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            outer = progress.add_task("united-degree chunks", total=1)
            res = process_one(
                args.file, out_path, figs_dir, log_dir, params,
                save_figs=not args.no_figs, progress=progress,
            )
            progress.update(outer, completed=1)
        console.print(
            f"  [bold]{res.status}[/]  kept {res.n_kept:,} / {res.n_input:,} "
            f"({res.keep_pct:.2f}%) in {res.seconds:.1f}s"
        )
        if res.status == "ok" and HAS_NAPARI:
            points, directions, _ = load_npz(args.file)
            united, _ = compute_united_degree(
                points, directions,
                radius=args.radius, sim_th=args.sim_th,
                min_neighbors=args.min_neighbors, workers=args.workers,
                chunk_size=args.chunk_size, max_neighbors=args.max_neighbors,
            )
            keep_mask = united > args.united_keep_th
            launch_napari_viz(points, united, keep_mask,
                              args.united_keep_th, args.file.stem)
        return 0 if res.status == "ok" else 2

    # ─── batch mode ────────────────────────────────────────────────────
    files = gather_inputs(args.input_dir, prefer_full=args.prefer_full)
    if args.sample:
        wanted = set(args.sample)
        files = [f for f in files if f.name.split("_pointcloud_")[0] in wanted]
        missing = wanted - {f.name.split("_pointcloud_")[0] for f in files}
        if missing:
            console.print(f"[red]No input for sample(s):[/red] {', '.join(sorted(missing))} under {args.input_dir}")
            return 1
    if not files:
        console.print(f"[yellow]No pointcloud .npz under[/yellow] {args.input_dir}")
        return 0

    console.rule(
        f"[bold cyan]United-degree denoise[/]  ({len(files)} pointclouds)  "
        f"[dim]{args.input_dir} → {args.output_dir}[/]"
    )

    results: list[FileResult] = []
    n_jobs = max(1, args.n_jobs)
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
        overall = progress.add_task(
            f"all files (n_jobs={n_jobs})", total=len(files)
        )
        if n_jobs == 1:
            for f in files:
                progress.update(overall, description=f"[bold]{f.name[:55]}")
                out_path = args.output_dir / f.name
                res = process_one(
                    f, out_path, figs_dir, log_dir, params,
                    save_figs=not args.no_figs, progress=progress,
                )
                results.append(res)
                progress.advance(overall)
        else:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            with ProcessPoolExecutor(max_workers=n_jobs) as ex:
                futures = {
                    ex.submit(
                        process_one, f, args.output_dir / f.name,
                        figs_dir, log_dir, params,
                        not args.no_figs, None,
                    ): f
                    for f in files
                }
                for fut in as_completed(futures):
                    f = futures[fut]
                    progress.update(overall, description=f"[bold]done: {f.name[:55]}")
                    try:
                        res = fut.result()
                    except Exception as e:
                        res = FileResult(name=f.name, status="fail", note=str(e)[:80])
                    results.append(res)
                    progress.advance(overall)
            results.sort(key=lambda r: r.name)
        progress.update(overall, description="[bold]done")

    table = Table(title="United-degree denoise summary", header_style="bold magenta")
    table.add_column("File", style="cyan", no_wrap=True)
    table.add_column("Status", justify="center")
    table.add_column("Input", justify="right")
    table.add_column("Kept", justify="right")
    table.add_column("Keep %", justify="right")
    table.add_column("Sat %", justify="right")
    table.add_column("Size", justify="right")
    table.add_column("Time", justify="right")
    sty = {"ok": "[green]ok[/]", "skip": "[yellow]skip[/]", "fail": "[red]fail[/]"}
    for r in results:
        table.add_row(
            r.name[:55], sty[r.status],
            f"{r.n_input:,}" if r.n_input else "-",
            f"{r.n_kept:,}" if r.status == "ok" else "-",
            f"{r.keep_pct:.2f}%" if r.status == "ok" else "-",
            f"{r.saturated_pct:.1f}%" if r.status == "ok" else "-",
            human_bytes(r.out_bytes) if r.out_bytes else "-",
            f"{r.seconds:.1f}s" if r.seconds else "-",
        )
    console.print(table)

    ok = sum(r.status == "ok" for r in results)
    sk = sum(r.status == "skip" for r in results)
    fl = sum(r.status == "fail" for r in results)
    console.print(
        f"[bold]Totals:[/] [green]{ok} ok[/], [yellow]{sk} skipped[/], [red]{fl} failed[/]"
    )

    if not args.no_figs:
        write_batch_summary(log_dir, results, args.input_dir, args.output_dir, params)
        console.print(f"[dim]Figs → {figs_dir}  •  Log → {log_dir}[/dim]")

    return 0 if fl == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
