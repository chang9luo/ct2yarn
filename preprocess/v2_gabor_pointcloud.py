"""Gabor structure-tensor pointcloud extraction (Sec. 3.2 of the paper).

Default: batch every .nrrd in DEFAULT_INPUT_DIR → .npz pointclouds in
DEFAULT_OUTPUT_DIR. With `--file <path>`, process one volume and open napari.

The Gabor parameters (sigma, freq, kernel_size) are chosen by --mode:
  manual  (default) always use the CONFIG values (SIGMA_GABOR=7, FREQ=0.05) and
          ignore the yaml files. This is the setting used for all results in the paper.
  auto    read <input>/params/<stem>.yaml written by v1_estimate_yarn_diameter.py,
          falling back to the CONFIG values below when the yaml is missing.

Existing outputs are NOT overwritten.

Usage:
    python preprocess/v2_gabor_pointcloud.py                               # batch, paper parameters
    python preprocess/v2_gabor_pointcloud.py --mode auto                   # batch, per-volume params from v1
    python preprocess/v2_gabor_pointcloud.py --file data/processed/X.nrrd  # debug one volume in napari
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib  # backend is chosen in main() (Qt5Agg for --file, Agg for batch)
import nrrd
import numpy as np
import torch
import torch.fft
import torch.nn.functional as F
import yaml
from rich.console import Console
from rich.table import Table
from scipy.ndimage import gaussian_filter as gaussian_filter_1d

try:
    import napari
    HAS_NAPARI = True
except ImportError:
    HAS_NAPARI = False


# ============================================================
# CONFIG — tune these per data scale
# ============================================================
# --- I/O defaults ---
DEFAULT_INPUT_DIR = Path("data/processed")
DEFAULT_OUTPUT_DIR = Path("data/gabor")

# --- Priority keywords (case-insensitive substring match on filename) ---
# Files whose name contains any of these strings jump to the FRONT of
# the batch queue, in the order listed below (first key has highest
# priority).  Everything else keeps the default alphabetical order.
PRIORITY_KEYS: list[str] = []

# --- Gabor structure-tensor parameters ---
SIGMA_GABOR = 7              # Gaussian envelope ≈ yarn radius (vox)
FREQ = 0.05                  # carrier λ = 1/freq ≈ 1.5× yarn diameter
NUM_DIRECTIONS = 128          # Fibonacci-hemisphere sampling
KERNEL_RADIUS = int(3.0 * SIGMA_GABOR)
KERNEL_SIZE = 2 * KERNEL_RADIUS + 1

# Tensor smoothing scale = ratio × per-volume σ.
# Same ratio applied whether σ comes from yaml or from CONFIG defaults.
SIGMA_NEIGHBOR_RATIO = 1.0

# --- Chunking (VRAM control) ---
GABOR_CHUNK_H: Optional[int] = None   # None → auto-pick inside the algo
EIGEN_CHUNK_H = 400                   # H-slab size for smooth + eigen step

# --- Masking thresholds ---
INTENSITY_THRESH = 4 * 8000.0    # Gabor-energy (intensity) threshold
LINEARITY_THRESH = 0.03      # linearity threshold ∈ (0, 1)

# --- Output sampling ---
DOWNSAMPLING_FACTOR = 4      # subsample factor used ONLY for napari debug preview (vectors layer)

# --- Per-volume params directory (next to inputs, written by estimate_yarn_diameter.py) ---
PARAMS_SUBDIR = "params"
# ============================================================


@dataclass
class GaborParams:
    """Per-volume Gabor params; either loaded from yaml or fallen-back to CONFIG."""
    sigma: int
    freq: float
    kernel_size: int
    source: str  # "yaml:<path>", "config" or "manual"


def config_params() -> GaborParams:
    return GaborParams(SIGMA_GABOR, FREQ, KERNEL_SIZE, source="config")


def load_params_yaml(nrrd_path: Path) -> Optional[GaborParams]:
    """Look for <nrrd parent>/PARAMS_SUBDIR/<stem>.yaml and parse it."""
    yaml_path = nrrd_path.parent / PARAMS_SUBDIR / f"{nrrd_path.stem}.yaml"
    if not yaml_path.exists():
        return None
    try:
        doc = yaml.safe_load(yaml_path.read_text()) or {}
        g = doc.get("gabor", {})
        sigma = int(g["sigma"])
        freq = float(g["freq"])
        kernel = int(g["kernel_size"])
        return GaborParams(sigma, freq, kernel, source=f"yaml:{yaml_path}")
    except (KeyError, TypeError, ValueError, yaml.YAMLError):
        return None


def resolve_params(nrrd_path: Path, mode: str) -> GaborParams:
    """Gabor params for one volume.

    manual: always SIGMA_GABOR / FREQ / KERNEL_SIZE from CONFIG (the paper
            setting). The params yaml files are not read.
    auto:   the per-volume yaml from v1_estimate_yarn_diameter.py if present,
            otherwise the CONFIG values.
    """
    if mode == "manual":
        return GaborParams(SIGMA_GABOR, FREQ, KERNEL_SIZE, source="manual")
    return load_params_yaml(nrrd_path) or config_params()

# ================= Helpers: statistics printout =================
def print_stats(name, tensor):
    """Print detailed statistics of a tensor."""
    t = tensor.float()
    vmin = t.min().item()
    vmax = t.max().item()
    vmean = t.mean().item()
    vstd = t.std().item()
    
    # Quantiles on a subsample (torch.quantile fails on very large tensors)
    if t.numel() > 200000:
        t_sub = t.flatten()[::int(t.numel()/200000)]
    else:
        t_sub = t.flatten()
    
    p50 = torch.quantile(t_sub, 0.5).item()
    p90 = torch.quantile(t_sub, 0.9).item()
    p99 = torch.quantile(t_sub, 0.99).item()
    
    print(f"  [DEBUG] {name}:")
    print(f"    Range: [{vmin:.4f}, {vmax:.4f}]")
    print(f"    Mean: {vmean:.4f} | Std: {vstd:.4f}")
    print(f"    P50: {p50:.4f} | P90: {p90:.4f} | P99: {p99:.4f}")
    return p90, p99

# ================= Basic functions =================

def fibonacci_hemisphere(num_samples):
    points = []
    phi = (math.sqrt(5.0) + 1.0) / 2.0
    for i in range(num_samples):
        z = (i + 0.5) / num_samples 
        radius = math.sqrt(1.0 - z * z)
        theta = 2.0 * math.pi * i / phi
        x = radius * math.cos(theta)
        y = radius * math.sin(theta)
        vec = np.array([x, y, z])
        if np.linalg.norm(vec) > 0: vec /= np.linalg.norm(vec)
        points.append(vec)
    return np.array(points)

def create_gabor_kernel_spatial(sigma, frequency, direction, kernel_size):
    center = kernel_size // 2
    direction = np.array(direction)
    if np.linalg.norm(direction) > 0: direction = direction / np.linalg.norm(direction)
    x = np.arange(-center, center + 1)
    y = np.arange(-center, center + 1)
    z = np.arange(-center, center + 1)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    pos = np.stack([xx, yy, zz], axis=-1)
    r2 = np.sum(pos**2, axis=-1)
    gaussian = np.exp(-r2 / (2.0 * sigma * sigma))
    proj = np.sum(pos * direction, axis=-1)
    phase = 2.0 * math.pi * frequency * proj
    k_real = gaussian * np.cos(phase)
    k_imag = gaussian * np.sin(phase)
    w_sum = np.sum(gaussian)
    if w_sum > 0:
        k_real /= w_sum
        k_imag /= w_sum
    return k_real, k_imag

def add_custom_axes(viewer, volume_shape):
    max_dim = max(volume_shape)
    L = max_dim * 0.15
    viewer.add_shapes(np.array([[0,0,0], [L,0,0]]), shape_type='line', edge_color='red', name='X', edge_width=3)
    viewer.add_shapes(np.array([[0,0,0], [0,L,0]]), shape_type='line', edge_color='green', name='Y', edge_width=3)
    viewer.add_shapes(np.array([[0,0,0], [0,0,L]]), shape_type='line', edge_color='blue', name='Z', edge_width=3)

# ================= Core algorithm (every step keeps its output) =================

@torch.no_grad() 
def compute_structure_tensor_field(input_tensor, directions, sigma_gabor, freq, kernel_size, device='cuda'):
    print("\n>>> Step 1: Gabor Accumulation (Structure Tensor Construction)")
    vol_shape = input_tensor.shape
    print(f"  Input volume shape: {vol_shape}")
    D, H, W = vol_shape
    
    t_xx = torch.zeros(vol_shape, dtype=torch.float32, device=device)
    t_yy = torch.zeros(vol_shape, dtype=torch.float32, device=device)
    t_zz = torch.zeros(vol_shape, dtype=torch.float32, device=device)
    t_xy = torch.zeros(vol_shape, dtype=torch.float32, device=device)
    t_xz = torch.zeros(vol_shape, dtype=torch.float32, device=device)
    t_yz = torch.zeros(vol_shape, dtype=torch.float32, device=device)
    
    total_energy = torch.zeros(vol_shape, dtype=torch.float32, device=device)
    
    # Keep input_fft as complex64 (float32 per component, about 2.4 GB)
    input_fft = torch.fft.fftn(input_tensor.to(device=device, dtype=torch.complex64))
    
    split = (kernel_size + 1) // 2
    rem = kernel_size - split
    z_src = [slice(0, split), slice(split, kernel_size)]
    y_src = [slice(0, split), slice(split, kernel_size)]
    x_src = [slice(0, split), slice(split, kernel_size)]
    z_dst = [slice(0, split), slice(D - rem, D)]
    y_dst = [slice(0, split), slice(H - rem, H)]
    x_dst = [slice(0, split), slice(W - rem, W)]
    
    for i, direction in enumerate(directions):
        sys.stdout.write(f"\r  Dir {i+1}/{len(directions)}...")
        sys.stdout.flush()
        
        k_real, k_imag = create_gabor_kernel_spatial(sigma_gabor, freq, direction, kernel_size)
        
        k_complex_small = torch.complex(
            torch.from_numpy(k_real).float(), 
            torch.from_numpy(k_imag).float()
        ).to(device)
        
        k_shifted_small = torch.fft.ifftshift(k_complex_small)
        del k_complex_small
        
        k_tensor = torch.zeros(vol_shape, dtype=torch.complex64, device=device)
        
        for iz in range(2):
            if z_dst[iz].start == z_dst[iz].stop: continue
            for iy in range(2):
                if y_dst[iy].start == y_dst[iy].stop: continue
                for ix in range(2):
                    if x_dst[ix].start == x_dst[ix].stop: continue
                    k_tensor[z_dst[iz], y_dst[iy], x_dst[ix]] = k_shifted_small[z_src[iz], y_src[iy], x_src[ix]]
                    
        del k_shifted_small
        
        kernel_fft = torch.fft.fftn(k_tensor)
        del k_tensor 
        
        kernel_fft.mul_(input_fft)
        
        # Inverse transform gives response_complex (about 2.4 GB)
        response_complex = torch.fft.ifftn(kernel_fft)
        del kernel_fft
        torch.cuda.empty_cache()  # free the 2.4 GB of kernel_fft right away, this keeps peak memory down
        
        # ==============================================================
        # Compute |response|^2 by hand to avoid the extra allocation made by .abs()
        # ==============================================================
        # Preallocate one 1.2 GB float32 tensor, the only new tensor in this iteration
        w = torch.empty(vol_shape, dtype=torch.float32, device=device)
        
        # 1. Square of the real part into w (view copy + in-place square, no extra memory)
        w.copy_(response_complex.real)
        w.square_()
        
        # 2. Add the square of the imaginary part (fused addcmul_: w = w + imag * imag, no extra memory)
        w.addcmul_(response_complex.imag, response_complex.imag)
        
        # w now holds the squared magnitude, used directly for the structure tensor
        
        # 3. Accumulate the energy
        # w.sqrt() briefly allocates 1.2 GB, which is freed right after the add
        total_energy.add_(w.sqrt()) 
        
        # 4. response_complex is no longer needed, release its 2.4 GB now
        del response_complex
        torch.cuda.empty_cache()
        # ==============================================================
        
        dx, dy, dz = direction[2], direction[1], direction[0]
        
        # Accumulate the structure tensor in place
        t_xx.add_(w, alpha=dx * dx)
        t_yy.add_(w, alpha=dy * dy)
        t_zz.add_(w, alpha=dz * dz)
        t_xy.add_(w, alpha=dx * dy)
        t_xz.add_(w, alpha=dx * dz)
        t_yz.add_(w, alpha=dy * dz)
        
        del w
        torch.cuda.empty_cache()
        
    print("\n  Accumulation Done.")
    
    del input_fft
    torch.cuda.empty_cache()
    
    return [t_xx, t_yy, t_zz, t_xy, t_xz, t_yz], total_energy

@torch.no_grad()
def compute_structure_tensor_field_chunked(volume_np, directions, sigma_gabor, freq, kernel_size, device='cuda', chunk_h=None):
    """Chunked version: splits volume along H to fit in VRAM."""
    print("\n>>> Step 1: Gabor Accumulation [CHUNKED]")
    D, H, W = volume_np.shape
    overlap = kernel_size // 2

    # Auto chunk size: use ~60% of VRAM
    if chunk_h is None:
        vram = torch.cuda.get_device_properties(device).total_memory
        max_voxels = int(vram * 0.5) // 56   # ~56 bytes/voxel at peak
        chunk_h = max(kernel_size * 2, min(H, max_voxels // (D * W)))

    print(f"  Volume: {D}x{H}x{W}, chunk_h={chunk_h}, overlap={overlap}")

    # Output on CPU
    out = [torch.zeros((D, H, W), dtype=torch.float32) for _ in range(6)]
    total_energy = torch.zeros((D, H, W), dtype=torch.float32)

    # Pre-compute small Gabor kernels (CPU)
    gabor_kernels = []
    for d in directions:
        kr, ki = create_gabor_kernel_spatial(sigma_gabor, freq, d, kernel_size)
        gabor_kernels.append((kr, ki, d))

    starts = list(range(0, H, chunk_h))

    for ci, h_start in enumerate(starts):
        h_end = min(h_start + chunk_h, H)
        src_s = max(0, h_start - overlap)
        src_e = min(H, h_end + overlap)
        val_s = h_start - src_s
        val_e = val_s + (h_end - h_start)
        cH = src_e - src_s
        cshape = (D, cH, W)

        print(f"\n  Chunk {ci+1}/{len(starts)}: H[{h_start}:{h_end}] "
              f"(padded [{src_s}:{src_e}], {D}x{cH}x{W})")

        input_fft = torch.fft.fftn(
            torch.from_numpy(volume_np[:, src_s:src_e, :].copy())
            .to(device=device, dtype=torch.complex64)
        )

        c_acc = [torch.zeros(cshape, dtype=torch.float32, device=device) for _ in range(6)]
        c_energy = torch.zeros(cshape, dtype=torch.float32, device=device)

        # Kernel padding slices for this chunk shape
        split = (kernel_size + 1) // 2
        rem = kernel_size - split
        z_src = [slice(0, split), slice(split, kernel_size)]
        y_src = [slice(0, split), slice(split, kernel_size)]
        x_src = [slice(0, split), slice(split, kernel_size)]
        z_dst = [slice(0, split), slice(D - rem, D)]
        y_dst = [slice(0, split), slice(cH - rem, cH)]
        x_dst = [slice(0, split), slice(W - rem, W)]

        for i, (k_real, k_imag, direction) in enumerate(gabor_kernels):
            sys.stdout.write(f"\r    Dir {i+1}/{len(directions)}...")
            sys.stdout.flush()

            k_complex_small = torch.complex(
                torch.from_numpy(k_real).float(),
                torch.from_numpy(k_imag).float()
            ).to(device)
            k_shifted = torch.fft.ifftshift(k_complex_small)
            del k_complex_small

            k_tensor = torch.zeros(cshape, dtype=torch.complex64, device=device)
            for iz in range(2):
                if z_dst[iz].start == z_dst[iz].stop: continue
                for iy in range(2):
                    if y_dst[iy].start == y_dst[iy].stop: continue
                    for ix in range(2):
                        if x_dst[ix].start == x_dst[ix].stop: continue
                        k_tensor[z_dst[iz], y_dst[iy], x_dst[ix]] = \
                            k_shifted[z_src[iz], y_src[iy], x_src[ix]]
            del k_shifted

            kernel_fft = torch.fft.fftn(k_tensor)
            del k_tensor
            kernel_fft.mul_(input_fft)
            response = torch.fft.ifftn(kernel_fft)
            del kernel_fft
            torch.cuda.empty_cache()

            w = torch.empty(cshape, dtype=torch.float32, device=device)
            w.copy_(response.real)
            w.square_()
            w.addcmul_(response.imag, response.imag)
            c_energy.add_(w.sqrt())
            del response
            torch.cuda.empty_cache()

            dx, dy, dz = direction[2], direction[1], direction[0]
            c_acc[0].add_(w, alpha=dx*dx)
            c_acc[1].add_(w, alpha=dy*dy)
            c_acc[2].add_(w, alpha=dz*dz)
            c_acc[3].add_(w, alpha=dx*dy)
            c_acc[4].add_(w, alpha=dx*dz)
            c_acc[5].add_(w, alpha=dy*dz)
            del w
            torch.cuda.empty_cache()

        print()
        del input_fft

        v = slice(val_s, val_e)
        for j in range(6):
            out[j][:, h_start:h_end, :] = c_acc[j][:, v, :].cpu()
        total_energy[:, h_start:h_end, :] = c_energy[:, v, :].cpu()

        del c_acc, c_energy
        torch.cuda.empty_cache()

    print("  Chunked Accumulation Done.")
    return out, total_energy

@torch.no_grad()
def smooth_and_solve_tangent(t_components, sigma_neighbor, device='cuda'):
    print(f"\n>>> Step 2: Neighborhood Smoothing (Sigma={sigma_neighbor}) & Eigen Solve")
    
    # Read the shape first. Do not unpack t_components all at once
    D, H, W = t_components[0].shape
    N = D * H * W
    
    k_size = int(4 * sigma_neighbor + 1) | 1
    x_c = torch.arange(k_size, device=device).float() - (k_size - 1) / 2
    k_1d = torch.exp(-x_c**2 / (2*sigma_neighbor**2))
    k_1d = k_1d / k_1d.sum()
    
    def smooth_vol(v):
        v = v.unsqueeze(0).unsqueeze(0)
        v = F.conv3d(v, k_1d.view(1,1,k_size,1,1), padding=(k_size//2,0,0))
        v = F.conv3d(v, k_1d.view(1,1,1,k_size,1), padding=(0,k_size//2,0))
        v = F.conv3d(v, k_1d.view(1,1,1,1,k_size), padding=(0,0,k_size//2))
        return v.squeeze().reshape(-1)
    
    # Consume t_components destructively with pop(0).
    # Each pop removes the tensor from the caller's list, so after smoothing
    # its last reference is gone and PyTorch frees the 1.2 GB immediately.
    
    s_xx = smooth_vol(t_components.pop(0))
    torch.cuda.empty_cache()
    
    s_yy = smooth_vol(t_components.pop(0))
    torch.cuda.empty_cache()
    
    s_zz = smooth_vol(t_components.pop(0))
    torch.cuda.empty_cache()
    
    s_xy = smooth_vol(t_components.pop(0))
    torch.cuda.empty_cache()
    
    s_xz = smooth_vol(t_components.pop(0))
    torch.cuda.empty_cache()
    
    s_yz = smooth_vol(t_components.pop(0))
    torch.cuda.empty_cache()
    
    # All six Gabor tensors have now been freed from GPU memory.
    
    # Now it is safe to allocate the output arrays
    tangents = torch.zeros((N, 3), dtype=torch.float32, device=device)
    linearity_map = torch.zeros((N,), dtype=torch.float32, device=device)
    
    print("  Solving Eigen System in chunks...")
    
    # Chunked eigen decomposition
    chunk_size = 200_000 
    eye3 = torch.eye(3, device=device, dtype=torch.float32).unsqueeze(0) * 1e-8

    for i in range(0, N, chunk_size):
        end = min(i + chunk_size, N)
        
        # Build the 3x3 matrices per chunk (under 10 MB of GPU memory per iteration)
        batch = torch.stack([
            s_zz[i:end], s_yz[i:end], s_xz[i:end],
            s_yz[i:end], s_yy[i:end], s_xy[i:end],
            s_xz[i:end], s_xy[i:end], s_xx[i:end]
        ], dim=1).view(-1, 3, 3)
        
        # Remove NaNs and add a small diagonal term for stability
        batch = torch.nan_to_num(batch, nan=0.0, posinf=0.0, neginf=0.0)
        batch = batch + eye3
        
        try:
            L, V = torch.linalg.eigh(batch)
        except Exception:
            # GPU cusolver failed — fall back to CPU
            batch_cpu = batch.cpu()
            L_cpu, V_cpu = torch.linalg.eigh(batch_cpu)
            L, V = L_cpu.to(device), V_cpu.to(device)
        
        # L0 <= L1 <= L2
        tangents[i:end] = V[:, :, 0] # Tangent (Min Eigenvector)
        
        l0 = L[:, 0]
        l1 = L[:, 1]
        
        metric = (l1 - l0) / (l1 + l0 + 1e-6)
        linearity_map[i:end] = metric
        
    # Restore the 3D shape
    linearity_map = linearity_map.view(D, H, W)
    tangents = tangents.view(D, H, W, 3)
    
    # print_stats("Linearity (Confidence)", linearity_map)
    
    return tangents, linearity_map

@torch.no_grad()
def smooth_and_solve_tangent_chunked(t_components, sigma_neighbor, device='cuda', chunk_h=2000):
    """Chunked smoothing and eigen solve for large volumes."""
    D, H, W = t_components[0].shape
    print(f"\n>>> Step 2: Smoothing (σ={sigma_neighbor}) & Eigen Solve [CHUNKED]")

    k_size = int(4 * sigma_neighbor + 1) | 1
    overlap = k_size // 2
    x_c = torch.arange(k_size, device=device).float() - (k_size - 1) / 2
    k_1d = torch.exp(-x_c**2 / (2 * sigma_neighbor**2))
    k_1d = k_1d / k_1d.sum()

    tangents = torch.zeros((D, H, W, 3), dtype=torch.float32)
    linearity_map = torch.zeros((D, H, W), dtype=torch.float32)
    eye3 = torch.eye(3, device=device, dtype=torch.float32).unsqueeze(0) * 1e-8

    starts = list(range(0, H, chunk_h))
    print(f"  Chunks: {len(starts)}, smooth kernel: {k_size}, overlap: {overlap}")

    for ci, h_start in enumerate(starts):
        h_end = min(h_start + chunk_h, H)
        src_s = max(0, h_start - overlap)
        src_e = min(H, h_end + overlap)
        val_s = h_start - src_s
        val_e = val_s + (h_end - h_start)
        out_h = h_end - h_start

        print(f"  Chunk {ci+1}/{len(starts)}: H[{h_start}:{h_end}]")

        def smooth_vol(v_cpu):
            v = v_cpu.to(device).unsqueeze(0).unsqueeze(0)
            v = F.conv3d(v, k_1d.view(1,1,k_size,1,1), padding=(k_size//2,0,0))
            v = F.conv3d(v, k_1d.view(1,1,1,k_size,1), padding=(0,k_size//2,0))
            v = F.conv3d(v, k_1d.view(1,1,1,1,k_size), padding=(0,0,k_size//2))
            return v.squeeze()

        smoothed = []
        for j in range(6):
            s = smooth_vol(t_components[j][:, src_s:src_e, :])
            smoothed.append(s[:, val_s:val_e, :].reshape(-1))
            del s
            torch.cuda.empty_cache()

        s_xx, s_yy, s_zz, s_xy, s_xz, s_yz = smoothed

        N_chunk = D * out_h * W
        c_tan = torch.zeros((N_chunk, 3), dtype=torch.float32, device=device)
        c_lin = torch.zeros((N_chunk,), dtype=torch.float32, device=device)

        eigen_bs = 200_000
        for i in range(0, N_chunk, eigen_bs):
            end_i = min(i + eigen_bs, N_chunk)
            batch = torch.stack([
                s_zz[i:end_i], s_yz[i:end_i], s_xz[i:end_i],
                s_yz[i:end_i], s_yy[i:end_i], s_xy[i:end_i],
                s_xz[i:end_i], s_xy[i:end_i], s_xx[i:end_i]
            ], dim=1).view(-1, 3, 3)
            batch = torch.nan_to_num(batch, nan=0.0, posinf=0.0, neginf=0.0)
            batch = batch + eye3
            try:
                L, V = torch.linalg.eigh(batch)
            except Exception:
                L, V = torch.linalg.eigh(batch.cpu())
                L, V = L.to(device), V.to(device)
            c_tan[i:end_i] = V[:, :, 0]
            l0, l1 = L[:, 0], L[:, 1]
            c_lin[i:end_i] = (l1 - l0) / (l1 + l0 + 1e-6)

        tangents[:, h_start:h_end, :, :] = c_tan.view(D, out_h, W, 3).cpu()
        linearity_map[:, h_start:h_end, :] = c_lin.view(D, out_h, W).cpu()

        del smoothed, s_xx, s_yy, s_zz, s_xy, s_xz, s_yz, c_tan, c_lin
        torch.cuda.empty_cache()

    return tangents, linearity_map

# ================= Pipeline (single-volume) =================

@dataclass
class PipelineResult:
    energy: np.ndarray
    linearity: np.ndarray
    tangents: np.ndarray
    mask_final: np.ndarray
    mask_sub: np.ndarray
    pts_raw: np.ndarray
    vecs_raw: np.ndarray
    raw_vector_data: np.ndarray
    raw_colors: np.ndarray
    volume: np.ndarray
    vol_shape: tuple


def run_pipeline(volume: np.ndarray, device: torch.device, console: Console,
                  params: GaborParams) -> PipelineResult:
    """Gabor accumulation → tensor smoothing/eigensolve → masking → raw vectors."""
    vol_shape = volume.shape
    directions = fibonacci_hemisphere(NUM_DIRECTIONS)
    sigma_neighbor = max(1, int(round(params.sigma * SIGMA_NEIGHBOR_RATIO)))
    console.print(
        f"  [dim]gabor params: σ={params.sigma}, σ_neighbor={sigma_neighbor} "
        f"(ratio={SIGMA_NEIGHBOR_RATIO}), freq={params.freq}, "
        f"kernel={params.kernel_size}  ← {params.source}[/dim]"
    )

    t_components, total_energy = compute_structure_tensor_field_chunked(
        volume, directions, params.sigma, params.freq, params.kernel_size,
        device, chunk_h=GABOR_CHUNK_H,
    )
    tangents, linearity = smooth_and_solve_tangent_chunked(
        t_components, sigma_neighbor, device, chunk_h=EIGEN_CHUNK_H
    )

    energy_np = total_energy.cpu().numpy()
    lin_np = linearity.cpu().numpy()
    tangents_np = tangents.cpu().numpy()

    mask_intensity = energy_np > INTENSITY_THRESH
    mask_linearity = lin_np > LINEARITY_THRESH
    mask_final = mask_intensity & mask_linearity
    console.print(
        f"  [dim]intensity({INTENSITY_THRESH:.0f}): {int(np.sum(mask_intensity)):,} | "
        f"linearity({LINEARITY_THRESH}): {int(np.sum(mask_linearity)):,} | "
        f"active: {int(np.sum(mask_final)):,}[/dim]"
    )

    sub = DOWNSAMPLING_FACTOR
    z, y, x = np.mgrid[0:vol_shape[0]:sub, 0:vol_shape[1]:sub, 0:vol_shape[2]:sub]
    mask_sub = mask_final[::sub, ::sub, ::sub]
    pts_raw = np.stack([z[mask_sub], y[mask_sub], x[mask_sub]], axis=-1)
    vecs_raw = tangents_np[::sub, ::sub, ::sub][mask_sub]
    raw_vector_data = np.zeros((len(pts_raw), 2, 3), dtype=np.float32)
    raw_vector_data[:, 0, :] = pts_raw
    raw_vector_data[:, 1, :] = vecs_raw * 15.0
    raw_colors = np.abs(vecs_raw)

    return PipelineResult(
        energy=energy_np, linearity=lin_np, tangents=tangents_np,
        mask_final=mask_final, mask_sub=mask_sub,
        pts_raw=pts_raw, vecs_raw=vecs_raw,
        raw_vector_data=raw_vector_data, raw_colors=raw_colors,
        volume=volume, vol_shape=vol_shape,
    )


def output_paths(output_dir: Path, base_name: str) -> Path:
    e_tag = f"{INTENSITY_THRESH:.0f}"
    l_tag = f"{LINEARITY_THRESH}"
    return output_dir / f"{base_name}_pointcloud_full_energy_{e_tag}_linearity_{l_tag}.npz"


def save_pointclouds(res: PipelineResult, output_dir: Path, base_name: str) -> tuple[Path, int]:
    """Write the single full-resolution pointcloud npz. Returns (path, n_pts)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_full = output_paths(output_dir, base_name)

    z_full, y_full, x_full = np.where(res.mask_final)
    pts_full = np.stack([z_full, y_full, x_full], axis=-1).astype(np.int32)
    vecs_full = res.tangents[z_full, y_full, x_full]
    np.savez(
        out_full,
        points=pts_full,
        directions=vecs_full,
        energy=res.energy[z_full, y_full, x_full],
        linearity=res.linearity[z_full, y_full, x_full],
    )
    return out_full, len(pts_full)


def make_inspector_figure(res: PipelineResult):
    """Two-panel histogram: Gabor Energy + Linearity with current thresholds marked."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('Threshold Inspector', fontsize=13)

    ax = axes[0]
    sample = res.energy.ravel()
    if sample.size > 500_000:
        sample = sample[::sample.size // 500_000]
    nonzero = sample[sample > 0]
    ax.hist(nonzero if nonzero.size else sample, bins=200, color='steelblue', alpha=0.8, log=True)
    ax.axvline(INTENSITY_THRESH, color='red', linewidth=2,
               label=f'INTENSITY_THRESH = {INTENSITY_THRESH:.0f}')
    pct_kept = 100.0 * np.mean(res.energy > INTENSITY_THRESH)
    ax.set_title(f'Gabor Energy  ({pct_kept:.1f}% kept)')
    ax.set_xlabel('Energy'); ax.set_ylabel('Count (log)'); ax.legend()

    ax = axes[1]
    lin_sample = res.linearity.ravel()
    if lin_sample.size > 500_000:
        lin_sample = lin_sample[::lin_sample.size // 500_000]
    ax.hist(lin_sample, bins=200, color='mediumseagreen', alpha=0.8, log=True)
    ax.axvline(LINEARITY_THRESH, color='red', linewidth=2,
               label=f'LINEARITY_THRESH = {LINEARITY_THRESH}')
    pct_lin = 100.0 * np.mean(res.linearity > LINEARITY_THRESH)
    ax.set_title(f'Linearity  ({pct_lin:.1f}% kept)')
    ax.set_xlabel('Linearity'); ax.set_ylabel('Count (log)'); ax.legend()
    fig.tight_layout()
    return fig


def make_slices_figure(res: PipelineResult):
    """4×3 grid: CT / Energy / Linearity / Mask rows × XY/XZ/YZ mid-slice cols."""
    import matplotlib.pyplot as plt

    D, H, W = res.vol_shape
    z_mid, y_mid, x_mid = D // 2, H // 2, W // 2

    rows = [
        ('CT',         res.volume,                          'gray'),
        ('Energy',     res.energy,                          'magma'),
        ('Linearity',  res.linearity,                       'viridis'),
        ('Mask',       res.mask_final.astype(np.uint8),     'gray'),
    ]

    fig, axes = plt.subplots(len(rows), 3, figsize=(15, 4 * len(rows)))
    fig.suptitle(
        f"Pipeline slices  (active voxels: {int(np.sum(res.mask_final)):,})",
        fontsize=13,
    )

    for r, (name, arr, cmap) in enumerate(rows):
        if name == 'Mask':
            vmin, vmax = 0, 1
        else:
            pos = arr[arr > 0]
            vmin, vmax = (np.percentile(pos, [5, 99]) if pos.size else (float(arr.min()), float(arr.max())))
        axes[r, 0].imshow(arr[z_mid, :, :], cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
        axes[r, 0].set_title(f'{name} XY (z={z_mid})')
        axes[r, 1].imshow(arr[:, y_mid, :].T, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower', aspect='auto')
        axes[r, 1].set_title(f'{name} XZ (y={y_mid})')
        axes[r, 2].imshow(arr[:, :, x_mid].T, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower', aspect='auto')
        axes[r, 2].set_title(f'{name} YZ (x={x_mid})')
        for c in range(3):
            axes[r, c].axis('off')

    fig.tight_layout()
    return fig


def save_log_artifacts(
    figs_dir: Path, log_dir: Path, base_name: str,
    res: PipelineResult, stats: dict,
) -> None:
    """Write per-volume PNG diagnostics to <figs_dir>/, text stats to <log_dir>/."""
    import matplotlib.pyplot as plt

    figs_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    fig_insp = make_inspector_figure(res)
    fig_insp.savefig(figs_dir / f"{base_name}_inspector.png", dpi=120, bbox_inches='tight')
    plt.close(fig_insp)

    fig_slices = make_slices_figure(res)
    fig_slices.savefig(figs_dir / f"{base_name}_slices.png", dpi=120, bbox_inches='tight')
    plt.close(fig_slices)

    params: GaborParams = stats.get("params", config_params())
    sigma_neighbor = max(1, int(round(params.sigma * SIGMA_NEIGHBOR_RATIO)))
    lines = [
        f"file              : {stats.get('file', base_name)}",
        f"volume shape      : {res.vol_shape[0]} x {res.vol_shape[1]} x {res.vol_shape[2]}",
        f"INTENSITY_THRESH  : {INTENSITY_THRESH}",
        f"LINEARITY_THRESH  : {LINEARITY_THRESH}",
        f"SIGMA_GABOR       : {params.sigma}    (source: {params.source})",
        f"FREQ              : {params.freq}",
        f"KERNEL_SIZE       : {params.kernel_size}",
        f"NUM_DIRECTIONS    : {NUM_DIRECTIONS}",
        f"SIGMA_NEIGHBOR    : {sigma_neighbor}    (ratio={SIGMA_NEIGHBOR_RATIO} × σ)",
        f"intensity kept %  : {100.0 * np.mean(res.energy > INTENSITY_THRESH):.3f}",
        f"linearity kept %  : {100.0 * np.mean(res.linearity > LINEARITY_THRESH):.3f}",
        f"active voxels     : {int(np.sum(res.mask_final))}",
        f"full pointcloud   : {stats.get('n_full', '?')}",
        f"elapsed seconds   : {stats.get('seconds', 0.0):.1f}",
    ]
    (log_dir / f"{base_name}_stats.txt").write_text("\n".join(lines) + "\n")


def launch_napari_debug(res: PipelineResult) -> None:
    import matplotlib.pyplot as plt

    print("\n>>> Launching Napari Debugger...")
    viewer = napari.Viewer(ndisplay=3)
    add_custom_axes(viewer, res.vol_shape)

    viewer.add_image(res.volume, name='1. CT Original', opacity=0.3, blending='translucent')
    viewer.add_image(res.energy, name='2. Gabor Energy (Response)', colormap='magma', visible=False)
    viewer.add_image(res.linearity, name='3. Linearity (Confidence)', colormap='viridis', visible=False)
    viewer.add_labels(res.mask_final.astype(int), name='4. Final Active Mask', opacity=0.5, visible=False)

    if len(res.raw_vector_data) > 0:
        viewer.add_vectors(
            res.raw_vector_data,
            name='5. Raw Vectors (Subsampled)',
            edge_color=res.raw_colors,
            edge_width=0.8,
            opacity=1.0,
        )
    else:
        print("!!! CRITICAL: No vectors passed the threshold. Check stats above.")

    fig = make_inspector_figure(res)
    plt.show(block=False)

    print("\n=== Debug Guide ===")
    print("1. Final Active Mask: if the threshold is too high, yarn regions get cut away.")
    print("2. Gabor Energy: dark regions mean SIGMA_GABOR/FREQ do not match the data scale.")
    print("3. Linearity: naturally low where yarns cross, and too high a LINEARITY_THRESH filters them out.")

    napari.run()
    plt.close(fig)


def view_pointcloud_npz(npz_path: Path, ct_dir: Optional[Path] = None) -> None:
    """Load an existing _pointcloud_*.npz and visualize in napari (no Gabor).

    CT layer is added but hidden by default; no matplotlib window is opened.
    """
    data = np.load(npz_path)
    points = np.asarray(data["points"])
    directions = np.asarray(data["directions"], dtype=np.float32)
    energy = np.asarray(data["energy"]) if "energy" in data.files else None
    linearity = np.asarray(data["linearity"]) if "linearity" in data.files else None

    name = npz_path.stem
    stem = name.split("_pointcloud_")[0] if "_pointcloud_" in name else name

    volume = None
    vol_shape = None
    if ct_dir is not None:
        ct_path = ct_dir / f"{stem}.nrrd"
        if ct_path.exists():
            print(f"Loading CT: {ct_path}")
            volume, _ = nrrd.read(str(ct_path))
            volume = volume.astype(np.float32)
            vol_shape = volume.shape

    if vol_shape is None and points.size:
        vol_shape = tuple(int(p.max()) + 1 for p in points.T)

    print(f"\n=== Viewing {npz_path.name} ===")
    print(f"  points       : {len(points):,}")
    print(f"  CT layer     : {'yes' if volume is not None else 'no (none found)'}")
    if energy is not None and energy.size:
        print(f"  energy range : [{float(energy.min()):.1f}, {float(energy.max()):.1f}]")
    if linearity is not None and linearity.size:
        print(f"  linearity    : [{float(linearity.min()):.3f}, {float(linearity.max()):.3f}]")

    viewer = napari.Viewer(ndisplay=3)
    if vol_shape is not None:
        add_custom_axes(viewer, vol_shape)
    if volume is not None:
        viewer.add_image(volume, name="1. CT Original", opacity=0.3,
                         blending="translucent", visible=False)

    vec_data = np.zeros((len(points), 2, 3), dtype=np.float32)
    vec_data[:, 0, :] = points
    vec_data[:, 1, :] = directions * 15.0
    colors = np.abs(directions)
    viewer.add_vectors(
        vec_data, name="2. Tangent Vectors",
        edge_color=colors, edge_width=0.8, opacity=1.0,
    )

    if energy is not None and energy.size:
        viewer.add_points(
            points.astype(np.float32), name="3. Points (energy)",
            size=1.5, face_color=np.log10(np.maximum(energy, 1.0)),
            face_colormap="magma", opacity=0.6, visible=False,
        )
    if linearity is not None and linearity.size:
        viewer.add_points(
            points.astype(np.float32), name="4. Points (linearity)",
            size=1.5, face_color=linearity.astype(np.float32),
            face_colormap="viridis", opacity=0.6, visible=False,
        )

    @viewer.bind_key("Escape", overwrite=True)
    def _advance(v):
        v.close()

    napari.run()


# ================= Batch driver =================

@dataclass
class FileResult:
    name: str
    status: str  # "ok" | "skip" | "fail"
    n_active: int = 0
    n_full: int = 0
    out_bytes: int = 0
    seconds: float = 0.0
    note: str = ""


def gather_inputs(input_dir: Path) -> list[Path]:
    if not input_dir.is_dir():
        return []
    files = sorted({f.resolve() for f in input_dir.rglob("*.nrrd") if f.is_file()})
    if not PRIORITY_KEYS:
        return files
    # Bucket by priority: first matching key wins.  Within each bucket
    # the original alphabetical order is preserved.  Files matching no
    # key fall to the tail in alphabetical order.
    keys_lc = [k.lower() for k in PRIORITY_KEYS]
    buckets: list[list[Path]] = [[] for _ in keys_lc]
    rest: list[Path] = []
    for f in files:
        name_lc = f.name.lower()
        for i, k in enumerate(keys_lc):
            if k in name_lc:
                buckets[i].append(f)
                break
        else:
            rest.append(f)
    out: list[Path] = []
    for b in buckets:
        out.extend(b)
    out.extend(rest)
    return out


def scan_pointclouds(output_dir: Path, prefer_full: bool = False) -> list[Path]:
    """One pointcloud npz per sample stem. Default: prefer lighter _sub_*.npz."""
    if not output_dir.is_dir():
        return []
    sub = sorted(output_dir.glob("*_pointcloud_*_sub_*.npz"))
    full = sorted(output_dir.glob("*_pointcloud_full_*.npz"))
    by_stem: dict[str, Path] = {}
    primary, fallback = (full, sub) if prefer_full else (sub, full)
    for f in primary:
        stem = f.name.split("_pointcloud_")[0]
        by_stem.setdefault(stem, f)
    for f in fallback:
        stem = f.name.split("_pointcloud_")[0]
        by_stem.setdefault(stem, f)
    return [by_stem[s] for s in sorted(by_stem)]


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
                    help=f"Input directory (default: {DEFAULT_INPUT_DIR})")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                    help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})")
    ap.add_argument("--file", type=Path, default=None,
                    help="Process a single .nrrd and launch napari for debugging.")
    ap.add_argument("--no-figs", action="store_true",
                    help="Skip writing PNG diagnostics (figs/) + per-volume stats text (log/)")
    ap.add_argument("--mode", choices=("manual", "auto"), default="manual",
                    help=f"manual (default): always SIGMA_GABOR={SIGMA_GABOR}, FREQ={FREQ}, the paper setting. "
                         f"auto: read <input>/{PARAMS_SUBDIR}/<stem>.yaml from v1, fall back to CONFIG.")
    ap.add_argument("--view", type=Path, default=None,
                    help="View an existing _pointcloud_*.npz in napari (skips Gabor entirely).")
    ap.add_argument("--view-all", action="store_true",
                    help="Iterate every _pointcloud_*.npz in --output-dir; close one to advance.")
    ap.add_argument("--prefer-full", action="store_true",
                    help="With --view-all: use _full.npz instead of the lighter _sub_*.npz.")
    ap.add_argument("--sample", nargs="+", default=None, metavar="NAME",
                    help="Only process these samples, matched exactly by name (e.g. bar).")
    args = ap.parse_args()

    single_mode = args.file is not None
    view_mode = args.view is not None
    view_all_mode = args.view_all
    matplotlib.use("Qt5Agg" if (single_mode or view_mode or view_all_mode) else "Agg")

    console = Console()

    if view_mode or view_all_mode:
        if not HAS_NAPARI:
            console.print("[red]napari not installed[/red] — install with: pip install napari[all]")
            return 1
        if view_all_mode:
            files = scan_pointclouds(args.output_dir, prefer_full=args.prefer_full)
            if not files:
                console.print(f"[yellow]No _pointcloud_*.npz under[/yellow] {args.output_dir}")
                return 0
            console.rule(f"[bold cyan]View-all:[/] {len(files)} pointclouds in {args.output_dir}")
            for i, p in enumerate(files):
                console.print(f"[bold cyan]\\[{i+1}/{len(files)}][/] {p.name}  [dim](close window to advance)[/dim]")
                view_pointcloud_npz(p, ct_dir=args.input_dir)
            return 0
        if not args.view.exists():
            console.print(f"[red]Not found:[/red] {args.view}")
            return 1
        console.rule(f"[bold cyan]View pointcloud:[/] {args.view.name}")
        view_pointcloud_npz(args.view, ct_dir=args.input_dir)
        return 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"[dim]Device: {device}[/dim]")

    figs_dir = args.output_dir / "figs"
    log_dir = args.output_dir / "log"

    if single_mode:
        path: Path = args.file
        if not path.is_file():
            console.print(f"[red]File not found:[/red] {path}")
            return 1
        if not HAS_NAPARI:
            console.print("[red]napari not installed[/red] — install with: pip install napari[all]")
            return 1
        console.rule(f"[bold cyan]Debug:[/] {path.name}")
        params = resolve_params(path, args.mode)
        t0 = time.monotonic()
        volume, _ = nrrd.read(str(path))
        volume = volume.astype(np.float32)
        res = run_pipeline(volume, device, console, params)
        of, n_full = save_pointclouds(res, args.output_dir, path.stem)
        dt = time.monotonic() - t0
        if not args.no_figs:
            save_log_artifacts(figs_dir, log_dir, path.stem, res,
                               stats={"file": path.name, "n_full": n_full, "seconds": dt,
                                      "params": params})
            console.print(f"[dim]Figs → {figs_dir}  •  Log → {log_dir}[/dim]")
        console.print(f"  [green]ok[/] — {n_full:,} pts in {dt:.0f}s, output {of.name}")
        launch_napari_debug(res)
        return 0

    # Batch mode
    files = gather_inputs(args.input_dir)
    if args.sample:
        wanted = set(args.sample)
        files = [f for f in files if f.stem in wanted]
        missing = wanted - {f.stem for f in files}
        if missing:
            console.print(f"[red]No input for sample(s):[/red] {', '.join(sorted(missing))} under {args.input_dir}")
            return 1
    if not files:
        console.print(f"[yellow]No .nrrd files under[/yellow] {args.input_dir}")
        return 0

    console.rule(
        f"[bold cyan]Gabor batch[/]  ({len(files)} volumes)  "
        f"[dim]{args.input_dir} → {args.output_dir}[/]"
    )

    results: list[FileResult] = []
    for i, f in enumerate(files):
        console.rule(f"[bold cyan]\\[{i+1}/{len(files)}] {f.name}[/]")
        out_full = output_paths(args.output_dir, f.stem)
        if out_full.exists():
            console.print(f"  [yellow]skip[/] — {out_full.name} exists")
            results.append(FileResult(f.name, "skip", out_bytes=out_full.stat().st_size, note="exists"))
            continue

        params = resolve_params(f, args.mode)
        t0 = time.monotonic()
        try:
            volume, _ = nrrd.read(str(f))
            volume = volume.astype(np.float32)
            res = run_pipeline(volume, device, console, params)
            of, n_full = save_pointclouds(res, args.output_dir, f.stem)
            dt = time.monotonic() - t0
            if not args.no_figs:
                save_log_artifacts(figs_dir, log_dir, f.stem, res,
                                   stats={"file": f.name, "n_full": n_full, "seconds": dt,
                                          "params": params})
            results.append(FileResult(
                f.name, "ok",
                n_active=int(np.sum(res.mask_final)),
                n_full=n_full,
                out_bytes=of.stat().st_size,
                seconds=dt,
            ))
            console.print(f"  [green]ok[/] — {n_full:,} pts in {dt:.0f}s")
            del volume, res
            if device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception as e:
            dt = time.monotonic() - t0
            console.print(f"  [red]FAIL:[/red] {e}")
            results.append(FileResult(f.name, "fail", seconds=dt, note=str(e)[:80]))

    # Summary table
    table = Table(title="Gabor batch summary", header_style="bold magenta")
    table.add_column("File", style="cyan", no_wrap=True)
    table.add_column("Status", justify="center")
    table.add_column("Active vox", justify="right")
    table.add_column("Full pts", justify="right")
    table.add_column("Size", justify="right")
    table.add_column("Time", justify="right")
    table.add_column("Note", style="dim")
    sty = {"ok": "[green]ok[/]", "skip": "[yellow]skip[/]", "fail": "[red]fail[/]"}
    for r in results:
        table.add_row(
            r.name, sty[r.status],
            f"{r.n_active:,}" if r.status == "ok" else "-",
            f"{r.n_full:,}" if r.status == "ok" else "-",
            human_bytes(r.out_bytes) if r.out_bytes else "-",
            f"{r.seconds:.0f}s" if r.seconds else "-",
            r.note,
        )
    console.print(table)
    ok = sum(r.status == "ok" for r in results)
    sk = sum(r.status == "skip" for r in results)
    fl = sum(r.status == "fail" for r in results)
    console.print(
        f"[bold]Totals:[/bold] [green]{ok} ok[/], [yellow]{sk} skipped[/], [red]{fl} failed[/]"
    )

    if not args.no_figs:
        log_dir.mkdir(parents=True, exist_ok=True)
        lines = [
            "Gabor batch summary",
            "=" * 60,
            f"input dir         : {args.input_dir}",
            f"output dir        : {args.output_dir}",
            f"INTENSITY_THRESH  : {INTENSITY_THRESH}",
            f"LINEARITY_THRESH  : {LINEARITY_THRESH}",
            f"NUM_DIRECTIONS    : {NUM_DIRECTIONS}",
            f"SIGMA_NEIGHBOR    : {SIGMA_NEIGHBOR_RATIO} × σ  (per-volume; recomputed from each file's σ)",
            f"params mode       : {args.mode}  "
            f"(manual: σ={SIGMA_GABOR}, freq={FREQ}, kernel={KERNEL_SIZE}. "
            f"auto: <input>/{PARAMS_SUBDIR}/<stem>.yaml, falling back to the same values)",
            "",
            f"{'file':<32} {'status':>6} {'active':>14} {'full pts':>14} {'size':>10} {'time':>8}  note",
            "-" * 110,
        ]
        for r in results:
            lines.append(
                f"{r.name:<32} {r.status:>6} "
                f"{r.n_active:>14,} {r.n_full:>14,} "
                f"{human_bytes(r.out_bytes):>10} {f'{r.seconds:.0f}s':>8}  {r.note}"
            )
        lines.append("-" * 110)
        lines.append(f"Totals: {ok} ok, {sk} skipped, {fl} failed")
        (log_dir / "batch_summary.txt").write_text("\n".join(lines) + "\n")
        console.print(f"[dim]Batch summary → {log_dir}/batch_summary.txt[/dim]")

    return 0 if fl == 0 else 2


if __name__ == "__main__":
    sys.exit(main())