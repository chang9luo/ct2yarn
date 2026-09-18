#!/usr/bin/env bash
# Run every preprocessing step in order:
#   v0  data/raw       -> data/processed   Otsu threshold (CPU)
#   v2  data/processed -> data/gabor       Gabor orientation field, paper parameters (GPU)
#   v3  data/gabor     -> data/denoised    tangent-coherence outlier removal (CPU)
#   v4  data/denoised  -> data/binned      voxel binning, DS=4 (CPU)
# v1 (yarn radius measurement) is not needed, because v2 runs in manual mode.
#
# Usage, inside the ct2yarn conda environment:
#   bash run_preprocess.sh                          # all samples in ./data
#   bash run_preprocess.sh --sample bar             # a single sample
#   bash run_preprocess.sh --sample bar --sample O  # several samples
#   bash run_preprocess.sh /path/data --sample bar  # any data directory that contains raw/
#
# A sample name is the raw file name without .nrrd, matched exactly.
# Every step skips outputs that already exist, so an interrupted run can be resumed.
set -euo pipefail

usage() { sed -n '2,16p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ARG="$ROOT/data"
SAMPLES=()
while [ $# -gt 0 ]; do
    case "$1" in
        -s|--sample)
            if [ $# -lt 2 ]; then echo "--sample needs a name" >&2; exit 1; fi
            SAMPLES+=("$2"); shift 2 ;;
        -h|--help) usage; exit 0 ;;
        -*) echo "Unknown option: $1 (see --help)" >&2; exit 1 ;;
        *) DATA_ARG="$1"; shift ;;
    esac
done

if ! DATA="$(cd "$DATA_ARG" 2>/dev/null && pwd)" || [ ! -d "$DATA/raw" ]; then
    echo "No raw/ folder under $DATA_ARG" >&2
    exit 1
fi
PY="${PYTHON:-python}"

SAMPLE_ARGS=()
if [ ${#SAMPLES[@]} -gt 0 ]; then
    SAMPLE_ARGS=(--sample "${SAMPLES[@]}")
fi

if ! "$PY" -c "import matplotlib, nrrd, numpy, rich, scipy, torch, yaml" >/dev/null 2>&1; then
    echo "Required packages are missing. Run 'bash setup_env.sh' and 'conda activate ct2yarn' first." >&2
    exit 1
fi
if ! "$PY" -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)"; then
    echo "WARNING: CUDA is not available, so v2 will run on the CPU and be very slow." >&2
fi

run_step() {
    local name="$1"; shift
    echo
    echo "==================== $name ===================="
    local t0=$SECONDS
    "$PY" "$@" ${SAMPLE_ARGS[@]+"${SAMPLE_ARGS[@]}"}
    echo "---- $name finished in $((SECONDS - t0)) s"
}

echo "Data:    $DATA"
echo "Samples: ${SAMPLES[*]:-all}"
START=$SECONDS
run_step "v0 threshold" "$ROOT/preprocess/v0_threshold_denoise.py" --input-dir "$DATA/raw"       --output-dir "$DATA/processed"
run_step "v2 gabor"     "$ROOT/preprocess/v2_gabor_pointcloud.py"  --input-dir "$DATA/processed" --output-dir "$DATA/gabor" --mode manual
run_step "v3 denoise"   "$ROOT/preprocess/v3_united_denoise.py"    --input-dir "$DATA/gabor"     --output-dir "$DATA/denoised"
run_step "v4 binning"   "$ROOT/preprocess/v4_voxel_binning.py"     --input-dir "$DATA/denoised"  --output-dir "$DATA/binned"

echo
echo "Preprocessing finished in $((SECONDS - START)) s. Binned point clouds are in $DATA/binned"
