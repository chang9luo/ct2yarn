#!/usr/bin/env bash
# Create the conda environment for CT2Yarn.
#   bash setup_env.sh            # environment name: ct2yarn
#   bash setup_env.sh my_env     # custom name
# If an environment with that name already exists, it is removed and rebuilt.
set -euo pipefail

ENV_NAME="${1:-ct2yarn}"
PY_VERSION="3.10"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v conda >/dev/null 2>&1; then
    echo "conda not found. Install Miniconda first: https://docs.conda.io/en/latest/miniconda.html" >&2
    exit 1
fi

if conda run -n "$ENV_NAME" true >/dev/null 2>&1; then
    if [ "$ENV_NAME" = "base" ]; then
        echo "Refusing to remove the conda base environment. Choose another name." >&2
        exit 1
    fi
    if [ "${CONDA_DEFAULT_ENV:-}" = "$ENV_NAME" ] || [ "$(basename "${CONDA_PREFIX:-}")" = "$ENV_NAME" ]; then
        echo "Environment '$ENV_NAME' is currently active. Run 'conda deactivate' first, then re-run this script." >&2
        exit 1
    fi
    echo "==> Environment '$ENV_NAME' already exists, removing it"
    conda env remove -y -n "$ENV_NAME"
fi

echo "==> Creating conda environment '$ENV_NAME' (Python $PY_VERSION)"
conda create -y -n "$ENV_NAME" --override-channels -c conda-forge "python=$PY_VERSION" pip

echo "==> Installing Python packages from requirements.txt"
conda run -n "$ENV_NAME" python -m pip install -r "$ROOT/requirements.txt"

echo "==> Checking imports"
conda run -n "$ENV_NAME" python -c "import cupy, drjit, matplotlib, mitsuba, nrrd, numpy, polyscope, rich, scipy, skimage, torch, yaml; print('all imports ok | torch', torch.__version__, '| CUDA available:', torch.cuda.is_available(), '| cupy', cupy.__version__, '| mitsuba', mitsuba.__version__)"

echo
echo "Done. Activate with:  conda activate $ENV_NAME"
