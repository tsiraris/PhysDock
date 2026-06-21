#!/usr/bin/env bash
# ============================================================================
# setup_diffdock.sh — reproducible DiffDock-L environment for PhysDock
#
# DiffDock-L's install is ORDER-DEPENDENT and cannot be expressed as a flat
# conda YAML: PyTorch must come from the CUDA-12.1 wheel index, the PyTorch
# Geometric companions must be the wheels built against that exact torch+CUDA,
# and ProDy must be installed with --no-deps (its metadata over-pins numpy<1.24,
# which would otherwise drag the whole stack down and break torch-scatter).
#
# This script encodes that sequence. It was validated by building from scratch
# on a fresh AWS SageMaker ml.g5.2xlarge (A10G, CUDA 12) instance.
#
# Usage:
#   conda create -y -n diffdock python=3.9
#   bash setup_diffdock.sh
#   git clone https://github.com/gcorso/DiffDock ~/DiffDock
#   export PHYSDOCK_DIFFDOCK_DIR=~/DiffDock
# ============================================================================
set -euo pipefail

ENV=diffdock

# 1. PyTorch 2.2.2 + CUDA 12.1 (DiffDock-L targets the torch-2.2 line).
conda run -n "$ENV" python -m pip install torch==2.2.2 \
  --index-url https://download.pytorch.org/whl/cu121

# 2. PyTorch Geometric core.
conda run -n "$ENV" python -m pip install torch-geometric

# 3. PyG compiled companions — MUST match torch 2.2 + cu121 exactly.
conda run -n "$ENV" python -m pip install torch-scatter torch-sparse torch-cluster \
  -f https://data.pyg.org/whl/torch-2.2.2+cu121.html

# 4. DiffDock-L's other runtime deps. Pin numpy 1.26.4: torch 2.2.2 and the PyG
#    wheels are compiled against the numpy-1.x ABI, and ProDy 2.4.1 still calls
#    numpy.alltrue (removed in numpy 2.0). Without this pin, pip silently
#    upgrades numpy to 2.x and both torch and ProDy fail to import.
conda run -n "$ENV" python -m pip install "numpy==1.26.4" e3nn fair-esm rdkit pyyaml pandas biopython

# 5. ProDy LAST, with --no-deps, so its numpy<1.24 metadata cap does not
#    downgrade numpy (which would break the compiled torch-scatter/cluster).
conda run -n "$ENV" python -m pip install --no-deps prody==2.4.1

# 6. Re-assert numpy 1.26.4 in case a step above nudged it, then verify.
conda run -n "$ENV" python -m pip install "numpy==1.26.4"

# 6. Verify the full stack imports and CUDA is visible.
conda run -n "$ENV" python -c "import torch, torch_geometric, torch_scatter, torch_cluster, torch_sparse, e3nn, esm, rdkit, prody; print('diffdock OK', torch.__version__, 'cuda', torch.cuda.is_available())"

echo
echo "DiffDock-L environment ready. Remember:"
echo "  git clone https://github.com/gcorso/DiffDock ~/DiffDock"
echo "  export PHYSDOCK_DIFFDOCK_DIR=~/DiffDock"