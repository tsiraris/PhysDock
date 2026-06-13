#!/usr/bin/env bash
# =============================================================================
# PhysDock: The Master Orchestrator (run_all.sh)
# =============================================================================
# This script executes the entire PhysDock pipeline end-to-end. 
# It acts as the CI/CD (Continuous Integration / Continuous Deployment) trigger, 
# sequentially passing data from stage to stage. 
#
# Hardware Note: 
# - Stages 03 (DiffDock) and 04 (Boltz) require GPU acceleration.
# - All other stages (00, 01, 02, 05, 06, 07) run on the CPU.
#
# Usage: 
#   bash scripts/run_all.sh [config_file.yaml] 
#   (Defaults to configs/kras_g12c.yaml if no argument is provided)

# Enable strict shell execution mode (Fail-fast architecture):
# -e : Exit immediately if any command returns a non-zero (failure) exit status.
# -u : Treat unset variables as an error and exit immediately.
# -o pipefail : If any command in a pipeline fails (e.g., `cmd1 | cmd2`), 
#               fail the entire pipeline, not just the last command.
set -euo pipefail

# Parse the configuration argument using bash parameter substitution.
# If $1 (the first argument) is empty, fallback to "configs/kras_g12c.yaml".
CFG="${1:-configs/kras_g12c.yaml}"

# -----------------------------------------------------------------------------
# Execution Sequence
# -----------------------------------------------------------------------------

# Stage 00: Environment Verification (CPU)
# Proves dependencies exist and the core Python logic functions before burning GPU credits.
echo "== 00 setup check =="          ; python scripts/00_setup_check.py

# Stage 01: Biological Target Preparation (CPU)
# Downloads the X-ray crystal structures, cleans the protein, and extracts ground-truth drugs.
echo "== 01 prepare target =="       ; python scripts/01_prepare_target.py --config "$CFG"

# Stage 02: Cheminformatics Filter (CPU)
# Blocks mathematically invalid or un-synthesizable drugs from entering the expensive GPU queue.
echo "== 02 chem gate =="            ; python scripts/02_chem_gate.py --config "$CFG"

# Stage 03: Generative Diffusion Docking (GPU)
# Runs DiffDock-L to generate 16 structural hypotheses (the conformational ensemble) per valid drug.
echo "== 03 diffdock (GPU) =="       ; python scripts/03_run_diffdock.py --config "$CFG"

# Stage 04: Joint Co-Folding and Affinity Prediction (GPU)
# Runs Boltz-2. Hardcoded with `--max-ligands 4` to strictly cap AWS budget expenditure during automated runs.
echo "== 04 boltz (GPU, capped) =="  ; python scripts/04_run_boltz.py --config "$CFG" --max-ligands 4

# Stage 05: Thermodynamic Physics Rescoring (GPU/CPU)
# Cleans AI hallucinations using OpenMM. Hardcoded to only process the top 3 highest-confidence poses per drug.
echo "== 05 physics rescore =="      ; python scripts/05_physics_rescore.py --config "$CFG" --top-per-ligand 3

# Stage 06: Conformational Landscape Analysis (CPU)
# Evaluates the spread and clustering of the 16 poses from Stage 03 to determine structural stability.
echo "== 06 ensemble =="             ; python scripts/06_ensemble_analysis.py --config "$CFG"

# Stage 07: Statistical Validation & Reporting (CPU)
# Aggregates all data, calculates geometric RMSD and Spearman rank correlation, and generates Markdown reports.
echo "== 07 evaluate + report =="    ; python scripts/07_evaluate_and_report.py --config "$CFG"

# Final Status Message
echo "DONE. See results/report/report.md"