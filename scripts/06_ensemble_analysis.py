#!/usr/bin/env python
"""
=============================================================================
PhysDock: Stage 06 — Conformational Ensemble Analysis & MD Surrogate
=============================================================================
This script directly addresses the industry need to accelerate/replace 
traditional molecular dynamics (MD) simulations. 

Generative diffusion models (like DiffDock-L) output a multi-seed "ensemble" 
of poses rather than a single static structure. This script treats that 
ensemble as a static map of the binding landscape, mathematically analyzing 
it to extract structural thermodynamics without spending weeks computing MD 
trajectories.

Architecturally, this module operates as a decoupled, parallel microservice. 
While Stage 05 (Physics) calculates absolute thermodynamic energy, this script 
independently evaluates overall structural flexibility. It performs three 
critical analyses:

1. The Confidence Gate: Discards poses with low algorithmic probability, 
   ensuring that downstream landscape mathematics are calculated strictly 
   on highly probable AI hypotheses, eliminating mathematical noise.
   
2. Pose Spread (Flexibility): Calculates the mean pairwise RMSD across the 
   surviving ensemble. A tight spread indicates a deep, highly stable 
   energy well. A wide spread indicates a highly flexible binding pocket 
   or algorithmic uncertainty.
   
3. Cluster Analysis: Uses single-linkage clustering to evaluate how many 
   distinct binding hypotheses (e.g., "head-first" vs. "tail-first") the 
   AI generated.

The output is a refined statistical dataset that, when combined with the 
thermodynamic scores from Stage 05, provides a comprehensive, multi-faceted 
evaluation of the drug candidate.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))                             # Dynamically append the parent directory to the Python path to allow internal module imports
from physdock import ensemble                                                            # Import the core ensemble mathematics and clustering logic module
from physdock.config import load_config                                                  # Import the YAML configuration loader to fetch threshold parameters
from physdock.utils import get_logger, ensure_dir                                        # Import utility functions for standardized logging and filesystem management

log = get_logger("06_ensemble")                                                          # Initialize the module-level logger under the specific tag "06_ensemble"


def main(cfg_path):
    # Load the YAML configuration file, which contains parameters like the confidence threshold for gating and clustering distance cutoffs.
    cfg = load_config(cfg_path)                                                          # Parse the target configuration YAML file into a Config object
    # Load the DiffDock poses
    poses_csv = Path("results/diffdock/poses.csv")                                       # Define the hardcoded path to the expected output file from Stage 03 (DiffDock)
    if not poses_csv.exists():                                                           # Implement a defensive check to ensure the required upstream data actually exists
        log.error("No DiffDock poses; run stage 03 first."); return                      # If missing, log a fatal error instructing the user on the correct pipeline order and exit
    poses = pd.read_csv(poses_csv)                                                       # Load the raw, unfiltered generative pose dataset into a Pandas DataFrame
    # Apply the YAML-defined confidence threshold to filter out low-probability AI hallucinations
    gated = ensemble.apply_confidence_gate(poses, cfg)                                   
    # Execute the core mathematical analysis (spread and clustering) on the surviving high-confidence poses
    df = ensemble.analyze(gated, cfg)                                                    
    # Write the analysis results to CSV and log the output
    out = ensure_dir("results/ensemble") / "ensemble.csv"                                # Ensure the 'results/ensemble' directory exists, and construct the final output file path
    df.to_csv(out, index=False)                                                          # Serialize the mathematically analyzed DataFrame to disk as a CSV, omitting the index column
    log.info("Ensemble -> %s\n%s", out, df.to_string(index=False))                       # Log the successful save location and print a clean string representation of the data to the console


if __name__ == "__main__":
    ap = argparse.ArgumentParser()                                                       # Initialize the standard command-line argument parser
    ap.add_argument("--config", default="configs/kras_g12c.yaml")                        # Define the optional argument for the configuration file, defaulting to the KRAS target
    main(ap.parse_args().config)                                                         # Parse the user's terminal arguments and trigger the main orchestrator function