#!/usr/bin/env python
"""
=============================================================================
PhysDock: Stage 07 — Scientific Validation & Executive Reporting
=============================================================================
This is the culmination of the PhysDock pipeline. While the previous stages 
generate data, this stage generates knowledge. 

It acts as a fault-tolerant aggregator, sweeping up the outputs of the 
decoupled microservices (DiffDock geometry, Boltz affinity, OpenMM physics). 
It mathematically compares these AI predictions against physical ground-truth 
data (X-ray crystal structures and wet-lab pChEMBL affinities).

By executing strict geometric validation (RMSD) and functional validation 
(Spearman rank correlation), it produces a centralized, falsifiable master 
ledger (`merged.csv`) and an automated Markdown report. This is the exact 
handoff point where Computational Chemistry delivers actionable insights 
to Medicinal Chemistry.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))                             # Dynamically insert the parent directory into Python's path to allow internal module resolution
from physdock import evaluate, report                                                    # Import the mathematical evaluation engine and the markdown/figure reporting module
from physdock.config import load_config                                                  # Import the YAML configuration loader
from physdock.utils import get_logger, load_json, save_json, ensure_dir                  # Import utility functions for logging, JSON I/O, and directory management

log = get_logger("07_eval")                                                              # Initialize the module-level logger under the specific tag "07_eval"


def _read(p):
    """
    Safely reads a CSV file into a Pandas DataFrame, returning an empty frame if missing.
    
    Acts as a fault-tolerant data ingester for decoupled pipelines. 
    If a researcher disabled an expensive stage (like Boltz-2) to save budget, 
    this function ensures the final report still compiles using whatever data is available.
    
    Checks if the `Path` exists on the filesystem. If true, it loads it via `pd.read_csv`. 
    If false, it returns an empty `pd.DataFrame()`.
    
    Args:
        p (str | Path): The filesystem path to the requested CSV file.
        
    Returns:
        pd.DataFrame: The loaded data, or an empty DataFrame if the file was not found.
        
    Example:
        >>> df = _read("results/boltz/boltz.csv")
        >>> type(df)
        <class 'pandas.core.frame.DataFrame'>
    """
    p = Path(p)                                                                          # Cast the incoming string path to a robust Pathlib object
    return pd.read_csv(p) if p.exists() else pd.DataFrame()                              # Return the parsed CSV if the file exists; otherwise, gracefully return an empty DataFrame


def main(cfg_path):
    # Load config YAML, target_prep JSON, and extract the label column name from the config (defaulting to 'pchembl' if not specified)
    cfg = load_config(cfg_path)                                                          # Load and parse the project configuration YAML
    prep = load_json("data/processed/target_prep.json")                                  # Load the master dictionary containing the ground-truth sequence and ligand metadata
    label_col = cfg.get("evaluate", "affinity_label_column", default="pchembl")          # Extract the column name holding the wet-lab affinity labels (defaulting to 'pchembl')

    # Create the foundational DataFrame with one row per ligand, containing the core metadata and the experimental 
    # affinity labels. This will be the base onto which the various evaluation metrics will be merged.
    base = pd.DataFrame([                                                                # Initialize the foundation DataFrame using list comprehension over the prepped metadata
        {"ligand_id": lid, "role": i.get("role"), label_col: i.get("pchembl"),           # Extract core identity, role, and experimental ground-truth affinity...
         "smiles": i.get("smiles")}                                                      # ...and the verified SMILES string for every drug processed in Stage 01
        for lid, i in prep["ligands"].items()])                                          # Loop through the master target dictionary
    # Build a rapid lookup dictionary mapping ligand IDs to their ground-truth crystal structures (SDF paths)
    refs = {lid: i.get("ref_sdf") for lid, i in prep["ligands"].items()}                 
    # Attempt to load the raw DiffDock poses (fallback to an empty DataFrame with the expected columns), 
    # sort them by AI confidence, and keep only the top-1 ranked pose (AI's most confident) per drug. 
    poses = _read("results/diffdock/poses.csv")                                          # Attempt to load the raw generative poses from Tier A
    top = (poses.sort_values(["ligand_id", "rank"]).groupby("ligand_id").head(1)         # Sort the poses by AI confidence and slice strictly the #1 rank pose per drug...
           if not poses.empty else pd.DataFrame(columns=["ligand_id", "pose_sdf"]))      # ...fallback to an empty scaffold if DiffDock failed or was skipped
    # Run the heavy-atom RMSD evaluation between the crystal structures and the most confident (DiffDock-L only) poses per ligand.
    pose_eval = evaluate.evaluate_poses(top[["ligand_id", "pose_sdf"]], refs, cfg) \
        if not top.empty else pd.DataFrame()                                             
    # Attempt to load the Boltz-2 affinity predictions and thermodynamic relaxation metrics (fallback to an empty DataFrame if missing) 
    # and slice the best pose per drug based on internal strain energy (lowest/kcal is best).
    boltz = _read("results/boltz/boltz.csv")                                             # Attempt to load the affinity and joint-folding metrics from Tier B
    physics = _read("results/physics/physics.csv")                                       # Attempt to load the thermodynamic relaxation metrics from the OpenMM proxy
    phys_best = (physics.sort_values("ligand_strain_kcal")                               # Sort the physics results by internal structural strain (lowest/best first)...
                 .groupby("ligand_id").head(1)) if not physics.empty else pd.DataFrame() # ...and slice the single most thermodynamically stable pose per drug
    # Initialize the master merged ledger with the foundational metadata DataFrame, then iteratively merge in specific columns from
    # the pose evaluation, Boltz-2, and physics metrics dataframes (only if they were successfully generated) using left joins on 'ligand_id'.
    merged = base                                                                        # Initialize the master merged ledger with the foundational metadata DataFrame
    for extra, cols in [(pose_eval, ["ligand_id", "pose_rmsd", "pose_success"]),         # Define a list of tuples: (DataFrame to merge, [List of specific columns to extract])
                        (boltz, [c for c in ["ligand_id", "iptm", "affinity_pred_value", # Dynamically extract Boltz columns only if they successfully generated
                                             "affinity_prob_binary"] if c in boltz.columns]),
                        (phys_best, [c for c in ["ligand_id", "ligand_strain_kcal",      # Dynamically extract Physics columns only if they successfully generated
                                                 "interaction_energy_kj", "pose_drift_rmsd"]
                                     if c in phys_best.columns])]:
        if not extra.empty:                                                              # Check if the specific microservice actually produced data
            merged = merged.merge(extra[cols], on="ligand_id", how="left")               # Perform a SQL-style LEFT JOIN to attach the metrics to the master ledger via the 'ligand_id' key

    # Iterate through the AI and physics (boltz affinity, ligand_strain and interaction_energy) columns of the merged dataset, and if they exist, 
    # calculate the Spearman rank correlation against the wet-lab affinity labels (pchembl), and append the results to the `corr` list. 
    corr = []                                                                            # Initialize an empty list to accumulate the Spearman rank correlation dictionaries
    for pred_col in ["affinity_pred_value", "ligand_strain_kcal", "interaction_energy_kj"]: # Iterate through the AI and physics scoring columns
        if pred_col in merged.columns:                                                   # Ensure the scoring column actually exists in the merged dataset
            corr.append(evaluate.affinity_correlation(merged, label_col, pred_col))      # Calculate the correlation against the wet-lab data and append the statistical result
    # Command the report module to generate Markdown files and Matplotlib scatter plots.
    rep = report.build("results/report", merged, pose_eval, corr, cfg)                   # Command the report module to generate Markdown files and Matplotlib scatter plots
    # Serialize a quick-read JSON summary of the core pipeline success metrics, and save the 
    # ultimate master ledger (merged) to disk for human review or downstream programmatic ingestion.
    save_json({"correlations": corr,                                                     # Serialize a quick-read JSON summary of the core pipeline success metrics...
               "n_pose_success": int(pose_eval["pose_success"].sum()) if not pose_eval.empty else 0}, # ...calculating exactly how many poses achieved an RMSD <= 2.0 Angstroms
              "results/report/summary.json")                                             # Save this summary payload to the report directory
    merged.to_csv("results/report/merged.csv", index=False)                              # Serialize the ultimate master ledger to disk for human review or downstream programmatic ingestion
    log.info("Report ready: %s", rep)                                                    # Log the final triumph of the pipeline, providing the path to the human-readable Markdown report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()                                                       # Initialize the CLI argument parser
    ap.add_argument("--config", default="configs/kras_g12c.yaml")                        # Accept the target configuration file
    main(ap.parse_args().config)                                                         # Execute the final aggregation and reporting stage