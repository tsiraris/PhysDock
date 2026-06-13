"""Evaluation Module — The Validation Engine of PhysDock

This script transforms PhysDock from a theoretical generative demo into a mathematically 
validated scientific pipeline. It enforces strict, falsifiable checks against ground-truth data 
to prove the efficacy of the AI models (DiffDock-L and Boltz-2) and the physics engine (OpenMM).

It performs two independent validations:
  
  1. Geometry (Pose Accuracy): Calculates the heavy-atom Root Mean Square Deviation (RMSD) 
     between the DiffDock's predicted 3D pose and the real X-ray crystal structure (ground-truth) 
     extracted in stage 01. A pose is deemed a "success" if the RMSD <= 2.0 Å (the industry standard).
  
  2. Function (Affinity Ranking): Calculates the Spearman rank correlation coefficient (rho) between 
     the AI's predicted binding score (either Boltz affinity or OpenMM interaction energy after DiffDock pose refinement) 
     and the actual experimental wet-lab data (pChEMBL). Spearman rank is utilized to prioritize the correct 
     ranking of the best candidates over the prediction of the absolute thermodynamic kcal/mol perfectly.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign
from scipy.stats import spearmanr

from .utils import get_logger

log = get_logger("physdock.evaluate")

MIN_POINTS_FOR_CORR = 4


def pose_rmsd_vs_reference(pred_sdf: str, ref_sdf: str) -> Optional[float]:
    """
    Calculates the symmetry-corrected heavy-atom RMSD between a predicted pose and a reference pose.
    
    This function measures the 3D geometric deviation between the AI's prediction and the real crystal. 
    It attempts to use the `spyrmsd` library first, which rigorously handles molecular symmetry (e.g., 
    if a benzene ring rotates 180 degrees, it shouldn't be penalized). If `spyrmsd` fails or is not 
    installed, it falls back to RDKit's `rdMolAlign.CalcRMS`, which is also symmetry-aware and,
    crucially, computes the RMSD IN-FRAME (no superposition) — consistent with this pipeline's
    shared coordinate frame (the receptor is the fixed anchor, so predicted and crystal ligands
    already live on the same XYZ grid and must NOT be re-aligned).
    
    Args:
        pred_sdf (str): File path to the SDF file containing the AI-predicted ligand pose.
        ref_sdf (str): File path to the SDF file containing the ground-truth crystal ligand pose.
        
    Returns:
        Optional[float]: The calculated RMSD value in Angstroms, or None if calculation fails.
        
    Example:
        >>> rmsd = pose_rmsd_vs_reference("results/poses/mol_pred.sdf", "data/ref/mol_crystal.sdf")
        >>> print(rmsd)
        1.45
    """
    # Attempt to use the spyrmsd library for symmetry-corrected RMSD: Convert the (heavy-atom) RDKit ligand reference and predicted molecules 
    # to spyrmsd Molecule objects, and use the 3D coordinates, atomic numbers, and connectivity to calculate the RMSD, handling symmetric cases correctly. 
    try:                                                                        
        from spyrmsd import rmsd as srmsd, molecule as smol                     # Import the required spyrmsd modules locally to avoid global dependency issues
        ref = smol.Molecule.from_rdkit(_read(ref_sdf))                          # Convert the reference RDKit molecule into a spyrmsd Molecule object
        mob = smol.Molecule.from_rdkit(_read(pred_sdf))                         # Convert the predicted (mobile) RDKit molecule into a spyrmsd Molecule object
        return float(srmsd.symmrmsd(                                            # Calculate and return the symmetry-corrected RMSD...
            ref.coordinates, mob.coordinates,                                   # ...using the 3D coordinates of both molecules
            ref.atomicnums, mob.atomicnums,                                     # ...and their atomic numbers (to match atom types)
            ref.adjacency_matrix, mob.adjacency_matrix))                        # ...and their connectivity graphs (to resolve symmetry)
    # Fallback: RDKit's CalcRMS
    except Exception as e:  # noqa: BLE001                                      # Catch any exceptions (e.g., spyrmsd missing or topology mismatch)
        log.info("spyrmsd unavailable/failed (%s); RDKit CalcRMS (in-frame) fallback.", e) # Log a notice that the script is falling back to RDKit
        # CalcRMS is also symmetry-aware (it minimises over atom-mapping automorphisms) and does not superimpose the molecules. 
        # Like spyrmsd, it measures the RMSD in the existing shared coordinate frame, so it wouldn't wrongly reward a pose that is in the wrong pocket location but the right shape.
        try:                                                                    # Attempt the fallback calculation
            return float(rdMolAlign.CalcRMS(_read(pred_sdf), _read(ref_sdf)))   # In-frame, symmetry-aware RMSD (probe=pred, ref=crystal); no superposition
        except Exception as e2:  # noqa: BLE001                                 # Catch exceptions if RDKit also fails (e.g., invalid SDF)
            log.warning("RMSD failed: %s", e2)                                  # Log a warning that the molecule cannot be evaluated
            return None                                                         # Return None to safely indicate a failed evaluation


def _read(sdf: str) -> Chem.Mol:
    """
    Helper function to robustly load an SDF file as an RDKit Mol object, stripping hydrogens.
    
    In structural biology, RMSD is traditionally calculated using only "heavy atoms" (Carbon, 
    Oxygen, Nitrogen, etc.) because hydrogen positions are rarely resolved accurately in X-ray 
    crystallography. This function ensures explicit hydrogens are removed before processing.
    
    Args:
        sdf (str): File path to the SDF file.
        
    Returns:
        Chem.Mol: The parsed RDKit Molecule object.
        
    Raises:
        ValueError: If the SDF file cannot be read or parsed by RDKit.
        
    Example:
        >>> mol = _read("my_ligand.sdf")
        >>> type(mol)
        <class 'rdkit.Chem.rdchem.Mol'>
    """
    m = next(iter(Chem.SDMolSupplier(str(sdf), removeHs=True)), None)           # Load the first molecule from the SDF, explicitly removing hydrogens
    if m is None:                                                               # Check if the resulting molecule object is None (parsing failed)
        raise ValueError(f"unreadable SDF: {sdf}")                              # Raise an error to halt the process and flag the corrupt file
    return m                                                                    # Return the successfully instantiated RDKit molecule object


def evaluate_poses(top_poses: pd.DataFrame, references: dict, cfg) -> pd.DataFrame:
    """
    Batch evaluates a dataset of predicted poses against their corresponding ground truths.
    
    It iterates through the generated poses, looks up the correct reference file, calculates 
    the RMSD, and applies a boolean "success" flag based on the config threshold (usually 2.0 Å).
    
    Args:
        top_poses (pd.DataFrame): DataFrame containing 'ligand_id' and 'pose_sdf' paths.
        references (dict): A dictionary mapping {ligand_id: reference_sdf_path}.
        cfg (DictConfig/dict): The loaded configuration object (from target_config.yaml).
        
    Returns:
        pd.DataFrame: A DataFrame appending the calculated 'pose_rmsd' and 'pose_success' boolean.
        
    Example:
        >>> df_results = evaluate_poses(preds_df, ref_dict, config)
        >>> print(df_results.head(1))
           ligand_id  pose_rmsd  pose_success
        0  sotorasib      1.25          True
    """
    # Extract the RMSD success threshold from the configuration file (YAML), iterate over each predicted ligand pose, and calculate the RMSD against the reference structure if available. 
    # Then determine if the pose is a "success" based on whether the RMSD is below or equal to the threshold, and compile these results into a new DataFrame.
    thr = float(cfg.get("evaluate", "rmsd_success_threshold", default=2.0))     # Extract the pass/fail RMSD threshold from the config file
    rows = []                                                                   # Initialize an empty list to store the evaluation results
    for _, r in top_poses.iterrows():                                           # Iterate over every row (each predicted ligand pose) in the DataFrame
        ref = references.get(r["ligand_id"])                                    # Retrieve the specific ground-truth SDF path for this ligand ID
        rmsd = pose_rmsd_vs_reference(r["pose_sdf"], ref) if ref else None      # Calculate RMSD if the reference exists, otherwise set to None
        rows.append({                                                           # Append a new dictionary of results for this ligand
            "ligand_id": r["ligand_id"],                                        # Record the ligand ID
            "pose_rmsd": round(rmsd, 3) if rmsd is not None else None,          # Record the RMSD, rounded to 3 decimal places for readability
            "pose_success": (rmsd is not None and rmsd <= thr),                 # Record True if RMSD is below or equal to threshold, else False
        })                                                                      
    df = pd.DataFrame(rows)                                                     # Convert the list of dictionaries into a new pandas DataFrame
    if not df.empty:                                                            # Check if the generated DataFrame actually contains data
        scored = df["pose_rmsd"].notna().sum()                                  # Count the total number of ligands that successfully received an RMSD score
        sr = df["pose_success"].sum()                                           # Count the total number of ligands that passed the success threshold
        log.info("Pose eval: %d/%d scored, %d <= %.1f A", scored, len(df), sr, thr) # Log the final summary statistics for the user
    return df                                                                   # Return the populated DataFrame containing the pose metrics


def affinity_correlation(pred: pd.DataFrame, label_col: str,
                         pred_col: str) -> dict:
    """
    Calculates the Spearman rank correlation between AI-predicted scores and lab-derived labels.
    
    This evaluates how well the model can rank a list of drugs from "best binder" to "worst binder".
    It includes a strict statistical safeguard: if there are fewer than MIN_POINTS_FOR_CORR (4) 
    labeled data points, it refuses to calculate a correlation to prevent reporting mathematically 
    meaningless or flattering "fluke" statistics.
    
    Args:
        pred (pd.DataFrame): DataFrame containing both the predicted scores and the ground truth labels.
        label_col (str): The column name holding the experimental ground truth (e.g., 'pchembl').
        pred_col (str): The column name holding the AI's prediction (e.g., 'boltz_affinity').
        
    Returns:
        dict: A dictionary containing the correlation coefficient (rho), p-value (p), sample size (n),
              and a 'usable' boolean flag indicating if the statistic is statistically valid.
              
    Example:
        >>> stats = affinity_correlation(df, "pchembl", "boltz_score")
        >>> print(stats)
        {'pred_col': 'boltz_score', 'n': 12, 'rho': 0.85, 'p': 0.001, 'usable': True, ...}
    """
    # Select the experimental label and predicted score columns from the DataFrame, 
    # convert them to numeric (coercing errors to NaN), and drop any rows with NaN values to ensure a clean dataset for correlation.
    sub = pred[[label_col, pred_col]].apply(pd.to_numeric, errors="coerce").dropna() # Select columns, force string values to numeric (invalid->NaN), and drop NaN rows
    n = len(sub)                                                                # Count the remaining number of valid, paired data points
    # Check if the sample size is below the minimum threshold required for correlation analysis. 
    # If it is, return a dictionary with null statistics, flag the result as unusable, and provide a clear explanation.
    if n < MIN_POINTS_FOR_CORR:                                                 
        return {"pred_col": pred_col, "n": n, "rho": None, "p": None,           # Return a dictionary with null statistics...
                "usable": False,                                                # ...flag the result as unusable...
                "note": f"underpowered: only {n} labelled points "              # ...and provide a clear explanation that the data is underpowered
                        f"(need >= {MIN_POINTS_FOR_CORR}); fill pchembl in the manifest"}
    # If there are enough data points, calculate the Spearman rank correlation coefficient (rho) 
    # and p-value using scipy's spearmanr function, and return these statistics in a dictionary, flagging the result as usable.
    rho, p = spearmanr(sub[pred_col], sub[label_col])                           # Calculate the Spearman rank correlation and p-value using scipy
    return {"pred_col": pred_col, "n": n, "rho": round(float(rho), 3),          # Return the calculated rho, rounded to 3 decimal places...
            "p": round(float(p), 4), "usable": True,                            # ...and the p-value, flagging the statistic as fully usable
            "note": "Spearman rank correlation (predicted score vs experimental)"} # Provide a summary note of the calculation performed