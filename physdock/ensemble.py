"""Conformational-ensemble analysis and filtering for PhysDock.

The generative diffusion models (DiffDock-L and Boltz-2) do not just return a
single static structure; they return a multi-seed set of predicted poses per
ligand. In traditional drug discovery, scientists run Molecular Dynamics (MD)
simulations to understand how a protein and ligand shift, vibrate, and explore
different geometric states. Because MD is computationally prohibitive at scale,
we use the generative pose set as a rapid surrogate "conformational landscape."

This script aims to replace traditional MDs by mathematically analyzing these ensembles.

Per ligand, we compute:
  * pose spread  : Mean pairwise heavy-atom RMSD across the generated ensemble.
                   A low spread indicates a highly confident, deep energy well
                   (peaked landscape). A high spread indicates a highly
                   flexible pocket or algorithmic uncertainty.
  * n_clusters   : Number of distinct binding hypotheses, calculated using
                   single-linkage clustering at `rmsd_cluster_threshold`.
  * confidence   : A gating mechanism (using DiffDock confidence or Boltz iptm)
                   to discard low-probability poses before they reach the
                   computationally expensive OpenMM physics relaxation stage.
"""
from __future__ import annotations                                                  
from itertools import combinations                                                  
from pathlib import Path                                                            
from typing import Optional                                                         

import numpy as np                                                                  
import pandas as pd                                                                 
from rdkit import Chem                                                              
from rdkit.Chem import AllChem                                                      

from .utils import get_logger                                                       # Import the internal logging utility for pipeline tracking

log = get_logger("physdock.ensemble")                                               # Instantiate the logger specifically for the ensemble module


def _load(sdf: str) -> Optional[Chem.Mol]:
    """
    Loads a 3D molecular conformation from an SDF file into an RDKit object.

    Reads a chemical structure file (SDF) and parses it into a computational 
    molecule object while stripping away explicit hydrogen atoms.
    
    Uses RDKit's SDMolSupplier to read the file. The `removeHs=True` flag is 
    critical because hydrogen positions fluctuate wildly and artificially 
    inflate RMSD calculations; we only care about the heavy-atom backbone.
    
    Args:
        sdf (str): The file path to the specific 3D SDF pose.
        
    Returns:
        Optional[Chem.Mol]: An RDKit molecule object ready for 3D alignment, or None if corrupted.
        
    Example:
        mol = _load("results/ligand_pose_seed_1.sdf")
        Returns an RDKit Mol object ready for 3D alignment, or None if corrupted.
    """
    return next(iter(Chem.SDMolSupplier(str(sdf), removeHs=True)), None)            # Read first mol from SDF, strip H-atoms, return None on fail


def _pose_rmsd(a: Chem.Mol, b: Chem.Mol) -> Optional[float]:
    """
    Calculates the Root Mean Square Deviation (RMSD) between two poses of the same ligand.

    Measures the geometric distance in 3D space between two different predicted 
    binding states of the same molecule.
    
    Crucially, it uses `GetBestRMS`, which is "symmetry-aware." If a drug has a 
    benzene ring, rotating that ring 180 degrees looks identical in physical 
    reality, but naive math would flag it as a huge deviation. This function 
    checks all symmetric automorphisms to find the truest, lowest RMSD.
        
    Args:
        a (Chem.Mol): The first 3D molecular pose.
        b (Chem.Mol): The second 3D molecular pose.
        
    Returns:
        Optional[float]: The calculated symmetric RMSD in Angstroms, or None if the calculation fails.
        
    Example:
        distance = _pose_rmsd(pose_1, pose_2)
        Returns 1.25 (meaning the poses are 1.25 Angstroms apart geometrically).
    """
    try:  # symmetry-aware best RMSD over automorphisms                             # Safely attempt to calculate the geometric distance
        return float(AllChem.GetBestRMS(Chem.Mol(a), Chem.Mol(b)))                  # Force copies of mols, calculate symmetric RMSD, cast to float
    except Exception:  # noqa: BLE001                                               # Catch any RDKit topology mismatch or internal crashes
        return None                                                                 # Fail gracefully by returning None instead of crashing


def _single_linkage(rmsd_matrix: np.ndarray, thr: float) -> int:
    """
    Clusters the conformational ensemble to count distinct "binding hypotheses."

    Takes a matrix of distances between all generated poses and groups them into 
    clusters. Poses closer together than the threshold (`thr`) merge into one cluster.
        
    Implements a classic connected-components algorithm using a Disjoint Set 
    (Union-Find) data structure. If pose A is close to pose B, and B is close 
    to C, they all become one cluster.
        
    Args:
        rmsd_matrix (np.ndarray): An NxN matrix of pairwise geometric distances between poses.
        thr (float): The distance threshold in Angstroms for merging poses into the same cluster.
        
    Returns:
        int: The total number of distinct clusters (binding hypotheses) identified.
        
    Example:
        matrix of 3 poses where poses 0 and 1 are 1.0 Angstroms apart, but pose 2 is 5.0 Angstroms away.
        n_clusters = _single_linkage(matrix, thr=2.0)
        Returns 2 (Poses 0 & 1 form Cluster A; Pose 2 forms Cluster B).
    """
    n = rmsd_matrix.shape[0]                                                        # Get the total number of poses from the matrix dimensions
    parent = list(range(n))                                                         # Initialize each pose as its own independent cluster parent

    def find(i):                                                                    # Define recursive helper function to find cluster root
        while parent[i] != i:                                                       # If the pose points to a different parent...
            parent[i] = parent[parent[i]]                                           # Path compression: point it directly to the grandparent
            i = parent[i]                                                           # Move up the tree hierarchy
        return i                                                                    # Return the ultimate root ID of this cluster

    for i, j in combinations(range(n), 2):                                          # Loop over every unique pair of poses (i, j)
        if rmsd_matrix[i, j] <= thr:                                                # If distance is below threshold, they belong together
            parent[find(i)] = find(j)                                               # Merge them by setting root of 'i' to point to root of 'j'
    return len({find(i) for i in range(n)})                                         # Return the count of unique cluster roots left at the end


def analyze(poses: pd.DataFrame, cfg, confidence_col: str = "confidence") -> pd.DataFrame:
    """
    Analyzes the entire conformational landscape for a batch of ligands.

    Takes a raw dataframe of all poses generated by the AI, groups them by 
    their respective ligand, and computes the ensemble statistics (spread, clusters).
    
    Groups the data, extracts the SDF files, and uses itertools to build a 
    pairwise RMSD matrix. It then calculates the mean spread of the upper 
    triangle of that matrix, runs the clustering algorithm, and returns a clean 
    summary table.
        
    Args:
        poses (pd.DataFrame): A DataFrame containing the paths to the generated 3D poses.
        cfg (dict/Config): The master configuration object defining the cluster threshold.
        confidence_col (str): The name of the DataFrame column containing the AI confidence scores.
        
    Returns:
        pd.DataFrame: A summary DataFrame containing ensemble metrics (spread, clusters, top confidence) per ligand.
        
    Example:
        summary_df = analyze(raw_diffdock_outputs, config)
        Returns a dataframe with columns: [ligand_id, n_poses, pose_spread_rmsd...]
    """
    thr = float(cfg.get("ensemble", "rmsd_cluster_threshold", default=2.0))         # Extract the clustering cutoff from the YAML config
    out = []                                                                        # Initialize an empty list to hold the summary row dictionaries
    for lid, grp in poses.groupby("ligand_id"):                                     # Iterate through the poses, grouped by the specific drug
        mols = [_load(p) for p in grp["pose_sdf"]]                                  # Load every SDF file for this drug into RDKit memory
        mols = [m for m in mols if m is not None]                                   # Filter out any files that failed to parse correctly
        spread, nclust = None, None                                                 # Initialize the output metrics as None by default
        if len(mols) >= 2:                                                          # We need at least 2 valid poses to calculate a landscape
            n = len(mols)                                                           # Store the count of valid poses
            mat = np.zeros((n, n))                                                  # Initialize a blank NxN matrix to hold pairwise distances
            for i, j in combinations(range(n), 2):                                  # Loop through every possible pair of poses
                r = _pose_rmsd(mols[i], mols[j])                                    # Calculate the symmetric geometric RMSD between pair
                mat[i, j] = mat[j, i] = r if r is not None else np.nan              # Populate the matrix symmetrically, injecting NaN on failure
            spread = float(np.nanmean(mat[np.triu_indices(n, 1)]))                  # Calculate average distance, ignoring diagonal and NaNs
            nclust = _single_linkage(np.nan_to_num(mat, nan=1e9), thr)              # Cluster the matrix, replacing NaNs with huge numbers
        conf = grp[confidence_col] if confidence_col in grp else pd.Series([np.nan])# Extract confidence scores safely, default to NaN if missing
        out.append({                                                                # Append the compiled statistics dictionary to our list
            "ligand_id": lid,                                                       # Log the name of the drug being analyzed
            "n_poses": len(mols),                                                   # Log how many valid AI poses we successfully parsed
            "pose_spread_rmsd": round(spread, 3) if spread is not None else None,   # Log the landscape spread, rounded for readability
            "n_clusters": nclust,                                                   # Log how many distinct binding hypotheses exist
            "best_confidence": float(conf.max()) if conf.notna().any() else None,   # Log the AI's highest confidence score for this drug
        })
    df = pd.DataFrame(out)                                                          # Convert the list of dictionaries into a clean Pandas dataframe
    log.info("Ensemble analysis for %d ligands", len(df))                           # Print a status update to the console/log file
    return df                                                                       # Return the finalized summary dataframe


def apply_confidence_gate(poses: pd.DataFrame, cfg, confidence_col: str = "confidence") -> pd.DataFrame:
    """
    Filters out garbage AI predictions before they burn expensive GPU compute.

    Reads the AI's self-reported confidence score for each pose and throws 
    away anything below the threshold defined in the configuration YAML.
        
    Uses pandas boolean masking. It treats missing scores as infinitely bad 
    (-1e9) to guarantee they are dropped. It also logs the exact number of 
    poses dropped for scientific transparency in the final report.

    Args:
        poses (pd.DataFrame): The raw DataFrame containing all generated poses and their AI confidence scores.
        cfg (dict/Config): The master configuration object defining the confidence gate threshold.
        confidence_col (str): The column name housing the confidence scores.
        
    Returns:
        pd.DataFrame: A filtered DataFrame keeping only the high-confidence poses.
        
    Example:
        filtered_poses = apply_confidence_gate(all_poses, config)
        Keeps 12 out of 16 poses, dropping 4 that the AI deemed unlikely.
    """
    gate = cfg.get("ensemble", "confidence_gate", "diffdock_min_confidence", default=-1.5)  # Extract the minimum allowed confidence from YAML
    if confidence_col not in poses:                                                         # Defensive check: if the AI didn't output scores...
        return poses                                                                        # ...skip gating entirely and return the raw dataframe
    before = len(poses)                                                                     # Record the total number of poses before filtering
    kept = poses[poses[confidence_col].fillna(-1e9) >= gate].copy()                         # Keep only rows where score >= gate; treat NaNs as -1e9
    log.info("Confidence gate >= %.2f: kept %d/%d poses", gate, len(kept), before)          # Log exact counts for pipeline transparency reports
    return kept                                                                             # Return the strictly filtered dataframe