#!/usr/bin/env python
"""
=============================================================================
PhysDock: Stage 01 — Target & Ground-Truth Preparation
=============================================================================
This script is the Data Engineering foundation of the pipeline. It automatically 
bridges raw, messy biological data (RCSB Protein Data Bank) to the strict, 
clean formats required by the generative AI models and the physics engine.

For every drug listed in the ligand manifest, this script:

1. Downloads the actual X-ray crystal structure of the protein-drug complex.
2. Slices the complex into a rigid receptor scaffolding (for DiffDock) and an 
   isolated, chemistry-repaired 3D ligand file (the RMSD ground truth).
3. Extracts the 1D amino acid sequence from the 3D protein (required for Boltz-2).
4. Compiles all file paths, sequences, and experimental affinities into a 
   single master JSON state file (`target_prep.json`).

By extracting the SMILES and 3D poses directly from physical reality, it 
eliminates human data-entry typos and ensures downstream evaluations are rigorous.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))                             # Dynamically add the parent directory to the system path to allow local imports
from physdock.config import load_config  # noqa: E402                                    # Import the custom YAML configuration loader
from physdock.receptor import prepare  # noqa: E402                                      # Import the complex PDB downloading and slicing logic from receptor.py
from physdock.utils import get_logger, save_json, ensure_dir  # noqa: E402               # Import basic utilities for logging and file I/O

log = get_logger("01_prepare")                                                           # Initialize the logger for this specific stage


def _seq_from_pdb(pdb_path: Path, chain: str) -> str:
    """
    Extracts the 1D amino acid sequence from a 3D PDB structure file.
    
    Reads the physical coordinates of a protein and translates 
    them back into the string of letters (amino acids) that make up its sequence. 
    This is strictly required for sequence-based co-folding models like Boltz-2.
    
    Uses BioPython's PDBParser to read the structure, filters 
    down to the specified chain, and uses PPBuilder (Polypeptide Builder) to 
    trace the carbon backbone and extract the sequence of continuous residues.
    
    Args:
        pdb_path (Path): Filepath to the cleaned protein PDB file.
        chain (str): The specific chain identifier to extract (e.g., "A").
        
    Returns:
        str: The continuous 1D amino acid sequence.
        
    Example:
        >>> _seq_from_pdb(Path("receptor.pdb"), "A")
        'MTEYKLVVVGAGGVGKSALTIQLI...'
    """
    from Bio.PDB import PDBParser, PPBuilder                                             # Import BioPython modules locally to delay loading until the function is called
    s = PDBParser(QUIET=True).get_structure("x", str(pdb_path))                          # Initialize the parser silently (ignoring PDB format warnings) and load the structure
    seq = ""                                                                             # Initialize an empty string to accumulate the amino acid sequence
    for ch in s[0]:                                                                      # Iterate through all chains found in the first structural model of the PDB
        if ch.id == chain:                                                               # Check if the current chain matches the target chain (e.g., Chain A)
            for pp in PPBuilder().build_peptides(ch):                                    # Use the Polypeptide Builder to trace the backbone and group continuous amino acids
                seq += str(pp.get_sequence())                                            # Convert the polypeptide chunk to a string and append it to the main sequence
    return seq                                                                           # Return the fully compiled 1D sequence

# The main function orchestrates the entire preparation process, from loading the config to 
# iterating through the ligand manifest and compiling the final JSON output.
def main(cfg_path):
    # Load the YAML configuration and prepare the workspace: ensure the root 'data' directory,
    # load the ligand manifest CSV, and extract the target chain identifier.
    cfg = load_config(cfg_path)                                                          # Load and expand the YAML configuration into a Python object
    work = ensure_dir("data")                                                            # Ensure the root 'data' directory exists, creating it if it does not
    manifest = pd.read_csv(cfg.get("ligands", "manifest_csv"), comment="#")              # Load the CSV manifest of drugs to test, ignoring lines starting with '#'
    chain = cfg.get("target", "receptor_chain", default="A")                             # Retrieve the target protein chain from the config, defaulting to 'A'

    # Iterate through each drug in the manifest, attempt to prepare the receptor and ligand files,  
    # extract the 1D sequence, and compile the results into a master dictionary.
    prepared = {}                                                                        # Initialize an empty dictionary to act as the master registry for processed targets
    seq = None                                                                           # Initialize the sequence variable as None (it will be populated once)
    for _, row in manifest.iterrows():                                                   # Iterate row by row through the loaded CSV manifest DataFrame
        lid, pdb, res = row["ligand_id"], str(row["pdb_id"]), str(row["ligand_resname"]) # Unpack the drug name, its PDB ID, and its 3-letter residue code into variables
        log.info("Preparing %s from %s:%s", lid, pdb, res)                               # Log the start of processing for the current drug
        try:                                                                             # Open a defensive try-block in case the PDB download or extraction fails
            # Downloads PDB, separates protein and ligand.
            pt = prepare(pdb, chain, res, work)                                          # Call the core extraction function 
        except Exception as e:  # noqa: BLE001                                           # Catch any structural biology or network errors
            log.error("Failed %s (%s) — skipping. Verify pdb_id/resname.", lid, e)       # Log the error but keep the script alive
            continue                                                                     # Skip to the next drug in the manifest without crashing the pipeline
        # If the sequence hasn't been extracted yet (first successful drug) and we have 
        # a valid receptor PDB, extract the 1D aminoacid sequence from the PDB file.
        if seq is None and pt.receptor_pdb.exists():                                     # Check if we still need the 1D sequence AND if a valid protein structure was just generated
            seq = _seq_from_pdb(pt.receptor_pdb, chain)                                  # Extract the 1D amino acid sequence from the newly generated protein file
        # Register the newly processed assets into the master dictionary under the drug's name.
        prepared[lid] = {                                                                
            "pdb_id": pdb,                                                               # Store the origin PDB code
            "receptor_pdb": str(pt.receptor_pdb),                                        # Store the file path to the cleaned, rigid protein scaffold
            "ref_sdf": str(pt.ligand_sdf) if pt.ligand_sdf else None,                    # Store the file path to the extracted 3D ground-truth ligand pose
            "smiles": pt.ligand_smiles,                                                  # Store the perfectly extracted 1D chemical formula (SMILES)
            "pocket_center": pt.pocket_center,                                           # Store the computed X,Y,Z coordinates of the binding pocket centroid
            "pchembl": (None if pd.isna(row.get("pchembl")) else float(row["pchembl"])), # Safely extract the experimental affinity value, converting Pandas NaNs to Python Nones
            "role": row.get("role", ""),                                                 # Extract any descriptive metadata about the drug's role
        }
    # After processing all drugs, compile the final output dictionary and flush it to disk as JSON.
    out = {"target": cfg.get("target", "name"), "protein_sequence": seq,                 # Construct the final JSON payload containing the target name, sequence, and...
           "ligands": prepared}                                                          # ...the entire nested dictionary of prepared ligand assets
    save_json(out, "data/processed/target_prep.json")                                    # Flush the final payload to disk as the master state file for the rest of the pipeline
    log.info("Prepared %d/%d ligands; protein length=%s",                                # Log the final success metrics
             len(prepared), len(manifest), len(seq) if seq else "?")                     # Report how many drugs succeeded vs failed, and the length of the extracted protein
    if not prepared:                                                                     # Safety check: if the dictionary is entirely empty (everything failed)...
        log.error("Nothing prepared. The manifest PDB/resname pairs are unverified "     # ...throw a critical error explaining why
                  "(verify=0). Confirm them on RCSB, then rerun.")                       # Remind the user about the manual verification safety switch

# Main function: triggered by the --config argument (path to the YAML config file) to kick off the entire target preparation process.
if __name__ == "__main__":
    ap = argparse.ArgumentParser()                                                       # Initialize the command-line argument parser
    ap.add_argument("--config", default="configs/kras_g12c.yaml")                        # Define the config argument, defaulting to the KRAS target
    main(ap.parse_args().config)                                                         # Parse the arguments and trigger the main function with the config path