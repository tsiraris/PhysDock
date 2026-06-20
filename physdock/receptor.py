"""Receptor & reference-pose preparation module.

This script bridges the gap between raw biological data (PDB files) and the strict 
chemical requirements of generative AI and physics engines. 

From a raw co-crystal PDB, it extracts three critical components:
  1. Receptor Scaffold: A cleaned protein chain (no waters, no salts) to act as 
     the rigid docking/co-folding target.
  2. Ground-Truth Ligand Pose: The bound drug is extracted as a fully valid 3D 
     molecule. Because raw PDB files lack chemical bond orders (single/double/aromatic), 
     this script dynamically fetches the ideal chemical topology from the RCSB Chemical 
     Component Dictionary and uses RDKit to template the correct bonds onto the 3D coordinates.
  3. Pocket Centroid: The geometric center of the bound drug is calculated to 
     tell the diffusion models exactly where the binding pocket is located.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import re
import requests
from rdkit import Chem
from rdkit.Chem import AllChem

from .utils import ensure_dir, fetch_pdb, get_logger

log = get_logger("physdock.receptor")                                           # Initialize the module-level logger for execution tracking

_AA3 = {                                                                        # Define a constant set of valid 3-letter amino acid codes
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",       # Include standard amino acids (A-I)
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",       # Include standard amino acids (L-V)
    "MSE", "SEC", "PYL",                                                        # Include rare/modified amino acids (Selenocysteine, etc.)
}                                                                               # Close the set definition


@dataclass
class PreparedTarget:
    """
    Data structure holding the paths and metadata of a successfully processed target.
    """
    pdb_id: str                                                                 # The 4-letter PDB identifier (e.g., '6OIM')
    receptor_pdb: Path                                                          # Filepath to the cleaned, protein-only PDB file
    ligand_sdf: Optional[Path]                                                  # Filepath to the extracted ligand with correct 3D bond orders
    ligand_smiles: Optional[str]                                                # The 1D chemical string (SMILES) of the extracted ligand
    pocket_center: Optional[tuple]                                              # The (X, Y, Z) coordinates of the binding pocket centroid


def _ccd_smiles(resname: str) -> Optional[str]:
    """
    Fetches the ideal chemical SMILES string for a specific ligand from the RCSB 
    Chemical Component Dictionary (CCD) API.
    
    Due to the fact that raw PDB files only contain atom coordinates and not bond orders, 
    we need the CCD's ideal SMILES to know where the double/aromatic bonds belong.

    Args:
        resname (str): The 3-letter HETATM code of the ligand (e.g., "6O7").

    Returns:
        Optional[str]: The SMILES string if successful, None if the network request fails.

    Example:
        >>> _ccd_smiles("6O7")
        'C1=CC(=C(C=C1F)O)C2=C(C(=O)N(C2=O)C3C(CC(C(C3)C(F)(F)F)C)N4CCN(CC4)C(=O)C=C)C(C)C'
    """
    url = f"https://files.rcsb.org/ligands/download/{resname.upper()}.cif"      # Canonical CCD component file for this 3-letter code
    try:                                                                        # Begin error-handling block for the external network request
        r = requests.get(url, timeout=30)                                       # Execute HTTP GET request with a strict 30-second timeout
        r.raise_for_status()                                                    # Throw an exception immediately if the HTTP status is not 200 OK
        text = r.text                                                           # Raw mmCIF text of the chemical component
        cactvs_canon, any_canon, any_smiles = None, None, None                  # Candidate SMILES in order of preference
        for line in text.splitlines():                                          # Scan each descriptor row in the CIF
            if "SMILES" not in line:
                continue
            m = re.search(r'"([^"]+)"', line)                                   # SMILES payload is the quoted field
            if not m:
                continue
            smi = m.group(1).strip()
            if "SMILES_CANONICAL" in line and "CACTVS" in line:
                cactvs_canon = cactvs_canon or smi
            elif "SMILES_CANONICAL" in line:
                any_canon = any_canon or smi
            else:
                any_smiles = any_smiles or smi
        return cactvs_canon or any_canon or any_smiles                          # Best available SMILES, or None
    except Exception as e:  # noqa: BLE001                                      # Catch any network or parsing exceptions (ignoring broad exception lint)
        log.warning("CCD lookup failed for %s (%s)", resname, e)                # Log a warning message with the ligand name and the specific error
        return None                                                             # Return None so the pipeline knows the fetch failed and can fallback


def _extract_ligand(pdb_path: Path, resname: str, out_sdf: Path) -> tuple[Optional[Path], Optional[str], Optional[tuple]]:
    """
    Parses a raw PDB file, extracts the 3D coordinates of the specified ligand, 
    repairs its chemical bond orders using the CCD template, and saves it as an SDF.

    Args:
        pdb_path (Path): Path to the raw, downloaded PDB file.
        resname (str): The 3-letter HETATM code of the target ligand.
        out_sdf (Path): The intended output file path for the repaired ligand.

    Returns:
        tuple: A 3-tuple containing (Path to saved SDF, SMILES string, (X,Y,Z) centroid).
               Returns (None, None, None) if extraction fails.

    Example:
        >>> _extract_ligand(Path("raw/6oim.pdb"), "6O7", Path("processed/6O7.sdf"))
        (PosixPath('processed/6O7.sdf'), 'C=CC(=O)N1CCN...', (-15.2, 12.1, 4.3))
    """
    lines = [ln for ln in pdb_path.read_text().splitlines()                     # Read the entire PDB file into memory and split it line by line
             if ln.startswith(("HETATM", "ATOM")) and ln[17:20].strip() == resname.upper()] # Filter for lines belonging to the specific ligand residue
    if not lines:                                                               # Check if the list comprehension came up empty
        log.warning("Ligand %s not found in %s", resname, pdb_path.name)        # Log a warning that the requested ligand doesn't exist in this PDB
        return None, None, None                                                 # Abort extraction and return nulls

    block = "\n".join(lines) + "\nEND\n"                                        # Reconstruct the extracted lines into a valid PDB-formatted string block
    raw = Chem.MolFromPDBBlock(block, sanitize=False, removeHs=False)           # Ask RDKit to parse the block into a molecule (bypassing strict sanitization)
    if raw is None:                                                             # Check if RDKit completely failed to parse the 3D coordinates
        log.warning("RDKit could not parse ligand block for %s", resname)       # Log a warning that the geometric data is corrupted or unreadable
        return None, None, None                                                 # Abort extraction and return nulls

    # A deposited HETATM group can contain several disconnected copies of the ligand (alternate conformers, multiple chains/occupancies). 
    # Keeping all of them yields a multi-fragment "molecule" (e.g. 4x or 2x the real atom count) that breaks both docking and symmetry-aware RMSD. 
    # Therefore we split into disconnected fragments and keep only the largest single copy — that is the one true ligand instance.
    frags = Chem.GetMolFrags(raw, asMols=True, sanitizeFrags=False)             # Decompose the parsed block into its disconnected molecular fragments
    if len(frags) > 1:                                                          # If more than one copy/fragment is present...
        raw = max(frags, key=lambda m: m.GetNumAtoms())                         # ...keep the largest, i.e. one complete copy of the ligand
        log.info("Ligand %s: %d fragments in block, kept largest (%d atoms)",   # Record that we collapsed a multi-copy block to a single instance
                 resname, len(frags), raw.GetNumAtoms())                        # (this is the durable fix for the multi-conformer reference bug)

    coords = raw.GetConformer().GetPositions()                                  # Extract the raw (X, Y, Z) numerical coordinate matrix for all atoms
    centroid = tuple(np.round(coords.mean(axis=0), 3))                          # Calculate the geometric center (mean of coordinates) and round to 3 decimals

    # Bond orders must come from the CCD template. Crystal coordinates carry no bond-order information, so RDKit's distance-based perception 
    # produces chemically wrong molecules (saturated rings, lost aromaticity) that silently corrupt docking, parameterization and RMSD. 
    # We therefore make templating authoritative: if it fails, fail loudly and return nulls rather than writing a plausible-looking but incorrect structure to disk.
    template_smi = _ccd_smiles(resname)                                         # Call the helper function to fetch the ideal 1D chemical topology
    if not template_smi:                                                        # If the network fetch for the canonical SMILES failed...
        log.error("No CCD SMILES for %s; refusing to write distance-perceived " # ...refuse to fall back to guessed bonds
                  "chemistry. Check network / resname.", resname)               # Tell the user exactly why extraction was aborted
        return None, None, None                                                 # Abort: a wrong reference is worse than a missing one
    template = Chem.MolFromSmiles(template_smi)                                 # Convert the ideal 1D SMILES string into an RDKit template molecule
    if template is None:                                                        # If even the canonical SMILES could not be parsed...
        log.error("CCD SMILES for %s did not parse (%s)", resname, template_smi)  # ...log it explicitly
        return None, None, None                                                 # Abort rather than guess
    try:                                                                        # Begin error-handling for the graph-matching templating step
        mol = AllChem.AssignBondOrdersFromTemplate(template, raw)               # Map the double/aromatic bonds from the 1D template onto the 3D coordinates
        smiles = Chem.MolToSmiles(Chem.RemoveHs(mol))                           # Strip hydrogens and generate a clean canonical SMILES from the repaired molecule
    except Exception as e:  # noqa: BLE001                                      # Catch algorithmic failures (e.g., coordinates don't match the template graph)
        log.error("Bond-order templating failed for %s (%s); skipping ligand "  # Fail loud: do not silently fall back to perceived bonds
                  "(do not trust distance-perceived bonds).", resname, e)       # Explain the deliberate refusal
        return None, None, None                                                 # Abort extraction for this ligand

    ensure_dir(out_sdf.parent)                                                  # Ensure the target output directory exists on the filesystem before writing
    writer = Chem.SDWriter(str(out_sdf))                                        # Initialize the RDKit Spatial Data File (SDF) writer engine
    mol.SetProp("_Name", resname)                                               # Inject the ligand's 3-letter code as the internal name property of the molecule
    writer.write(mol)                                                           # Execute the write operation, serializing the 3D molecule to disk
    writer.close()                                                              # Close the file handler to prevent memory leaks
    log.info("Ligand %s -> %s (smiles=%s)", resname, out_sdf.name, smiles)      # Log successful extraction, including the resolved SMILES string
    return out_sdf, smiles, centroid                                            # Return the filepath, the chemical string, and the pocket center coordinates


def _write_receptor(pdb_path: Path, chain: str, out_pdb: Path) -> Path:
    """
    Cleans a raw PDB file by stripping out water molecules, crystallization salts, 
    unwanted protein chains, and ligands, leaving only the bare protein scaffold.

    Args:
        pdb_path (Path): Path to the raw, multi-chain PDB file.
        chain (str): The specific protein chain identifier to isolate (e.g., "A").
        out_pdb (Path): The intended output file path for the cleaned receptor.

    Returns:
        Path: The filepath to the saved, cleaned receptor PDB.

    Example:
        >>> _write_receptor(Path("raw/6oim.pdb"), "A", Path("processed/receptor.pdb"))
        PosixPath('processed/receptor.pdb')
    """
    keep = []                                                                   # Initialize an empty list to store the lines of text we want to keep
    for ln in pdb_path.read_text().splitlines():                                # Read the raw PDB file into memory and iterate through it line by line
        if ln.startswith("ATOM") and ln[21] == chain and ln[17:20].strip() in _AA3: # Keep lines that are standard atoms, belong to the target chain, and are valid amino acids
            keep.append(ln)                                                     # Add the validated protein atom line to our keep list
        elif ln.startswith("TER") and (len(ln) <= 21 or ln[21] == chain):       # Keep chain termination records to ensure valid PDB syntax
            keep.append(ln)                                                     # Add the termination line to our keep list
    ensure_dir(out_pdb.parent)                                                  # Ensure the target output directory exists on the filesystem before writing
    out_pdb.write_text("\n".join(keep) + "\nEND\n")                             # Join the kept lines with newlines, append an END tag, and save to disk
    log.info("Receptor (chain %s) -> %s (%d atom records)", chain, out_pdb.name, len(keep)) # Log the successful receptor creation and the atom count
    return out_pdb                                                              # Return the filepath to the cleaned protein receptor


def prepare(pdb_id: str, chain: str, ligand_resname: str, work_dir) -> PreparedTarget:
    """
    The main orchestrator function for the target preparation stage. It downloads 
    the raw data, coordinates the cleanup of the receptor, orchestrates the 
    extraction/repair of the ligand, and packages the results.

    Args:
        pdb_id (str): The 4-letter RCSB PDB identifier.
        chain (str): The protein chain to isolate.
        ligand_resname (str): The 3-letter code of the bound drug.
        work_dir (str/Path): The root directory for saving data.

    Returns:
        PreparedTarget: A dataclass containing paths to the processed files and metadata.

    Example:
        >>> prepare("6OIM", "A", "6O7", "./data")
        PreparedTarget(pdb_id='6OIM', receptor_pdb=..., ligand_sdf=..., ...)
    """
    work_dir = Path(work_dir).resolve()                                         # Absolute path: DiffDock/Boltz run with a different cwd, so relative paths break
    raw = fetch_pdb(pdb_id, work_dir / "raw")                                   # Call utility function to download the PDB file and save it to the 'raw' folder
    receptor = _write_receptor(raw, chain, work_dir / "processed" / f"{pdb_id}_receptor.pdb") # Clean the protein and save it to the 'processed' folder
    sdf, smiles, center = _extract_ligand(                                      # Call the complex ligand extraction function and unpack its three return values
        raw, ligand_resname, work_dir / "processed" / f"{pdb_id}_{ligand_resname}.sdf") # Define the output path for the repaired 3D ligand file
    return PreparedTarget(pdb_id, receptor, sdf, smiles, center)                # Construct and return the final data structure containing all processed assets