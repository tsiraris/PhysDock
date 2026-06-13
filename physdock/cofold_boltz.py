"""Tier B: co-folding + binding-affinity with Boltz-2.

PhysDock utilizes a two-tier generative architecture. While Tier A (DiffDock) treats the 
protein as a rigid scaffold to rapidly sample ligand poses, Tier B (this script) deploys 
Boltz-2. Boltz-2 is an AlphaFold3-class *diffusion* model that jointly folds a protein and
a small molecule into a single complex from scratch. 

This approach captures "induced fit" (how the protein morphs to accommodate the drug), 
acting as a highly advanced surrogate for structural conformational sampling. Uniquely, 
Boltz-2 also emits a thermodynamic binding-affinity prediction directly from the structural 
generation process.

This squarely hits two Pierre Fabre JD bullets at once:
  * 'state-of-the-art diffusion models to characterize the interactions between
     proteins, nucleic acids, and small molecules'
  * accelerating affinity assessment without a docking box.

Running several diffusion samples per complex gives a SECOND, structure-native
conformational ensemble to compare against DiffDock's pose set (Triangulation).

VRAM note: KRAS (~169 aa) + a drug-like ligand fits comfortably on a 24 GB A10G.
Larger targets may need --subsample_msa / fewer recycles, or will OOM (Out Of Memory).
"""
from __future__ import annotations                                          # Enable modern type hinting features in older Python versions
import json                                                                 # Standard library for parsing and generating JSON data
import subprocess                                                           # Standard library to spawn new OS-level processes (for the CLI)
from pathlib import Path                                                    # Object-oriented filesystem path manipulation
from typing import Optional                                                 # Type hint for variables that can be of a specific type or None

import pandas as pd                                                         # Third-party library for powerful data manipulation (DataFrames)

from .utils import ensure_dir, get_logger, load_json                        # Import internal project utility functions

log = get_logger("physdock.boltz")                                          # Initialize the dedicated logger for the Boltz module


def write_input_yaml(ligand_id: str, protein_seq: str, ligand_smiles: str,
                     out_yaml: Path, predict_affinity: bool = True) -> Path:
    """
    Generates the strict YAML configuration file required by the Boltz-2 CLI.
    
    Boltz requires inputs in a specific schema: one sequence block for the protein chain 
    and one for the ligand chain (represented by its SMILES string). If affinity prediction 
    is requested, it appends a 'properties' block binding the affinity calculation to the 
    specific ligand ID.
    
    Args:
        ligand_id (str): The human-readable identifier for the ligand.
        protein_seq (str): The 1D amino acid sequence of the target protein.
        ligand_smiles (str): The 1D chemical string representation of the ligand.
        out_yaml (Path): The designated file path to save the generated YAML.
        predict_affinity (bool): Flag to enable/disable the affinity prediction head.
        
    Returns:
        Path: The file path to the successfully written YAML file.
        
    Example:
        >>> write_input_yaml("sotorasib", "MTEYKLVVVGAC...", "CC(C)C1=CC...", Path("input.yaml"))
        Generates a YAML file instructing Boltz to fold chain A (protein) with 
        chain B (ligand) and predict how tightly B binds to A.
    """
    ensure_dir(out_yaml.parent)                                             # Create the parent directory for the YAML file if it doesn't exist
    lines = [                                                               # Initialize a list to hold the lines of the YAML file
        "version: 1",                                                       # Declare the Boltz schema version
        "sequences:",                                                       # Begin the sequences block defining the biological entities
        "  - protein:",                                                     # Define the first entity as a protein
        "      id: A",                                                      # Assign the protein to chain identifier 'A'
        f"      sequence: {protein_seq}",                                   # Inject the actual amino acid string into the YAML
        "  - ligand:",                                                      # Define the second entity as a small molecule ligand
        "      id: B",                                                      # Assign the ligand to chain identifier 'B'
        f"      smiles: '{ligand_smiles}'",                                 # Inject the actual chemical SMILES string into the YAML
    ]                                                                       # Close the list initialization
    if predict_affinity:                                                    # Check if the pipeline configuration requested affinity scoring
        lines += ["properties:", "  - affinity:", "      binder: B"]        # Append the properties block to tell Boltz to score chain B
    out_yaml.write_text("\n".join(lines) + "\n")                            # Combine the list into a single string and write it to disk
    log.info("Boltz input -> %s", out_yaml)                                 # Log the successful creation and location of the YAML file
    return out_yaml                                                         # Return the path object for the next stage in the pipeline


def run(cfg, input_yaml: Path, out_dir: Path) -> None:
    """
    Orchestrates the execution of the Boltz-2 diffusion model via system subprocess.
    
    This function acts as the bridge between the Python pipeline and the underlying 
    deep learning CLI. It constructs the execution command, passes in hardware and 
    sampling constraints (like diffusion_samples to create conformational ensembles), 
    and handles standard error streams to catch out-of-memory (OOM) failures.
    
    Args:
        cfg (Config): The master pipeline configuration object (from the YAML).
        input_yaml (Path): The path to the target definition YAML created previously.
        out_dir (Path): The designated directory for Boltz to dump its structural outputs.
        
    Returns:
        None (Outputs are written directly to the filesystem).
        
    Example:
        >>> run(cfg, Path("input.yaml"), Path("results/"))
        Executes: `boltz predict input.yaml --out_dir results/ --diffusion_samples 5 ...`
        Will raise a RuntimeError with a tailored message if the GPU runs out of VRAM.
    """
    out_dir = ensure_dir(out_dir)                                           # Ensure the output directory exists so the CLI doesn't crash
    cmd = ["boltz", "predict", str(input_yaml),                             # Begin building the CLI command list, starting with the base invocation
           "--out_dir", str(out_dir),                                       # Pass the output directory argument to the CLI
           "--diffusion_samples", str(cfg.get("boltz", "diffusion_samples", default=5)),  # Inject the multi-seed sampling count (creates the ensemble)
           "--recycling_steps", str(cfg.get("boltz", "recycling_steps", default=3)),      # Inject the neural network recycling iterations (refinement)
           "--output_format", "pdb"]                                        # Force the structural output format to standard PDB
    if cfg.get("boltz", "use_msa_server", default=True):                    # Check if the config dictates using cloud MSA over local compute
        cmd.append("--use_msa_server")                                      # Append the cloud MSA flag to save local CPU/Storage resources
    log.info("Boltz cmd:\n  %s", " ".join(cmd))                             # Log the exact CLI command being executed for reproducibility
    proc = subprocess.run(cmd, capture_output=True, text=True)              # Execute the command, capturing stdout and stderr as text
    if proc.returncode != 0:                                                # Check if the process exited with an error code (anything but 0)
        log.error("Boltz stderr tail:\n%s", proc.stderr[-2000:])            # Log the last 2000 characters of the error stream for debugging
        raise RuntimeError(f"boltz predict exited {proc.returncode}. "      # Raise a fatal error in the pipeline
                           "Check VRAM (try fewer diffusion_samples) and that " # Provide actionable advice regarding GPU memory limits
                           "`boltz` is installed in this env.")             # Remind the user about the environment dependencies
    log.info("Boltz finished -> %s", out_dir)                               # Log successful execution and point to the results


def parse_results(out_dir: Path) -> pd.DataFrame:
    """
    Harvests structural and thermodynamic confidence metrics from Boltz-2 outputs.
    
    Boltz-2 generates deep folder structures containing raw PDBs and multiple JSON 
    files per diffusion sample. This function crawls the output directory, maps 
    structural files to their respective confidence JSONs, extracts key metrics 
    (pTM, ipTM, pLDDT), and securely attempts to extract affinity predictions.
    
    Args:
        out_dir (Path): The root directory where Boltz saved all predictions.
        
    Returns:
        pd.DataFrame: A structured, tabular dataset of all complexes and their scores.
        
    Example:
        >>> parse_results(Path("results/"))
        Returns a DataFrame with columns like:
        ligand_id | complex_pdb       | ptm  | iptm | affinity_pred_value
        sotorasib | sotorasib_001.pdb | 0.85 | 0.72 | 8.4
    """
    rows = []                                                               # Initialize an empty list to hold data dictionaries for each complex
    pred_root = Path(out_dir)                                               # Convert the output directory string to a Path object
    candidates = list(pred_root.rglob("confidence_*.json"))                 # Recursively search the directory for all files matching the confidence pattern
    for conf_path in candidates:                                            # Iterate through every confidence JSON found
        comp = conf_path.parent.name                                        # Extract the complex's name from its parent directory name
        conf = load_json(conf_path)                                         # Load the confidence JSON contents into a Python dictionary
        row = {                                                             # Begin constructing the data row for this specific complex prediction
            "ligand_id": comp,                                              # Record the identifier
            "complex_pdb": _sibling(conf_path, ".pdb"),                     # Find and record the absolute path to the associated 3D PDB structure
            "ptm": conf.get("ptm"),                                         # Extract the Predicted Template Modeling score (global structural confidence)
            "iptm": conf.get("iptm"),                                       # Extract the Interface pTM score (confidence in the protein-ligand contact area)
            "complex_plddt": conf.get("complex_plddt"),                     # Extract the overall local distance difference test score
            "confidence_score": conf.get("confidence_score"),               # Extract Boltz's proprietary aggregate confidence metric
        }                                                                   # Close the dictionary initialization
        aff = _find_affinity(conf_path.parent)                              # Attempt to find and load the corresponding affinity JSON in the same folder
        if aff:                                                             # If the affinity JSON was successfully found and parsed
            # affinity_pred_value: Boltz reports log10(IC50/uM)-style scale;
            # affinity_probability_binary: P(binder). We keep both, raw.    
            row["affinity_pred_value"] = aff.get("affinity_pred_value")     # Extract the continuous thermodynamic binding score
            row["affinity_prob_binary"] = aff.get("affinity_probability_binary")  # Extract the binary probability score of whether it binds at all
        rows.append(row)                                                    # Append the fully constructed row to the master list
    df = pd.DataFrame(rows)                                                 # Convert the list of dictionaries into a pandas DataFrame for easy analysis
    if df.empty:                                                            # Check if the DataFrame is empty (no JSONs were found)
        log.warning("No Boltz confidence JSONs found under %s", out_dir)    # Warn the user that parsing failed silently due to lack of data
    else:                                                                   # If data was successfully parsed
        log.info("Parsed Boltz results for %d complexes", len(df))          # Log the total number of complexes successfully processed
    return df                                                               # Return the DataFrame to the pipeline


def _sibling(json_path: Path, suffix: str) -> Optional[str]:
    """
    Locates a sibling file with a specific extension in the same directory.
    
    A lightweight utility to associate the JSON metadata files with their 
    corresponding physical 3D structural files (e.g., .pdb) without relying 
    on strict filename parsing, which can break across version updates.
    
    Args:
        json_path (Path): The filepath of the reference file.
        suffix (str): The file extension to search for (e.g., ".pdb").
        
    Returns:
        Optional[str]: The absolute path to the sibling file, or None if missing.
        
    Example:
        >>> _sibling(Path("output/confidence_1.json"), ".pdb")
        '/absolute/path/to/output/complex_1.pdb'
    """
    for p in json_path.parent.glob(f"*{suffix}"):                           # Iterate over all files in the parent directory that end with the target suffix
        return str(p)                                                       # Return the absolute string path of the first match found
    return None                                                             # Return None if the loop completes without finding a matching file


def _find_affinity(folder: Path) -> Optional[dict]:
    """
    Safely locates and parses the affinity prediction JSON file.
    
    Because affinity prediction is an optional flag in the generative model, 
    this file may not exist. This function uses a defensive try/except block 
    to handle missing or corrupted files gracefully without crashing the pipeline.
    
    Args:
        folder (Path): The directory to search for the affinity JSON.
        
    Returns:
        Optional[dict]: The parsed JSON data as a dictionary, or None if missing/corrupt.
        
    Example:
        >>> _find_affinity(Path("results/sotorasib/"))
        {'affinity_pred_value': 8.4, 'affinity_probability_binary': 0.99}
    """
    for p in folder.glob("affinity_*.json"):                                # Iterate over all files in the folder that match the affinity JSON naming pattern
        try:                                                                # Open a defensive block to catch file read or JSON parsing errors
            return json.loads(p.read_text())                                # Read the file text, parse it into a dictionary, and return it immediately
        except Exception:  # noqa: BLE001                                   # Catch any error (BLE001 suppresses a linter warning about broad exceptions)
            return None                                                     # Return None if the file was unreadable or the JSON was malformed
    return None                                                             # Return None if the loop completes without finding a file matching the pattern