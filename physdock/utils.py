"""
PhysDock: Core Utilities Module (utils.py)

This module is the infrastructure backbone of the PhysDock pipeline. 

It handles "Shared Plumbing":
  1. Observability (Standardized logging instead of basic print statements).
  2. Reproducibility (Global deterministic seed enforcement).
  3. Safe I/O (Directory orchestration and JSON serialization).
  4. External API integration (RCSB PDB fetching).
  5. Cheminformatics standardization (SMILES canonicalization).

"""
from __future__ import annotations
import json
import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"                # Define a strict, readable format string for all terminal and file logs


def get_logger(name: str) -> logging.Logger:
    """
    Initializes and returns a standardized logger for the pipeline.
    
    Ensures every script outputs messages in the exact same format 
    (Timestamp | Level | Module | Message) rather than using messy print() statements.
    
    Configures the Python built-in logging module's root basicConfig 
    and returns a named logger instance.
    
    Args:
        name (str): The namespace for the logger (usually the module name).
        
    Returns:
        logging.Logger: The configured logger object.
        
    Example:
        >>> log = get_logger("physdock.evaluate")
        >>> log.info("Starting evaluation...")
        2026-06-05 19:30:00,000 | INFO    | physdock.evaluate | Starting evaluation...
    """
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)                       # Configure the root logger to catch INFO-level and above, applying the standard format
    return logging.getLogger(name)                                                   # Request and return a specific logger instance tagged with the provided module name


log = get_logger("physdock")                                                         # Initialize a module-level logger immediately so utils.py can log its own actions


def set_seed(seed: int = 42) -> None:
    """
    Enforces deterministic execution across the entire Python environment.
    
    Machine Learning models are inherently stochastic (random). This 
    function forces them to use the same random numbers every time the script runs, 
    ensuring scientific reproducibility.
    
    Sets the seed for Python's built-in `random`, NumPy's random 
    number generator, the Python hash environment variable, and (if installed) 
    PyTorch's CPU and GPU generators.
    
    Args:
        seed (int): The master integer seed to use (defaults to 42).
        
    Returns:
        None
        
    Example:
        >>> set_seed(42)
    """
    random.seed(seed)                                                                # Lock the built-in Python standard library random number generator
    np.random.seed(seed)                                                             # Lock the NumPy mathematical random number generator (crucial for arrays)
    os.environ["PYTHONHASHSEED"] = str(seed)                                         # Lock Python's dictionary/set hashing algorithm to prevent random ordering
    try:  # torch is only present once the heavy models are installed                # Open a try-block because PyTorch might not be installed in the lightweight CPU env
        import torch                                                                 # Attempt to import PyTorch dynamically
        torch.manual_seed(seed)                                                      # Lock PyTorch's CPU random number generator
        if torch.cuda.is_available():                                                # Check if the system has an active Nvidia GPU
            torch.cuda.manual_seed_all(seed)                                         # Lock the random number generators for all connected GPUs
    except ImportError:                                                              # Catch the error if PyTorch is not installed in this environment
        pass                                                                         # Silently continue, as this is expected during the stage 00 setup check
    log.info("Seed set to %d", seed)                                                 # Log the successful seed lock for the scientific reproducibility audit trail


def ensure_dir(p: str | Path) -> Path:
    """
    Safely creates a directory path and all necessary parent directories.
    
    Prevents "FileNotFoundError" crashes when saving results by 
    making sure the destination folder actually exists before writing to it.
    
    Uses pathlib's mkdir with parents=True (acts like `mkdir -p` 
    in Linux) and exist_ok=True (prevents crashing if the folder is already there).
    
    Args:
        p (str | Path): The requested directory path.
        
    Returns:
        Path: The guaranteed-to-exist directory path object.
        
    Example:
        >>> out = ensure_dir("results/run_01/poses")
    """
    p = Path(p)                                                                      # Ensure the incoming string or Path is firmly cast as a pathlib Path object
    p.mkdir(parents=True, exist_ok=True)                                             # Create the folder, build any missing parent folders, and don't error if it exists
    return p                                                                         # Return the validated Path object for immediate use in the calling function


def save_json(obj: Any, path: str | Path) -> None:
    """
    Safely serializes a Python object into a formatted JSON file.
    
    Saves complex nested dictionaries (like pipeline configurations 
    or final metrics) to disk in a human-readable format.
    
    Ensures the target directory exists, opens the file, and uses 
    `json.dump` with a 2-space indent. It crucially uses `default=str` to prevent 
    crashes if the dict contains non-standard objects (like NumPy arrays or Paths).
    
    Args:
        obj (Any): The Python dictionary or list to save.
        path (str | Path): The destination file path.
        
    Returns:
        None
        
    Example:
        >>> save_json({"RMSD": 1.5, "Success": True}, "output.json")
    """
    path = Path(path)                                                                # Ensure the destination path is a pathlib Path object
    ensure_dir(path.parent)                                                          # Call the utility function to build the folder holding the file, if it doesn't exist
    with open(path, "w") as fh:                                                      # Open the file in write ('w') mode using a safe context manager
        json.dump(obj, fh, indent=2, default=str)                                    # Dump the JSON data, pretty-print it (indent=2), and cast weird objects to strings
    log.info("Wrote %s", path)                                                       # Log the successful file write operation


def load_json(path: str | Path) -> Any:
    """
    Reads a JSON file from disk into a Python object.
    
    Loads configuration files or previous stage results into memory.
    
    Opens the file via a context manager and runs `json.load`.
    
    Args:
        path (str | Path): The path to the JSON file.
        
    Returns:
        Any: The parsed Python dictionary or list.
        
    Example:
        >>> data = load_json("config.json")
    """
    with open(path) as fh:                                                           # Open the file in default read ('r') mode using a context manager
        return json.load(fh)                                                         # Parse the JSON text stream into a Python dictionary and return it


def fetch_pdb(pdb_id: str, out_dir: str | Path, fmt: str = "pdb") -> Path:
    """
    Downloads structural biology files directly from the RCSB Protein Data Bank.
    
    Automates the acquisition of raw ground-truth crystal structures.
    It includes a smart-caching mechanism to avoid re-downloading files already present.
    
    Checks if the file exists locally. If not, it constructs the RCSB URL and uses the 
    `requests` library to fetch the data. It will aggressively raise an error if the 
    download fails, intentionally stopping the pipeline.
    
    Args:
        pdb_id (str): The 4-letter alphanumeric PDB code (e.g., '6OIM').
        out_dir (str | Path): The directory to save the downloaded file.
        fmt (str): The file format to request ('pdb' or 'cif'). Defaults to 'pdb'.
        
    Returns:
        Path: The local path to the cached or newly downloaded file.
        
    Example:
        >>> file_path = fetch_pdb("6OIM", "./data/raw")
    """
    import requests                                                                  # Import the requests library locally to keep module-level dependencies lightweight

    pdb_id = pdb_id.lower()                                                          # Force the user-provided PDB ID to lowercase to match RCSB server URL standards
    ext = "pdb" if fmt == "pdb" else "cif"                                           # Determine the file extension based on the requested format
    url = f"https://files.rcsb.org/download/{pdb_id}.{ext}"                          # Construct the exact URL to the RCSB static file server
    out_dir = ensure_dir(out_dir)                                                    # Guarantee the destination download folder exists
    dest = out_dir / f"{pdb_id}.{ext}"                                               # Construct the full, final local file path
    if dest.exists():                                                                # Check if this exact file was already downloaded in a previous run
        log.info("Cached %s", dest)                                                  # Log that we are using a locally cached version to save time/bandwidth
        return dest                                                                  # Immediately return the path without making an internet request
    log.info("Fetching %s", url)                                                     # Log that a fresh internet download is commencing
    r = requests.get(url, timeout=60)                                                # Execute the HTTP GET request, explicitly timing out after 60 seconds to prevent hanging
    r.raise_for_status()                                                             # CRITICAL: If the server returns a 404 Not Found or 500 error, crash loudly right here
    dest.write_bytes(r.content)                                                      # Write the downloaded raw byte content to the local file
    return dest                                                                      # Return the path to the newly saved file


def canonical_smiles(smiles: str) -> str | None:
    """
    Standardizes a chemical SMILES string into its absolute, canonical form.
    
    In chemistry, the same molecule can be written as a SMILES string 
    in dozens of different ways. This function processes any valid string and forces 
    it into one universal, standardized RDKit string representation.
    
    Uses RDKit to parse the input string into a 3D chemical graph 
    object in memory, and then asks RDKit to rewrite that graph back into a SMILES 
    string using its strict canonicalization rules.
    
    Args:
        smiles (str): The raw, unstandardized input SMILES string.
        
    Returns:
        str | None: The canonicalized SMILES string, or None if the input was invalid chemistry.
        
    Example:
        >>> canonical_smiles("C(C)O")  # Unstandardized ethanol
        'CCO'                          # Standardized ethanol
    """
    from rdkit import Chem                                                           # Import the RDKit chemistry engine locally

    m = Chem.MolFromSmiles(smiles)                                                   # Attempt to parse the raw text string into a mathematical chemical graph object
    return Chem.MolToSmiles(m) if m is not None else None                            # If parsing succeeded, generate and return the strictly canonicalized string