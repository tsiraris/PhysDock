"""Cheminformatics gate.

This script integrates physical constraints such as synthetic accessibility,
valency, and physicochemical properties into the Machine Learning framework. 
In the context of the PhysDock pipeline, it acts as a critical *pre-flight filter*.

Deep Learning models (like DiffDock or Boltz-2) are computationally expensive and 
will happily attempt to fold or dock molecules that defy the laws of chemistry. 
By running this CPU-bound script first, we save expensive GPU hours from being 
wasted on ligands that are chemically implausible, infinitely strained, or 
theoretically impossible for a human chemist to synthesize in a real lab.

Every ligand gets:
  * RDKit sanitization (kills bad valences and impossible structures immediately).
  * Property windows (MW, cLogP, TPSA, HBD/HBA, rotatable bonds) -- soft, scored.
  * Synthetic accessibility (SA score, Ertl & Schuffenhauer 1-10; lower=easier).
  * PAINS / structural-alert flags (warns of compounds that break assay tests).

The gate returns a standardized per-ligand verdict (ChemVerdict) so downstream 
stages can conditionally drop the molecule or down-rank its priority.
"""

from __future__ import annotations                                     
from dataclasses import dataclass, asdict                              
from typing import Optional                                            

from rdkit import Chem                                                 
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors          
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams 

from .utils import get_logger                                          

log = get_logger("physdock.chem")                                      

# Desirability windows (oral small-molecule oncology-ish; deliberately permissive because covalent KRAS binders are large/greasy and would fail strict Ro5).
WINDOWS = {                                                            # Dictionary defining the acceptable min/max ranges for chemical properties
    "mw":   (150.0, 750.0),                                            # Molecular Weight: KRAS covalent drugs are often large, hence the 750 max
    "logp": (-1.0, 6.5),                                               # Lipophilicity (greasiness): KRAS pocket is hydrophobic, allowing up to 6.5
    "tpsa": (20.0, 180.0),                                             # Topological Polar Surface Area: Cell permeability indicator
    "hbd":  (0, 5),                                                    # Hydrogen Bond Donors: Keeps molecule from being too polar
    "hba":  (0, 12),                                                   # Hydrogen Bond Acceptors: Higher allowance for complex oncology drugs
    "rotb": (0, 12),                                                   # Rotatable Bonds: Limits excessive flexibility (entropy penalty upon binding)
}

_PAINS_CATALOG: Optional[FilterCatalog] = None                         # Global variable to cache the PAINS catalog so it only loads into memory once


def _pains_catalog() -> FilterCatalog:
    """
    Initializes and caches the PAINS (Pan-Assay Interference Compounds) filter catalog.
    
    Loads RDKit's internal database of PAINS structures. It uses a singleton pattern 
    (caching via a global variable) to ensure the heavy catalog is only loaded once 
    per runtime, saving memory and processing time.
    
    Checks if the global `_PAINS_CATALOG` is None. If true, it builds the catalog 
    using RDKit's `FilterCatalogParams` set to the PAINS preset.
    
    Args:
        None
        
    Returns:
        FilterCatalog: The compiled RDKit catalog containing PAINS structural alerts.
        
    Example:
    >>> catalog = _pains_catalog()
    >>> type(catalog)
    <class 'rdkit.Chem.rdfiltercatalog.FilterCatalog'>
    """
    global _PAINS_CATALOG                                              # Declares intent to modify the module-level _PAINS_CATALOG variable
    if _PAINS_CATALOG is None:                                         # Checks if the catalog hasn't been built yet (Singleton pattern)
        params = FilterCatalogParams()                                 # Initializes an empty parameter object for the RDKit filter catalog
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)    # Instructs the parameters to include the standard PAINS alerts library
        _PAINS_CATALOG = FilterCatalog(params)                         # Builds the actual catalog object using the specified parameters and caches it
    return _PAINS_CATALOG                                              # Returns the loaded catalog for use in screening


def _sa_score(mol: Chem.Mol) -> Optional[float]:
    """
    Calculates the Synthetic Accessibility (SA) score of a molecule.
    
    Estimates how difficult it would be for a human chemist to synthesize the 
    molecule in a lab. Scores range from 1 (very easy) to 10 (virtually impossible).
    
    Dynamically attempts to import RDKit's contributed `sascorer` module. If the 
    module is missing (common in stripped-down environments), it catches the error 
    and returns None rather than crashing the entire pipeline.
    
    Args:
        mol (Chem.Mol): The RDKit molecule object to be evaluated.
        
    Returns:
        Optional[float]: The calculated SA score, or None if the sascorer module is missing.
        
    Example:
    >>> aspirin = Chem.MolFromSmiles('CC(=O)OC1=CC=CC=C1C(=O)O')
    >>> _sa_score(aspirin)
    1.42  # Very easy to synthesize
    """
    try:                                                                # Starts a try-catch block to gracefully handle missing dependencies
        from rdkit.Chem import RDConfig                                 # Imports RDKit's configuration module to locate the contrib directory
        import os, sys                                                  # Imports operating system utilities to manipulate file paths
        sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))    # Appends the specific SA_Score directory to Python's import search path
        import sascorer  # type: ignore                                 # Imports the sascorer script (type ignore suppresses linter warnings)
        return float(sascorer.calculateScore(mol))                      # Calculates the SA score for the molecule and returns it as a float
    except Exception as e:  # noqa: BLE001                              # Catches any error during import or calculation (suppressing broad-except lint)
        log.warning("SA score unavailable (%s); skipping SA term.", e)      # Logs a warning message so the user knows the metric is missing, without crashing
        return None                                                     # Returns None so the downstream code knows the score was uncalculable


@dataclass
class ChemVerdict:
    """
    A structured data container representing the final chemistry evaluation of a ligand.
    
    Stores all calculated physicochemical properties, flags, and the final pass/fail 
    verdict for a single molecule in a standardized, immutable-like format.
    
    Uses Python's `@dataclass` decorator to automatically generate `__init__` and 
    string representation methods, keeping the code clean.
    
    Example:
    >>> verdict = ChemVerdict(ligand_id="test1", smiles="C", valid=True, passes=True)
    >>> verdict.to_row()
    {'ligand_id': 'test1', 'smiles': 'C', 'valid': True, ...}
    """
    ligand_id: str                                                     # The human-readable or database identifier for the drug (e.g., "sotorasib")
    smiles: Optional[str]                                              # The 1D chemical string representation (None if unparsable)
    valid: bool                                                        # Boolean indicating if RDKit could successfully parse and sanitize the molecule
    mw: Optional[float] = None                                         # Molecular Weight (defaults to None if invalid)
    logp: Optional[float] = None                                       # Calculated LogP (lipophilicity)
    tpsa: Optional[float] = None                                       # Topological Polar Surface Area
    hbd: Optional[int] = None                                          # Count of Hydrogen Bond Donors
    hba: Optional[int] = None                                          # Count of Hydrogen Bond Acceptors
    rotb: Optional[int] = None                                         # Count of Rotatable Bonds
    sa_score: Optional[float] = None                                   # Synthetic Accessibility Score (1 to 10)
    pains_alerts: int = 0                                              # Number of structural red flags found in the molecule
    in_windows: bool = False                                           # Boolean indicating if ALL properties fell within the defined WINDOWS
    passes: bool = False                                               # Final verdict boolean on whether the molecule is greenlit for downstream ML
    reason: str = ""                                                   # Text explanation detailing exactly why a molecule was flagged or failed

    def to_row(self) -> dict:
        """
        Converts the dataclass instance into a dictionary.
        
        Allows the structured verdict data to be easily exported to CSVs or Pandas 
        DataFrames by downstream logging scripts.
        
        Args:
            None
            
        Returns:
            dict: A dictionary representation of all fields within the dataclass.
        """
        return asdict(self)                                            # Uses the dataclass built-in asdict function to convert properties to a dict


def _within(val, lo, hi) -> bool:
    """
    Helper function to check if a numerical value falls within a specified range.
    
    Safely checks range boundaries while handling cases where the value might be None.
    
    Returns False immediately if the value is None, otherwise evaluates `lo <= val <= hi`.
    
    Args:
        val (int | float | None): The numerical value to check.
        lo (int | float): The lower bound of the range.
        hi (int | float): The upper bound of the range.
        
    Returns:
        bool: True if the value is within the inclusive range, False otherwise (or if val is None).
        
    Example:
    >>> _within(500, 150, 750)
    True
    """
    return val is not None and lo <= val <= hi                         # Ensures value exists, then performs a standard inclusive range check


def evaluate_ligand(ligand_id: str, smiles: str) -> ChemVerdict:
    """
    The main execution function that runs the complete cheminformatics gate on a molecule.
    
    Attempts to parse the SMILES string into a 3D molecule, calculates all required 
    physicochemical properties, checks for PAINS alerts, calculates synthetic 
    accessibility, and compiles a final `ChemVerdict`.
    
    Uses RDKit to build the molecule. If RDKit fails (due to bad valency), it immediately 
    returns a failed verdict. Otherwise, it calculates descriptors, compares them against 
    the predefined `WINDOWS`, and compiles any warnings into the `reason` string.
    
    Args:
        ligand_id (str): The identifier label for the molecule.
        smiles (str): The chemical SMILES string to be evaluated.
        
    Returns:
        ChemVerdict: A fully populated dataclass containing all calculated metrics and the pass/fail flags.
        
    Example:
    >>> evaluate_ligand("benzene", "C1=CC=CC=C1")
    ChemVerdict(ligand_id='benzene', valid=True, mw=78.11, in_windows=False, passes=True, reason='out_of_property_window')
    """
    mol = Chem.MolFromSmiles(smiles) if smiles else None               # Attempts to convert the SMILES text into an RDKit Mol object (returns None on failure)
    if mol is None:                                                    # Catches cases where the SMILES was empty, malformed, or chemically impossible
        return ChemVerdict(ligand_id, smiles, valid=False, reason="unparsable_or_bad_valence") # Early exit: returns a failed verdict saving compute time

    mw = Descriptors.MolWt(mol)                                        # Calculates the exact Molecular Weight
    logp = Crippen.MolLogP(mol)                                        # Calculates the partition coefficient (greasiness)
    tpsa = rdMolDescriptors.CalcTPSA(mol)                              # Calculates the Topological Polar Surface Area
    hbd = rdMolDescriptors.CalcNumHBD(mol)                             # Counts specific atoms capable of donating hydrogen bonds
    hba = rdMolDescriptors.CalcNumHBA(mol)                             # Counts specific atoms capable of accepting hydrogen bonds
    rotb = rdMolDescriptors.CalcNumRotatableBonds(mol)                 # Counts single, non-ring bonds that allow the molecule to flex
    sa = _sa_score(mol)                                                # Calls the helper function to estimate synthesis difficulty
    pains = len(_pains_catalog().GetMatches(mol))                      # Screens the molecule against the PAINS catalog and counts any matches

    in_windows = all([                                                 # Evaluates to True ONLY if every single property check below returns True
        _within(mw, *WINDOWS["mw"]),                                   # Unpacks the MW tuple (150, 750) and checks if the molecule's MW fits
        _within(logp, *WINDOWS["logp"]),                               # Checks if LogP is within acceptable limits
        _within(tpsa, *WINDOWS["tpsa"]),                               # Checks if TPSA is within acceptable limits
        _within(hbd, *WINDOWS["hbd"]),                                 # Checks if Hydrogen Bond Donors are within limits
        _within(hba, *WINDOWS["hba"]),                                 # Checks if Hydrogen Bond Acceptors are within limits
        _within(rotb, *WINDOWS["rotb"]),                               # Checks if Rotatable Bonds are within limits
    ])

    # Gate policy: must be valid; PAINS is a warning, not a hard fail for known clinical compounds, but we record it. 
    # Out-of-window is allowed through but flagged, because real KRAS covalent drugs sit at the edges of these windows.
    passes = True                                                      # Initializes the baseline assumption: if it parsed successfully, it passes to the GPU
    reasons = []                                                       # Initializes an empty list to collect any warnings or yellow flags
    if pains > 0:                                                      # Checks if any PAINS alerts were triggered
        reasons.append(f"pains={pains}")                               # Appends a warning to the log rather than failing the molecule
    if not in_windows:                                                 # Checks if the molecule violated the ideal property windows
        reasons.append("out_of_property_window")                       # Appends an out-of-window warning to the log
    if sa is not None and sa > 6.0:                                    # Checks if the synthesis score is alarmingly high (closer to 10 means impossible)
        reasons.append(f"hard_to_synthesize(SA={sa:.1f})")             # Appends a synthetic difficulty warning to the log

    return ChemVerdict(                                                # Constructs the final standardized data object to return to the pipeline
        ligand_id=ligand_id, smiles=Chem.MolToSmiles(mol), valid=True, # Passes basic identifiers (regenerating standard SMILES from the mol object)
        mw=round(mw, 2), logp=round(logp, 2), tpsa=round(tpsa, 2),     # Passes calculated properties rounded to 2 decimal places for clean logging
        hbd=hbd, hba=hba, rotb=rotb,                                   # Passes integer-based property counts
        sa_score=round(sa, 2) if sa is not None else None,             # Passes the rounded SA score (or None if the contrib module was missing)
        pains_alerts=pains, in_windows=in_windows, passes=passes,      # Passes the boolean flags and alert counts
        reason=";".join(reasons) if reasons else "ok",                 # Joins all warnings with a semicolon, or outputs "ok" if the list is empty
    )

def passing_ligand_ids(gate_csv: str = "results/chem/chem_gate.csv") -> Optional[set]:
    """
    Reads the Stage-02 chem-gate ledger and returns the set of ligand IDs that PASSED.

    This is what makes the cheminformatics gate a real filter rather than an advisory
    artifact: Stages 03/04 call this and skip any ligand whose `passes` flag is False
    (i.e. chemically invalid / unparseable). PAINS, out-of-window, and high-SA molecules
    are deliberately kept (passes=True) because real covalent KRAS binders trip those
    soft alerts; they are recorded in `reason` for prioritisation, not dropped.

    Args:
        gate_csv (str): Path to the chem-gate CSV written by 02_chem_gate.py.

    Returns:
        Optional[set]: A set of passing ligand_id strings, or None if the gate has not
                       been run yet (in which case callers should not filter).
    """
    import pandas as pd                                                 # Local import keeps module load lightweight.
    from pathlib import Path                                            # Local import for the existence check.

    p = Path(gate_csv)                                                  # Resolve the ledger path.
    if not p.exists() or p.stat().st_size == 0:                         # Gate never ran, or wrote an empty ledger...
        return None                                                     # ...signal "do not filter" so the pipeline still works.
    try:                                                                # An empty/headerless file raises EmptyDataError...
        df = pd.read_csv(p)                                             # Load the per-ligand verdicts.
    except pd.errors.EmptyDataError:                                    # ...treat that the same as "gate not run".
        return None                                                     # Fail open rather than crashing the GPU stage.
    if "passes" not in df.columns or "ligand_id" not in df.columns:     # Malformed/empty ledger...
        return None                                                     # ...fail open rather than dropping everything.
    mask = df["passes"].astype(str).str.strip().str.lower().isin(["true", "1"])  # Robustly coerce the bool column (handles "True"/True/1).
    return set(df.loc[mask, "ligand_id"].astype(str))                   # Return the IDs cleared for the GPU stages.
