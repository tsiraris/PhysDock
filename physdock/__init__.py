"""
=============================================================================
PhysDock: Physics-Aware ML for Biomolecular Interactions
=============================================================================
This initialization file formally registers the `physdock` directory as a 
modular Python package. It serves as the core computational library powering 
the drug discovery pipeline.

By encapsulating the dense scientific logic (cheminformatics, diffusion 
inference, and thermodynamic relaxation) into this localized package, the 
upstream execution scripts can remain clean, decoupled, and easily orchestrable.

Architectural Mapping (Pipeline Scripts -> Core Library Modules):
  Stage 01 (Target Prep)       -> receptor.py
  Stage 02 (Cheminformatics)   -> chem.py
  Stage 03 (Diffusion Dock)    -> docking_diffdock.py  [Tier A: Geometry]
  Stage 04 (Co-Folding)        -> cofold_boltz.py      [Tier B: Affinity]
  Stage 05 (Thermodynamics)    -> physics_openmm.py
  Stage 06 (MD Surrogate)      -> ensemble.py
  Stage 07 (Validation)        -> evaluate.py, report.py
"""
__version__ = "0.1.0"                                                                    # Define the package's semantic version to ensure strict MLOps reproducibility and tracking