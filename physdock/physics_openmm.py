"""
PhysDock: Physics Rescoring Module (physics_openmm.py)
=============================================================================
This module bridges the gap between Machine Learning and Biomolecular Physics.
Generative AI models (like DiffDock-L) frequently hallucinate physically 
impossible geometries (e.g., overlapping atoms or strained chemical bonds).
This module serves as the thermodynamic reality-check, ensuring that the 
AI's structural predictions are physically viable.

It implements two distinct tiers of physics integration:

1. Lightweight Proxy (CPU-bound, Always Available):
   Acts as a rapid triage filter. It calculates the internal strain energy 
   of the generated ligand pose using the MMFF94 force field, and counts 
   hard steric clashes against the receptor. High strain or clashes indicate 
   a chemically valid molecule placed in an impossible conformation.

2. OpenMM Proper Path (GPU-accelerated, Requires [physics] Extra):
   A localized, rigorous thermodynamic simulation. It parameterizes the 
   ligand with GAFF2/OpenFF and the protein with Amber14. Because Amber14 
   needs explicit hydrogens, the heavy-atom receptor is first completed and 
   protonated with PDBFixer, and the ligand pose is hydrogenated with RDKit. 
   It restrains the heavy atoms of the protein to prevent unravelling, and runs 
   an energy minimization sequence on the AI-generated pose. 
   
   It outputs two critical metrics:
   - Pose Drift (RMSD): How far the physics engine had to move the ligand 
     to make it stable. (Low drift = Excellent AI prediction).
   - Interaction Energy: E(complex) - E(receptor) - E(ligand), computed on the 
     minimized, restraint-free geometry (a binding-energy proxy for ranking).

"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from .utils import get_logger

log = get_logger("physdock.physics")


@dataclass
class PhysicsResult:
    ligand_id: str
    pose_sdf: str
    mode: str
    ligand_strain_kcal: Optional[float] = None
    clash_count: Optional[int] = None
    pose_drift_rmsd: Optional[float] = None
    interaction_energy_kj: Optional[float] = None
    ok: bool = True
    note: str = ""

    def to_row(self) -> dict:
        """
        Converts the PhysicsResult dataclass instance into a standard dictionary.
        
        Uses the built-in `asdict` function from the dataclasses module.
        
        Returns:
            dict: A dictionary representation of the dataclass fields.
            
        Example:
            >>> result = PhysicsResult(ligand_id="drug1", pose_sdf="pose.sdf", mode="lightweight")
            >>> result.to_row()
            {'ligand_id': 'drug1', 'pose_sdf': 'pose.sdf', 'mode': 'lightweight', ...}
        """
        return asdict(self)                                                              # Convert and return the dataclass instance as a dictionary for CSV logging.


# --------------------------------------------------------------------------- #
# Lightweight proxy
# --------------------------------------------------------------------------- #
def _mmff_strain(mol: Chem.Mol) -> Optional[float]:
    """
    Calculates the internal strain energy of a ligand pose.
    
    Determines if the AI-generated 3D pose of the ligand is 
    "strained" (like a stretched rubber band) compared to its natural, relaxed state.
    
    Evaluates the energy of the given pose using the MMFF94 
    force field, then runs an energy minimization (relaxation) on a copy of the 
    molecule, and subtracts the relaxed energy from the original energy.
    
    Args:
        mol (Chem.Mol): The RDKit molecule object representing the 3D pose.
        
    Returns:
        Optional[float]: The strain energy in kcal/mol. Large positive values mean highly strained. Returns None if calculation fails.
        
    Example:
        >>> _mmff_strain(my_rdkit_mol)
        12.45
    """
    try:                                                                                 # Wrap in a try-block because MMFF parameterization can fail on weird chemistry.
        mol = Chem.AddHs(mol, addCoords=True)                                            # Add explicit hydrogen atoms and compute their 3D coordinates based on the heavy atoms.
        props = AllChem.MMFFGetMoleculeProperties(mol)                                   # Generate Merck Molecular Force Field (MMFF) properties for the molecule.
        if props is None:                                                                # If MMFF cannot parameterize the molecule (e.g., unrecognized atom types)...
            return None                                                                  # ...safely return None to indicate failure without crashing the pipeline.
        ff = AllChem.MMFFGetMoleculeForceField(mol, props)                               # Construct the force field object for the initial, AI-generated pose.
        e_pose = ff.CalcEnergy()                                                         # Calculate the internal energy of this initial, unrelaxed pose.
        relaxed = Chem.Mol(mol)                                                          # Create a deep copy of the molecule object to undergo structural relaxation.
        AllChem.MMFFOptimizeMolecule(relaxed, maxIters=500)                              # Mathematically nudge atoms to minimize energy, up to 500 steps.
        ff2 = AllChem.MMFFGetMoleculeForceField(                                         # Construct a new force field object...
            relaxed, AllChem.MMFFGetMoleculeProperties(relaxed))                         # ...specifically for the newly relaxed geometry.
        return float(e_pose - ff2.CalcEnergy())                                          # Return the difference: (energy of AI pose) - (energy of physically relaxed pose).
    except Exception as e:  # noqa: BLE001                                               # Catch any generic RDKit exceptions during the physics calculations.
        log.warning("MMFF strain failed: %s", e)                                         # Log the failure reason so the user knows why a strain score is missing.
        return None                                                                      # Return None to allow the pipeline to continue processing other molecules.


def _clash_count(ligand: Chem.Mol, receptor_pdb: Path, cutoff: float = 2.0) -> Optional[int]:
    """
    Counts the number of severe steric clashes between the ligand and the protein.
    
    Checks if the AI hallucinated the drug physically overlapping 
    with the atoms of the protein.
    
    Extracts all heavy-atom coordinates from both the ligand and 
    the receptor, computes a pairwise Euclidean distance matrix, and counts how 
    many distances are below the physical cutoff (usually 2.0 Angstroms).
    
    Args:
        ligand (Chem.Mol): The RDKit molecule object of the ligand.
        receptor_pdb (Path): Filepath to the protein PDB structure.
        cutoff (float): The distance in Angstroms below which atoms are considered clashing.
        
    Returns:
        Optional[int]: The total number of clashing atoms. None if extraction fails.
        
    Example:
        >>> _clash_count(my_ligand, Path("receptor.pdb"), 2.0)
        3
    """
    try:                                                                                 # Wrap in a try-block to catch missing files or coordinate extraction errors.
        rec_coords = _pdb_heavy_coords(receptor_pdb)                                     # Parse the PDB file and extract a NumPy array of the receptor's heavy atom coordinates.
        lig = ligand.GetConformer().GetPositions()                                       # Extract a NumPy array of the ligand's current 3D atomic coordinates.
        if rec_coords.size == 0:                                                         # Check if the receptor coordinate array is empty (e.g., malformed PDB file).
            return None                                                                  # Safely return None if no receptor atoms were found.
        d = np.linalg.norm(lig[:, None, :] - rec_coords[None, :, :], axis=-1)            # Compute the pairwise Euclidean distance matrix using NumPy broadcasting.
        return int((d < cutoff).any(axis=1).sum())                                       # Count how many ligand atoms have AT LEAST ONE receptor atom closer than the cutoff.
    except Exception as e:  # noqa: BLE001                                               # Catch mathematical or parsing exceptions.
        log.warning("Clash count failed: %s", e)                                         # Log the failure with the error message.
        return None                                                                      # Return None to prevent crashing.


def _pdb_heavy_coords(pdb_path: Path) -> np.ndarray:
    """
    Extracts 3D coordinates of all heavy (non-hydrogen) atoms from a PDB file.
    
    Reads a standard Protein Data Bank file and strips out everything 
    except the XYZ coordinates of the main structural atoms.
    
    Iterates through the file line by line, filters for ATOM/HETATM 
    records that are not Hydrogens, and slices the text to extract the floats.
    
    Args:
        pdb_path (Path): Filepath to the PDB file.
        
    Returns:
        np.ndarray: An Nx3 array of float coordinates.
        
    Example:
        >>> _pdb_heavy_coords(Path("protein.pdb"))
        array([[ 12.3, -4.5,  8.9], ...])
    """
    xyz = []                                                                             # Initialize an empty list to accumulate the X, Y, Z coordinate tuples.
    for ln in Path(pdb_path).read_text().splitlines():                                   # Open the PDB file, read its contents as text, and iterate line by line.
        if ln.startswith(("ATOM", "HETATM")) and ln[76:78].strip() != "H":               # Check if the line is an atom record AND the element column is NOT Hydrogen.
            try:                                                                         # Wrap parsing in a try-block because PDB fixed-width columns can be malformed.
                xyz.append((float(ln[30:38]), float(ln[38:46]), float(ln[46:54])))       # Slice the exact character columns for X, Y, Z coordinates and cast to float.
            except ValueError:                                                           # Catch casting errors if the columns contain unexpected characters.
                pass                                                                     # Silently ignore the bad line and move to the next atom.
    return np.asarray(xyz, dtype=float)                                                  # Convert the accumulated list of tuples into a fast, structured NumPy array.


def score_lightweight(ligand_id: str, pose_sdf: Path, receptor_pdb: Path) -> PhysicsResult:
    """
    Executes the lightweight physics proxy scoring pipeline.
    
    Calculates chemical bond strain and steric clashes without invoking heavy MD simulations.
    
    Loads the ligand from an SDF file, runs the MMFF strain and steric
    clash count functions, and packages the results into a PhysicsResult object.
    
    Args:
        ligand_id (str): Identifier for the ligand.
        pose_sdf (Path): Path to the AI-generated ligand pose in SDF format.
        receptor_pdb (Path): Path to the target protein PDB.
        
    Returns:
        PhysicsResult: A populated dataclass with the lightweight physics scores.
        
    Example:
        >>> score_lightweight("drug1", Path("pose.sdf"), Path("receptor.pdb"))
        PhysicsResult(ligand_id='drug1', ..., ligand_strain_kcal=5.2, clash_count=0, ...)
    """
    mol = next(iter(Chem.SDMolSupplier(str(pose_sdf), removeHs=False)), None)            # Read the first molecule from the SDF file, preserving explicit Hydrogens if present.
    if mol is None:                                                                      # Check if the SDF file was unreadable or empty.
        return PhysicsResult(ligand_id, str(pose_sdf), "lightweight", ok=False,          # Return a failed result object...
                             note="unreadable_pose")                                     # ...and annotate that the SDF could not be parsed.
    return PhysicsResult(                                                                # Construct and return the successful PhysicsResult dataclass.
        ligand_id, str(pose_sdf), "lightweight",                                         # Populate basic identifiers and the chosen mode.
        ligand_strain_kcal=_round(_mmff_strain(mol)),                                    # Compute strain, round to 3 decimals, and store it.
        clash_count=_clash_count(mol, receptor_pdb),                                     # Compute and store the number of steric clashes.
        note="proxy_only(MMFF strain + steric clashes); not a free energy",              # Add a note clarifying that this is an approximation, not a true thermodynamic run.
    )


# --------------------------------------------------------------------------- #
# OpenMM proper path
# --------------------------------------------------------------------------- #
def score_openmm(ligand_id: str, pose_sdf: Path, receptor_pdb: Path, cfg) -> PhysicsResult:
    """
    Executes the rigorous OpenMM thermodynamic physics relaxation.
    
    Builds a full mathematical physics system of the protein and ligand, 
    restrains the protein, and relaxes the system to find a thermodynamically stable state.
    
    Uses SystemGenerator to assign Amber14 parameters to the protein 
    and GAFF2 to the ligand. It builds an OpenMM system, applies positional restraints 
    to the protein's heavy atoms, runs a Langevin energy minimization, and calculates 
    the spatial drift and energy difference.
    
    Args:
        ligand_id (str): Identifier for the ligand.
        pose_sdf (Path): Path to the AI-generated ligand pose.
        receptor_pdb (Path): Path to the target protein PDB.
        cfg (dict-like): The configuration object containing OpenMM parameters.
        
    Returns:
        PhysicsResult: Dataclass containing pose drift and interaction energy proxy.
        
    Example:
        >>> score_openmm("drug1", Path("pose.sdf"), Path("receptor.pdb"), cfg)
        PhysicsResult(ligand_id='drug1', ..., pose_drift_rmsd=0.45, interaction_energy_kj=-120.5, ...)
    """
    # Attempt to import heavy OpenMM dependencies locally.
    try:                                                                                 
        from openmm import app, unit, LangevinMiddleIntegrator, CustomExternalForce      # Import core OpenMM classes for simulation setup and execution.
        from openmm import Platform                                                      # Import Platform to allow hardware acceleration selection (GPU/CPU).
        from openmmforcefields.generators import SystemGenerator                         # Import SystemGenerator to automatically assign physics parameters.
        from openff.toolkit import Molecule                                              # Import OpenFF Molecule to parse the ligand chemistry for GAFF2.
        from pdbfixer import PDBFixer                                                    # Import PDBFixer to complete missing heavy atoms and add hydrogens to the receptor.
        import openmm                                                                    # Import the base openmm module.
    except ImportError as e:                                                             # Catch the error if the user hasn't installed the heavy [physics] environment.
        return PhysicsResult(ligand_id, str(pose_sdf), "openmm", ok=False,               # Return a failed result object...
                             note=f"openmm extra not installed ({e}); "                  # ...explaining the missing dependency...
                                  "use physics.mode=lightweight")                        # ...and suggesting the fallback method.

    # Start the main physics execution block.
    try:                                                                                 
        oc = cfg.get("physics", "openmm", default={})                                    # Extract the 'openmm' specific settings dictionary from the main config.

        # Ligand: Since GAFF2/OpenFF cannot parameterise a heavy-atom-only ligand, load the
        # AI-predicted pose SDF (with any hydrogens that might be present), and because 
        # DiffDock's SDFs are usually heavy-atom only, add the missing H onto the existing 3D coords.
        rd_lig = next(iter(Chem.SDMolSupplier(str(pose_sdf), removeHs=False)), None)     # Read the pose with RDKit, keeping any hydrogens already present.
        if rd_lig is None:                                                               # Guard: the SDF was empty or unreadable.
            return PhysicsResult(ligand_id, str(pose_sdf), "openmm", ok=False,           # Fail this ligand cleanly...
                                 note="unreadable_pose")                                 # ...so the batch continues.
        rd_lig = Chem.AddHs(rd_lig, addCoords=True)                                      # Add any missing hydrogens, placing them on the existing heavy-atom geometry.
        ligand = Molecule.from_rdkit(rd_lig, allow_undefined_stereo=True)                # Build the OpenFF Molecule (tolerate undefined stereo from crystal-derived poses).

        # Receptor: the cleaned PDB is heavy-atom only. Amber14 templates require
        # explicit hydrogens, so a PDBFixer is used to complete partial residues and
        # protonate the protein. We deliberately do NOT model whole missing loops. 
        fixer = PDBFixer(filename=str(receptor_pdb))                                     # Load the heavy-atom receptor into PDBFixer.
        fixer.findMissingResidues()                                                      # Detect gaps (whole missing residues/loops)...
        fixer.missingResidues = {}                                                       # ...but skip modelling them (avoids inventing loop coordinates near/away from the pocket).
        fixer.findMissingAtoms()                                                         # Detect missing heavy atoms within existing residues (e.g. truncated side chains).
        fixer.addMissingAtoms()                                                          # Rebuild those missing heavy atoms.
        fixer.addMissingHydrogens(7.0)                                                   # Add hydrogens at physiological pH 7.0 (sets HIS/CYS protonation states).
        rec_top, rec_pos = fixer.topology, fixer.positions                               # Protonated, completed protein topology + coordinates.

        # Define basic force field rules (physical simulation constraints).
        ff_kwargs = dict(constraints=app.HBonds, rigidWater=True,                        # Constrain hydrogen bonds, keep water rigid...
                         removeCMMotion=False, hydrogenMass=1.5 * unit.amu)              # ...do not remove center-of-mass motion, and repartition hydrogen mass for stability.

        # SystemGenerator takes the abstract topology (the map of atoms) and applies the
        # Amber14 and GAFF2 parameters, converting the static map into an openmm.System
        sysgen = SystemGenerator(                                                        
            forcefields=[oc.get("forcefield_protein", "amber14-all.xml"),                # Use Amber14 force field for the protein amino acids.
                         oc.get("water_model", "amber14/tip3pfb.xml")],                  # Use TIP3P force field for any explicit water molecules.
            small_molecule_forcefield="gaff-2.11",                                       # Use the General Amber Force Field (GAFF2) for the synthetic drug molecule.
            molecules=[ligand], forcefield_kwargs=ff_kwargs)                             # Register the specific ligand molecule and apply the defined constraints.

        # The drug is mathematically injected into the exact 3D coordinate space of the protein, 
        # matching the precise location the AI (DiffDock) predicted it would bind.
        modeller = app.Modeller(rec_top, rec_pos)                                        # Create a Modeller object starting with the protonated protein.
        rec_n = modeller.topology.getNumAtoms()                                          # Protonated receptor atom count; the ligand block is appended AFTER this.
        lig_top = ligand.to_topology().to_openmm()                                       # Convert the OpenFF ligand topology into an OpenMM-compatible topology.
        lig_pos = ligand.conformers[0].to_openmm()                                       # Convert the ligand's 3D coordinates into OpenMM format.
        modeller.add(lig_top, lig_pos)                                                   # Merge the ligand into the protein's Modeller object to create the full complex.

        # Generate the full mathematical physics system (forces, masses, bonds) for the complex.
        system = sysgen.create_system(modeller.topology)                                 

        # Restrain the protein heavy atoms in place, anchoring them to their starting XYZ coordinates 
        # with a harmonic potential and a stiffness (constant 'k') of 1000 kJ/mol/nm^2.
        if oc.get("restrain_protein_heavy", True):                                       # Check config to see if the protein should be locked in place.
            _restrain_protein(system, modeller, openmm, unit,                            # Apply positional restraints to the protein...
                              float(oc.get("restraint_k_kj_mol_nm2", 1000.0)))           # ...using the stiffness constant defined in the configuration.

        # Create the Langevin integrator that will calculate atomic motion over time, and set the simulation parameters.
        integrator = LangevinMiddleIntegrator(                                           
            300 * unit.kelvin, 1 / unit.picosecond, 0.002 * unit.picoseconds)            # Set temperature (300K), friction coefficient, and timestep (2 femtoseconds).
        platform = _pick_platform(Platform)                                              # Automatically select the fastest available hardware (CUDA > OpenCL > CPU).
        # Combine topology, system, and hardware platform (GPU, CPU etc) into a final Simulation object.
        sim = app.Simulation(modeller.topology, system, integrator, platform)            
        sim.context.setPositions(modeller.positions)                                     # Inject the starting XYZ coordinates (AI generated) into the simulation context.

        # Relax the AI pose: protein heavy atoms restrained near the crystal; ligand + pocket free.
        e_before = _potential(sim, unit)                                                 # System potential energy BEFORE minimization (diagnostic).
        sim.minimizeEnergy(maxIterations=int(oc.get("minimize_max_iterations", 2000)))   # Energy-minimize (L-BFGS) to resolve clashes / strain.
        e_after = _potential(sim, unit)                                                  # System potential energy AFTER minimization (diagnostic).
        log.info("%s: relaxation dE = %.1f kJ/mol (diagnostic; NOT the interaction energy)", # Log the relaxation change for insight only...
                 ligand_id, e_after - e_before)                                          # ...explicitly flagged so it is never mistaken for binding energy.

        # Pose drift calculation: ligand heavy-atom RMSD pre vs post energy minimization.
        drift = None                                                                     # Default None so a drift hiccup never voids the whole result.
        try:                                                                             # Isolate drift so its failure can't kill the interaction energy below.
            n_lig = ligand.n_atoms                                                       # Total ligand atoms (incl. H): matches both the slice length and the mask length.
            # Α boolean mask to select only the heavy (non-hydrogen) atoms of the ligand (avoiding the heavy/H count mismatches).
            heavy_mask = np.array([a.atomic_number > 1 for a in ligand.atoms])           # Boolean mask selecting heavy (non-hydrogen) ligand atoms.
            pre_lig = np.asarray(lig_pos.value_in_unit(unit.angstrom))[heavy_mask]       # Pre-min ligand heavy-atom coords (Angstrom), OpenFF atom order.
            # Extract the relaxed coordinates of the whole complex (quantity), slice the ligand coordinates (appended last)
            # applying the same heavy-atom mask/order, and calculate the in-frame RMSD.
            post_q = sim.context.getState(getPositions=True).getPositions()              # Full relaxed complex coordinates (Quantity).
            post_lig = np.asarray(post_q.value_in_unit(unit.angstrom))[-n_lig:][heavy_mask] # Slice the ligand block (appended last); apply the same heavy mask/order.
            drift = _rmsd(pre_lig, post_lig)                                              # In-frame RMSD between two identical, identically-ordered atom sets.
        except Exception as de:  # noqa: BLE001                                          # Catch any coordinate/units edge case.
            log.warning("Pose-drift RMSD failed for %s: %s", ligand_id, de)              # Degrade gracefully; the interaction energy can still be reported.

        # Calculate the interaction energy proxy: E(complex) - E(receptor) - E(ligand)
        interaction = None                                                               # Default None so a decomposition failure never voids the drift metric.
        try:                                                                             # Isolate the decomposition so its failure can't kill drift above.
            # Extract the minimized coordinates of the whole complex, and the receptor atom count.
            post_q = sim.context.getState(getPositions=True).getPositions()              # Minimized coordinates of the whole complex.
            rec_n = rec_top.getNumAtoms()                                                # Protonated-receptor atom count; the ligand block immediately follows it.
            # Compute the single-point potential energy of the complex, the receptor alone, 
            # and the ligand alone, all at the minimized geometry with no restraints.
            e_cpx = _single_point_energy(sysgen.create_system(modeller.topology),        # E(complex): fresh restraint-free system at the minimized geometry.
                                         post_q, openmm, unit, platform)                 # (same coordinates, no positional restraint force).
            e_rec = _single_point_energy(sysgen.create_system(rec_top),                  # E(receptor alone) at those same receptor coordinates.
                                         post_q[:rec_n], openmm, unit, platform)         # Slice the receptor block.
            e_lig = _single_point_energy(sysgen.create_system(lig_top),                  # E(ligand alone) at those same ligand coordinates.
                                         post_q[rec_n:], openmm, unit, platform)         # Slice the ligand block.
            # Interaction energy proxy: the difference between the complex energy and the sum of the parts.
            interaction = e_cpx - e_rec - e_lig                                          # Interaction-energy proxy in kJ/mol (negative is stronger binding).
        except Exception as ie:  # noqa: BLE001                                          # Catch any parameterization/context error.
            log.warning("Interaction-energy decomposition failed for %s: %s", ligand_id, ie) # Degrade gracefully; drift can still be reported.

        # Construct the successful result dataclass.
        return PhysicsResult(                                                            # Build the populated result object.
            ligand_id, str(pose_sdf), "openmm",                                          # Set identifiers and mode.
            pose_drift_rmsd=_round(drift),                                               # Geometric drift (low => the AI pose was already physically sound).
            interaction_energy_kj=_round(interaction),                                   # TRUE interaction energy E(cpx)-E(rec)-E(lig); negative => favorable.
            note="restrained minimization; interaction_energy=E(complex)-E(receptor)-E(ligand) "  # Honest description of the energy metric...
                 "on minimized restraint-free geometry; drift=ligand heavy-atom RMSD pre/post",   # ...and of the drift metric.
        )
    # Catch any physics engine explosions (e.g., atoms placed exactly on top of each other).
    except Exception as e:  # noqa: BLE001                                               
        log.warning("OpenMM scoring failed for %s: %s", ligand_id, e)                    # Log the failure with the specific ligand ID and OpenMM error trace.
        return PhysicsResult(ligand_id, str(pose_sdf), "openmm", ok=False, note=str(e))  # Return a failed result object to prevent pipeline crash.


def _restrain_protein(system, modeller, openmm, unit, k):
    """
    Applies harmonic positional restraints to the heavy atoms of a protein.
    
    Acts like molecular duct tape, holding the protein structure 
    in place so that only the ligand moves during energy minimization.
    
    Creates a CustomExternalForce in OpenMM using a harmonic 
    spring equation. It iterates through the topology, finds protein heavy atoms, 
    and tethers them to their original starting coordinates.
    
    Args:
        system (openmm.System): The OpenMM system object being modified.
        modeller (app.Modeller): The modeller containing topology and positions.
        openmm (module): The openmm base module.
        unit (module): The openmm unit module for handling physics units.
        k (float): The spring constant (stiffness) of the restraint.
        
    Returns:
        None (Modifies the system object in-place).
        
    Example:
        >>> _restrain_protein(my_sys, my_modeller, openmm, unit, 1000.0)
    """
    # Define the structural force object inside OpenMM for a harmonic spring, tethering the heavy atoms to their starting coordinates.
    force = openmm.CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")             # Define the mathematical formula for a harmonic spring tether (Hooke's Law).
    force.addGlobalParameter("k", k * unit.kilojoule_per_mole / unit.nanometer**2)       # Add the spring stiffness constant 'k' as a global parameter with proper physics units.
    # Loop through the reference coordinate axis variables and register them
    # as parameters that will be specific to each individual atom.
    for p in ("x0", "y0", "z0"):                                                         # Loop through the reference coordinate axis variables.
        force.addPerParticleParameter(p)                                                 # Register x0, y0, and z0 as parameters that will be specific to each individual atom.
    pos = modeller.positions                                                             # Extract the starting coordinates from the modeller to use as the tether anchor points.
    # For every heavy atom in the protein, get its coordinates from the modeller 
    # and register that specific atom index to the force object.
    for atom in modeller.topology.atoms():                                               # Iterate through every single atom in the complex's topology.
        if atom.residue.name in _PROTEIN_RES and atom.element is not None and atom.element.symbol != "H": # Check if the atom belongs to an amino acid AND is not a Hydrogen atom.
            xyz = pos[atom.index].value_in_unit(unit.nanometer)                         # Starting X,Y,Z stripped to plain nm floats (per-particle params must be raw doubles, not Quantities).
            force.addParticle(atom.index, [xyz.x, xyz.y, xyz.z])                         # Attach the spring force to this atom, anchoring it to its current xyz position.
    # Registers this entirely new package of constraint rules directly into the main physics object
    system.addForce(force)                                                               # Add the fully constructed restraint force to the main physics system.


_PROTEIN_RES = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "HID",
    "HIE", "HIP", "CYX",
}


def _pick_platform(Platform):
    """
    Selects the fastest available hardware computing platform.
    
    Ensures OpenMM uses the GPU if available, falling back to CPU.
    
    Iterates through a priority list ("CUDA", "OpenCL", "CPU") 
    and returns the first one that successfully initializes.
    
    Args:
        Platform (openmm.Platform): The OpenMM Platform class.
        
    Returns:
        openmm.Platform: The optimal platform object, or None if all fail.
        
    Example:
        >>> _pick_platform(Platform)
        <openmm.Platform 'CUDA'>
    """
    for name in ("CUDA", "OpenCL", "CPU"):                                               # Loop through hardware platforms in order of preference (Nvidia GPU -> Generic GPU -> CPU).
        try:                                                                             # Attempt to initialize the platform name.
            return Platform.getPlatformByName(name)                                      # If the hardware drivers exist, return this platform immediately.
        except Exception:  # noqa: BLE001                                                # If OpenMM throws an error (e.g., CUDA not installed)...
            continue                                                                     # ...ignore the error and try the next platform in the list.
    return None                                                                          # Return None if no computational platforms are available.


def _potential(sim, unit) -> float:
    """
    Retrieves the current potential thermodynamic energy of the simulation.
    
    Checks the "score" of the physics system at its current state.
    
    Queries the simulation context for energy and converts it 
    to standard kilojoules per mole.
    
    Args:
        sim (app.Simulation): The active OpenMM simulation object.
        unit (module): The openmm unit module.
        
    Returns:
        float: The potential energy in kJ/mol.
        
    Example:
        >>> _potential(sim, unit)
        -450.25
    """
    st = sim.context.getState(getEnergy=True)                                            # Request the current physical state of the simulation, specifically asking to calculate energy.
    return st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)                # Extract the potential energy value, strip the OpenMM unit wrapper, and return as a float.


def _single_point_energy(system, positions, openmm, unit, platform):
    """
    Calculates the single-point potential energy (kJ/mol) of a system at frozen, fixed coordinates.
    
    Evaluates the exact thermodynamic energy of a molecular system 
    without allowing any atoms to move or relax. This is specifically used to decompose 
    the total system energy into the Interaction Energy formula: 
    E(complex) - E(receptor) - E(ligand).
    
    It constructs a "dummy" VerletIntegrator (which is strictly required 
    by OpenMM to build a Context, but is never actually instructed to step forward in time). 
    It injects the frozen XYZ coordinates into this temporary context, queries the 
    potential energy, and then immediately deletes the context to free up GPU memory.
    
    Args:
        system (openmm.System): The mathematically parameterized OpenMM system to evaluate.
        positions (unit.Quantity): The frozen 3D coordinates to inject into the system.
        openmm (module): The base openmm module.
        unit (module): The openmm unit module for handling physics units.
        platform (openmm.Platform | None): The hardware platform (GPU/CPU) to run the calculation on.
        
    Returns:
        float: The static potential energy of the geometry in kJ/mol.
        
    Example:
        >>> _single_point_energy(ligand_system, isolated_ligand_coords, openmm, unit, platform)
        -42.5
    """
    integ = openmm.VerletIntegrator(1.0 * unit.femtoseconds)                              # Dummy integrator (required to build a Context; never stepped).
    ctx = (openmm.Context(system, integ, platform) if platform is not None                # Reuse the chosen hardware platform when available...
           else openmm.Context(system, integ))                                           # ...otherwise let OpenMM select a default platform.
    ctx.setPositions(positions)                                                          # Inject the coordinates to evaluate.
    energy = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole) # Read the potential energy in kJ/mol.
    del ctx                                                                              # Release the context (and any GPU memory) promptly.
    return energy                                                                        # Return the scalar energy.


def _rmsd(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    """
    Calculates the Root Mean Square Deviation between two coordinate sets.
    
    Measures the average physical distance (drift) between two 
    sets of 3D points.
    
    Subtracts the matrices, squares the differences, sums along 
    the axes, averages them, and takes the square root.
    
    Args:
        a (np.ndarray): First Nx3 coordinate array.
        b (np.ndarray): Second Nx3 coordinate array.
        
    Returns:
        Optional[float]: The RMSD value. None if inputs are invalid or mismatched.
        
    Example:
        >>> _rmsd(coords_pre, coords_post)
        1.45
    """
    if a is None or b is None or a.shape != b.shape:                                     # Guard clause: Ensure both arrays exist and have the exact same number of atoms/dimensions.
        return None                                                                      # Return None if the geometries are incomparable.
    return float(np.sqrt(((a - b) ** 2).sum(axis=1).mean()))                             # Execute the mathematical RMSD formula using vectorized NumPy operations for speed.


def _round(x, n=3):
    """
    Safely rounds a value if it is a number.
    
    Cleans up float formatting for clean CSV logging.
    
    Checks if the input is a float or int before applying Python's round().
    
    Args:
        x (Any): The value to round.
        n (int): Number of decimal places.
        
    Returns:
        Any: The rounded number, or the original value if not a number.
        
    Example:
        >>> _round(1.234567)
        1.235
    """
    return round(x, n) if isinstance(x, (int, float)) else x                             # Check the type; apply rounding if numeric, otherwise return the raw object (like None).


def score(ligand_id, pose_sdf, receptor_pdb, cfg) -> PhysicsResult:
    """
    Master router function for physics scoring.
    
    Reads the configuration and routes the request to either the 
    lightweight proxy or the proper OpenMM engine.
    
    Checks the 'physics.mode' key in the config object and calls 
    the respective scoring function.
    
    Args:
        ligand_id (str): Identifier for the ligand.
        pose_sdf (str/Path): Path to the ligand SDF file.
        receptor_pdb (str/Path): Path to the receptor PDB file.
        cfg (dict-like): The main configuration object.
        
    Returns:
        PhysicsResult: The final scored dataclass.
        
    Example:
        >>> score("drug1", "pose.sdf", "rec.pdb", cfg)
        PhysicsResult(...)
    """
    mode = cfg.get("physics", "mode", default="lightweight")                             # Read the configuration file to determine which physics methodology the user requested.
    pose_sdf, receptor_pdb = Path(pose_sdf), Path(receptor_pdb)                          # Convert the incoming file path strings into robust pathlib Path objects.
    if mode == "openmm":                                                                 # Check if the user requested the rigorous GPU physics engine.
        return score_openmm(ligand_id, pose_sdf, receptor_pdb, cfg)                      # Route execution to the OpenMM pipeline and return its result.
    return score_lightweight(ligand_id, pose_sdf, receptor_pdb)                          # If not OpenMM, default to routing execution to the CPU lightweight proxy pipeline.