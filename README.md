# PhysDock

**Physics-Aware Diffusion Co-Folding & Conformational Sampling for Protein–Ligand Interactions** *An Oncology (KRAS G12C) Case Study & Validation Pipeline*

---

## 🎯 Executive Summary

PhysDock is a modular, reproducible, and inference-optimized computational chemistry pipeline. It integrates state-of-the-art diffusion generative models with rigorous thermodynamic simulations to predict and validate small-molecule binding interactions against complex oncology targets.

Designed for efficiency and falsifiability, PhysDock answers two critical drug discovery questions:

1. **Geometry ("Where does it bind?"):** Utilizing diffusion models to predict structural poses.
2. **Function ("How well does it bind?"):** Utilizing physics-based simulations and neural-network affinity heads to rank candidate efficacy.

The pipeline is engineered with strict MLOps principles: it hard-fails chemically invalid chemistry (and flags PAINS/property/synthetic-accessibility advisories) *before* GPU execution, triangulates predictions across independent AI architectures, and validates all outputs against crystallographic (RMSD) and experimental (pChEMBL) ground truths.

**Hardware Profile:** Engineered to execute end-to-end on a single 24 GB GPU (e.g., AWS A10G), with a decoupled CPU-bound analytical core.

---

## 🧬 The Scientific Architecture

PhysDock orchestrates a multi-tier approach to molecular prediction, explicitly bridging the gap between Machine Learning and Molecular Physics:

* **Tier A (Geometry via Diffusion):** *DiffDock-L* generates a conformational ensemble of ligand poses within a rigid receptor, bypassing traditional heuristic docking searches.
* **Tier B (Induced Fit & Affinity):** *Boltz-2* (AlphaFold3-class architecture) performs joint protein-ligand co-folding from sequences/SMILES to capture dynamic pocket shifts and regresses a predicted binding affinity.
* **Tier C (Thermodynamic Reality Check):** *OpenMM* executes a restrained physics relaxation on the AI-generated poses using Amber14/GAFF2 force fields, quantifying geometric drift and interaction energy to penalize AI hallucinations (e.g., steric clashes).

### 🚀 The Execution Pipeline

| Stage | Script | Core Module | Scientific Objective |
| --- | --- | --- | --- |
| **00** | `00_setup_check.py` | — | **Pre-Flight:** Validates environment dependencies and core CPU logic. |
| **01** | `01_prepare_target.py` | `receptor.py` | **Data Prep:** Cleans receptor, extracts crystal reference poses & SMILES. |
| **02** | `02_chem_gate.py` | `chem.py` | **Triage:** Hard-fails invalid valence/chemistry; flags PAINS/SA/property advisories. |
| **03** | `03_run_diffdock.py` | `docking_diffdock.py` | **Generative AI:** Predicts binding poses via diffusion sampling. |
| **04** | `04_run_boltz.py` | `cofold_boltz.py` | **Co-Folding:** Predicts induced-fit complex and binding affinity. |
| **05** | `05_physics_rescore.py` | `physics_openmm.py` | **Physics:** Evaluates thermodynamic stability and pose drift. |
| **06** | `06_ensemble_analysis.py` | `ensemble.py` | **MD Surrogate:** Analyzes conformational landscape flexibility & clustering. |
| **07** | `07_evaluate_and_report.py` | `evaluate.py` | **Validation:** Calculates RMSD vs. crystal and Spearman rank vs. experiment. |

---

## 📐 Core Design Philosophy

* **Generative Triangulation:** By utilizing two distinct generative engines (DiffDock and Boltz-2), the pipeline actively seeks consensus. Agreement across models, physics, and ground truth yields a high-confidence hypothesis. Disagreement is explicitly flagged for human review.
* **Ensemble Sampling over MD:** Multi-seed diffusion generation is utilized as a rapid surrogate for computationally prohibitive Molecular Dynamics (MD) simulations, allowing for statistical landscape analysis in minutes rather than weeks.
* **Explicit Physics Integration:** Physics is not hidden in a black-box scoring function. The pipeline exposes inspectable metrics (internal strain, steric clashes, relaxation drift) to ensure predictions obey thermodynamic laws.
* **Falsification over Flattery:** Success is defined strictly by geometric alignment against actual X-ray crystallography (RMSD ≤ 2.0 Å) and statistical rank correlation against wet-lab assays. Underpowered statistics are automatically flagged and rejected.

---

## 🛠️ Quickstart & Usage

The pipeline is decoupled. The heavy ML models and physics engines are optional installations, allowing the analytical core (RDKit, Pandas, Biopython) to run on any local CPU machine for rapid data prep and reporting.

```bash
# 1. Install the CPU-bound analytical core
pip install -e .
python scripts/00_setup_check.py          # -> SMOKE TEST PASSED

# 2. Extract targets & run cheminformatics triage (CPU)
python scripts/01_prepare_target.py
python scripts/02_chem_gate.py

# 3. Execute Generative Inference (Requires GPU & Heavy Dependencies)
# Note: clone DiffDock-L into its own env and `pip install boltz` (pin the CUDA
#       build to your AMI driver); install physics extras with `pip install ".[physics,rmsd]"`.
export PHYSDOCK_DIFFDOCK_DIR=/path/to/DiffDock
python scripts/03_run_diffdock.py
python scripts/04_run_boltz.py --max-ligands 4   # Implements AWS budget capping

# 4. Thermodynamic Relaxation & Statistical Validation (CPU/GPU)
python scripts/05_physics_rescore.py --top-per-ligand 3
python scripts/06_ensemble_analysis.py
python scripts/07_evaluate_and_report.py

```

*Outputs are serialized to `results/report/report.md` alongside unified CSV ledgers.*

---

## ⚠️ Data Integrity & Pre-Flight Checks

PhysDock enforces strict data provenance. The project ships with `data/ligands/kras_g12c_ligands.csv`, mapping candidate drugs (e.g., Sotorasib) to their known PDB co-crystal identifiers (e.g., `6OIM`).

Before executing a full run, the researcher **must**:

1. Confirm the `pdb_id` / `ligand_resname` pairing on the RCSB PDB and set `verify=1` in the manifest.
2. Populate the `pchembl` column using `scripts/fetch_chembl_affinities.py` to enable meaningful functional correlation.

*Stage 01 extracts SMILES and spatial coordinates directly from the deposited structures. An incorrect PDB pairing will fail loudly by design, preventing the fabrication of chemistry.*

---

## 📂 Repository Structure

```text
PhysDock/
├── configs/
│   └── kras_g12c.yaml                  # Master execution parameters & thresholds
├── data/
│   └── ligands/kras_g12c_ligands.csv   # Ground-truth co-crystal and affinity manifest
├── physdock/                           # Core Scientific Library
│   ├── chem.py                         # Cheminformatics triage (Valency, SA, PAINS)
│   ├── receptor.py                     # PDB parser & spatial coordinate extractor
│   ├── docking_diffdock.py             # Tier A (Geometry) wrapper & parser
│   ├── cofold_boltz.py                 # Tier B (Affinity) wrapper & parser
│   ├── physics_openmm.py               # Thermodynamic relaxation & strain calculation
│   ├── ensemble.py                     # Conformational clustering & spread analysis
│   └── evaluate.py / report.py         # Statistical correlation & Markdown generation
├── scripts/                            # Orchestration Microservices
│   ├── 00_setup_check.py -> 07_evaluate_and_report.py
│   ├── run_all.sh                      # CI/CD end-to-end execution script
│   └── fetch_chembl_affinities.py      # Auxiliary biological API scraper
├── notebooks/01_quickstart.ipynb       # Interactive tutorial
├── pyproject.toml                      # Modern PEP 517/518 build configuration
├── README.md                           # Project documentation
├── requirements.txt                    # Python package dependencies
├── LICENSE                             # MIT License
└── .gitignore                          # CI/CD exclusions
```

---

## 🔮 Roadmap & Extensions

While this pipeline represents a state-of-the-art inference architecture, future development is targeted at expanding its biophysical scope:

* **Covalent Modeling:** Upgrading the physics engine to explicitly model the covalent adduct formation inherent to KRAS G12C inhibitors (Cys12 warheads), currently treated non-covalently.
* **Enhanced Dynamics:** Integrating learned conformational generators (e.g., AlphaFlow) followed by targeted, short-trajectory OpenMM MD simulations seeded from top AI poses.
* **Free Energy Perturbation (FEP):** Replacing the lightweight interaction energy proxy with full MM-GBSA or FEP calculations for rigorous $\Delta G$ estimations on the shortlisted candidates.
* **Closed-Loop Active Learning:** Connecting the evaluation validation stage directly to a Reinforcement Learning generator to autonomously hypothesize and evaluate novel chemical spaces.