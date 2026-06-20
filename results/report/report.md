# PhysDock report — KRAS_G12C

## What this run tested
A physics-aware diffusion pipeline for protein–ligand interaction on an oncology target: diffusion docking (DiffDock-L) and/or co-folding (Boltz-2) → conformational-ensemble analysis → physics rescoring → validation against crystal poses and experimental affinity.

## Pose accuracy (geometry)
- Ligands with a crystal reference scored: **4**
- Top-pose successes (RMSD ≤ 2.0 Å): **1/4**

![pose rmsd](figures/pose_rmsd.png)

## Affinity ranking (function)
_Sign convention: `affinity_pred_value` (Boltz, log10 IC50 µM) and `interaction_energy_kj` are **lower = stronger binder**, while pChEMBL is **higher = stronger**. So for these predictors a **negative** Spearman ρ indicates correct ranking. (`ligand_strain_kcal` is a pose-quality proxy, not an affinity term — no sign is expected.)_

- `affinity_pred_value`: underpowered: only 0 labelled points (need >= 4); fill pchembl in the manifest
- `ligand_strain_kcal`: underpowered: only 0 labelled points (need >= 4); fill pchembl in the manifest
- `interaction_energy_kj`: underpowered: only 0 labelled points (need >= 4); fill pchembl in the manifest

## Per-ligand results

| ligand_id | role | pchembl | smiles | pose_rmsd | pose_success | iptm | affinity_pred_value | affinity_prob_binary | ligand_strain_kcal | interaction_energy_kj | pose_drift_rmsd | n_poses | pose_spread_rmsd | n_clusters |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sotorasib | covalent_clinical |  | CCC(=O)N1CCN(c2nc(=O)n(-c3c(C)ccnc3C(C)C)c3nc(-c4c(O)cccc4F)c(F)cc23)[C@@H](C)C1 | 1.189 | True | 0.9755867719650269 | 0.7074805498123169 | 0.8509381413459778 |  | -413.994 | 0.997 | 1 |  |  |
| adagrasib | covalent_clinical |  | C[C@H](F)C(=O)N1CCN(c2nc(OC[C@@H]3CCCN3C)nc3c2CCN(c2cccc4cccc(Cl)c24)C3)C[C@@H]1CC#N | 2.923 | False | 0.9867632985115052 | -1.9054808616638184 | 0.9503950476646424 |  | -475.938 | 1.421 | 1 |  |  |
| ars1620 | covalent_tool |  | CCC(=O)N1CCN(c2ncnc3c(F)c(-c4c(O)cccc4F)c(Cl)cc23)CC1 | 4.788 | False | 0.9847179651260376 | -0.6808910965919495 | 0.9489796161651612 |  | -297.293 | 1.076 | 3 | 1.589 | 1.0 |
| ars853 | covalent_tool |  | C=CC(=O)N1CC(N2CCN(C(=O)CNc3cc(C4(C)CC4)c(Cl)cc3O)CC2)C1 | 7.37 | False | 0.980377435684204 | -0.1065341830253601 | 0.7453189492225647 |  | -277.834 | 1.51 | 1 |  |  |

## What is NOT claimed
- No wet-lab validation; all signals are *in silico*.
- Lightweight physics mode is a strain/clash **proxy**, not a free energy. OpenMM mode reports restrained-minimization drift + an interaction-energy proxy (E_complex - E_receptor - E_ligand), still short of MM-GBSA/MD.
- Affinity correlation is only meaningful once experimental labels are filled and enough points remain; an underpowered ρ is reported as such, not spun.
- Boltz affinity and DiffDock confidence are model self-estimates, validated here only against the small labelled subset.