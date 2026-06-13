"""
PhysDock: Automated Reporting & Falsification Module (report.py)

This module translates raw algorithmic outputs into transparent scientific reports
for a multidisciplinary audience (e.g., medicinal chemists and biologists).

Instead of just outputting a CSV, this script automatically generates a 
publication-ready Markdown report complete with embedded Matplotlib visualizations.
Most importantly, it strictly enforces a "falsification-first" philosophy. It 
explicitly states the limitations of the pipeline in a "What is NOT claimed" 
section, acknowledging the difference between a physics proxy and a true free-energy
perturbation (FEP) calculation, for credibility and scientific rigor.
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .utils import ensure_dir, get_logger

log = get_logger("physdock.report")


def _to_md_table(df: pd.DataFrame) -> str:
    """
    Render a DataFrame as a GitHub-flavoured Markdown table WITHOUT requiring `tabulate`.

    pandas' `DataFrame.to_markdown()` raises ImportError unless the optional `tabulate`
    package is installed, which would crash the final report stage after all GPU spend.
    We try it first (nicer alignment); if `tabulate` is missing we build the pipe-table
    by hand so report generation never fails on a missing optional dependency.

    Args:
        df (pd.DataFrame): The table to render.

    Returns:
        str: A Markdown table string ("_no rows_" if the frame is empty).
    """
    if df is None or df.empty:                                                           # Guard against an empty/None frame.
        return "_no rows_"                                                               # Mirror the previous fallback text.
    try:                                                                                 # Prefer pandas' native renderer when available.
        return df.to_markdown(index=False)                                               # Uses `tabulate` under the hood (prettier alignment).
    except Exception:  # noqa: BLE001                                                    # Typically ImportError (tabulate absent); never fatal.
        cols = [str(c) for c in df.columns]                                              # Stringify the header cells.
        header = "| " + " | ".join(cols) + " |"                                          # Build the header row.
        sep = "| " + " | ".join("---" for _ in cols) + " |"                              # Build the separator row.
        rows = ["| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |"      # Build each data row (NaN -> empty cell)...
                for row in df.itertuples(index=False, name=None)]                        # ...iterating rows as plain tuples.
        return "\n".join([header, sep, *rows])                                           # Join into a single Markdown table string.


def _bar_rmsd(df: pd.DataFrame, out_png: Path, thr: float):
    """
    Generates a color-coded bar chart of ligand RMSD values vs crystal structures.
    
    Creates a visual summary of the AI's geometric docking accuracy, 
    making it instantly clear to a medicinal chemist which poses succeeded and which failed.
    
    Filters out missing data, dynamically assigns colors based on 
    the success threshold (green for success, red for failure), and uses Matplotlib 
    to render and save a static PNG image.
    
    Args:
        df (pd.DataFrame): The DataFrame containing 'ligand_id' and 'pose_rmsd' columns.
        out_png (Path): The file path where the generated PNG image should be saved.
        thr (float): The RMSD success threshold (usually 2.0 A) to draw the cutoff line.
        
    Returns:
        Path or None: Returns the path to the saved image, or None if there was no data.
        
    Example:
        >>> _bar_rmsd(pose_df, Path("results/figures/rmsd.png"), 2.0)
        Path('results/figures/rmsd.png')
    """
    sub = df.dropna(subset=["pose_rmsd"])                                                # Filter out any rows where the RMSD calculation failed or was missing.
    if sub.empty:                                                                        # Guard clause: Check if there is actually any valid data left to plot.
        return None                                                                      # Return None gracefully to skip chart generation without crashing the pipeline.
    fig, ax = plt.subplots(figsize=(6, 3.5))                                             # Initialize a Matplotlib figure and axis with specific, report-friendly dimensions.
    colors = ["#2a9d8f" if v <= thr else "#e76f51" for v in sub["pose_rmsd"]]        # Use list comprehension to assign teal (success) or orange/red (failure) based on threshold.
    ax.bar(sub["ligand_id"], sub["pose_rmsd"], color=colors)                             # Plot the bar chart mapping the ligand IDs on the X-axis to their RMSD on the Y-axis.
    ax.axhline(thr, ls="--", c="k", lw=1, label=f"success <= {thr} A")                   # Draw a black, dashed horizontal line exactly at the success threshold for visual reference.
    ax.set_ylabel("top-pose RMSD vs crystal (A)")                                        # Label the Y-axis with the metric and units (Angstroms).
    ax.set_title("Diffusion-docking pose accuracy")                                      # Add a clear, scientific title to the chart.
    ax.legend(); plt.xticks(rotation=45, ha="right"); plt.tight_layout()                 # Render the legend, rotate the X-axis labels 45 degrees for readability, and pack the layout tightly.
    fig.savefig(out_png, dpi=150); plt.close(fig)                                        # Save the figure to disk at 150 DPI resolution, then clear it from memory to prevent RAM leaks.
    return out_png                                                                       # Return the file path so the Markdown builder knows where to link the image.


def _scatter_affinity(df: pd.DataFrame, pred_col: str, label_col: str, out_png: Path):
    """
    Generates a scatter plot comparing AI-predicted affinity against experimental labels.
    
    Visually demonstrates the functional correlation (or lack thereof) 
    between the pipeline's thermodynamic scoring and actual wet-lab results.
    
    Extracts the predicted and experimental columns, coerces them 
    into strict numerics (dropping text or NaNs), checks for minimum sample size, 
    and plots the distribution.
    
    Args:
        df (pd.DataFrame): The merged DataFrame containing both scores and labels.
        pred_col (str): The column name for the AI prediction (e.g., 'openmm_energy').
        label_col (str): The column name for the experimental ground truth (e.g., 'pchembl').
        out_png (Path): The file path where the generated PNG image should be saved.
        
    Returns:
        Path or None: Returns the path to the saved image, or None if data was insufficient.
        
    Example:
        >>> _scatter_affinity(df, "boltz_affinity", "pchembl", Path("scatter.png"))
    """
    sub = df[[pred_col, label_col]].apply(pd.to_numeric, errors="coerce").dropna()       # Isolate the two columns, force values to float, and drop any rows containing NaNs.
    if len(sub) < 3:                                                                     # Guard clause: A scatter plot of 1 or 2 points is visually meaningless.
        return None                                                                      # Skip chart generation if the dataset is too small.
    fig, ax = plt.subplots(figsize=(4.5, 4.5))                                           # Initialize a square Matplotlib figure for the scatter plot.
    ax.scatter(sub[pred_col], sub[label_col], c="#264653")                             # Plot the data points using a professional dark blue hex color.
    ax.set_xlabel(f"predicted ({pred_col})")                                             # Dynamically label the X-axis with the specific AI prediction metric used.
    ax.set_ylabel(f"experimental ({label_col})")                                         # Dynamically label the Y-axis with the specific experimental metric used.
    ax.set_title("Predicted vs experimental affinity")                                   # Set the overarching title for the plot.
    plt.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)                    # Pack the layout, save the high-res PNG to disk, and release the figure memory.
    return out_png                                                                       # Return the file path for the Markdown markdown image link.


def build(results_dir, merged: pd.DataFrame, pose_eval: pd.DataFrame,
          corr_stats: list[dict], cfg) -> Path:
    """
    Master orchestration function that compiles the final scientific Markdown report.
    
    Aggregates all data frames, statistics, and figures from the pipeline 
    stages into a single, highly readable Markdown document. It structures the data 
    logically, providing summary statistics and explicitly bounding the scientific claims.
    
    Creates necessary directories, calls the plotting functions, 
    calculates summary integers, constructs a list of Markdown-formatted text strings, 
    and writes them to a final 'report.md' file.
    
    Args:
        results_dir (str/Path): The root output directory for the pipeline run.
        merged (pd.DataFrame): The master DataFrame containing all ligand data and scores.
        pose_eval (pd.DataFrame): The DataFrame specifically tracking RMSD pose accuracy.
        corr_stats (list[dict]): The list of Spearman correlation statistics dictionaries.
        cfg (DictConfig/dict): The main configuration object driving the pipeline.
        
    Returns:
        Path: The absolute path to the newly generated Markdown report file.
        
    Example:
        >>> report_path = build("results/", df_merged, df_pose, stats_list, cfg)
    """
    results_dir = ensure_dir(results_dir)                                                # Ensure the main output directory exists; create it if it doesn't.
    fig_dir = ensure_dir(Path(results_dir) / "figures")                                  # Ensure a dedicated subfolder exists strictly for the image files.
    thr = float(cfg.get("evaluate", "rmsd_success_threshold", default=2.0))              # Extract the RMSD success threshold from the configuration YAML.
    label_col = cfg.get("evaluate", "affinity_label_column", default="pchembl")          # Extract the name of the experimental column (e.g., 'pchembl') from YAML.

    rmsd_png = _bar_rmsd(pose_eval, fig_dir / "pose_rmsd.png", thr)                      # Call the geometric plotting function and store the resulting image path.
    scat_png = None                                                                      # Initialize the scatter plot path variable as None.
    for s in corr_stats:                                                                 # Iterate through all the evaluated correlation statistic dictionaries.
        if s.get("usable"):                                                              # If a statistic passed the minimum-data threshold (is usable)...
            scat_png = _scatter_affinity(merged, s["pred_col"], label_col,               # ...generate the scatter plot for that specific prediction vs label...
                                         fig_dir / "affinity_scatter.png")               # ...saving it to the figures directory.
            break                                                                        # Stop after generating the first valid scatter plot to avoid overwriting.

    n_scored = pose_eval["pose_rmsd"].notna().sum() if not pose_eval.empty else 0        # Count the total number of ligands that successfully received an RMSD score.
    n_success = int(pose_eval["pose_success"].sum()) if not pose_eval.empty else 0       # Count how many of those RMSD scores were strictly below the success threshold.

    lines = [                                                                            # Begin constructing the raw text lines for the Markdown document.
        f"# PhysDock report — {cfg.get('target', 'name', default='target')}",            # Markdown H1 Title dynamically naming the biological target.
        "",                                                                              # Blank line for formatting.
        "## What this run tested",                                                       # Markdown H2 section explaining the pipeline architecture.
        "A physics-aware diffusion pipeline for protein–ligand interaction on an "       # Narrative text describing the method: Generative AI...
        "oncology target: diffusion docking (DiffDock-L) and/or co-folding (Boltz-2) "   # ...specifically citing the diffusion models...
        "→ conformational-ensemble analysis → physics rescoring → validation against "   # ...and the sequential data-flow of the pipeline.
        "crystal poses and experimental affinity.",                                      # ...ending with the validation steps.
        "",                                                                              # Blank line.
        "## Pose accuracy (geometry)",                                                   # Markdown H2 section for Geometric Validation summary.
        f"- Ligands with a crystal reference scored: **{n_scored}**",                    # Bullet point showing the total sample size of scored ligands.
        f"- Top-pose successes (RMSD ≤ {thr} Å): **{n_success}/{n_scored}**",            # Bullet point showing the final success ratio (e.g., 3/4).
        "",                                                                              # Blank line.
    ]
    if rmsd_png:                                                                         # Check if the RMSD bar chart PNG was successfully generated.
        lines += [f"![pose rmsd](figures/{rmsd_png.name})", ""]                          # Inject the Markdown image syntax linking to the relative file path.

    lines += ["## Affinity ranking (function)"]                                          # Markdown H2 section for Functional Validation summary.
    for s in corr_stats:                                                                 # Iterate through the calculated correlation dictionaries again.
        if s.get("usable"):                                                              # If the statistic is statistically valid...
            lines.append(f"- `{s['pred_col']}`: Spearman ρ = **{s['rho']}** "            # ...format a bullet point showing the prediction metric and the Rho score...
                         f"(p={s['p']}, n={s['n']}). {s['note']}")                       # ...and append the p-value, sample size (n), and explanatory note.
        else:                                                                            # If the statistic was flagged as unusable (e.g., underpowered)...
            lines.append(f"- `{s['pred_col']}`: {s['note']}")                            # ...print the prediction metric and the warning note directly.
    lines.append("")                                                                     # Blank line.
    if scat_png:                                                                         # Check if the Affinity scatter plot PNG was successfully generated.
        lines += [f"![affinity](figures/{scat_png.name})", ""]                           # Inject the Markdown image syntax linking to the scatter plot.

    lines += [                                                                           # Append the final sections of the report.
        "## Per-ligand results",                                                         # Markdown H2 for the raw data table.
        "",                                                                              # Blank line.
        _to_md_table(merged),                                                            # Render the per-ligand table without requiring the optional `tabulate` package.
        "",                                                                              # Blank line.
        "## What is NOT claimed",                                                        # Markdown H2: The crucial scientific transparency and boundary-setting section.
        "- No wet-lab validation; all signals are *in silico*.",                         # Disclaimer 1: Computational predictions are hypotheses, not lab realities.
        "- Lightweight physics mode is a strain/clash **proxy**, not a free energy. "    # Disclaimer 2: Clarify that the CPU method is an approximation...
        "OpenMM mode reports restrained-minimization drift + an interaction-energy proxy "  # ...and that OpenMM gives drift plus E(complex)-E(receptor)-E(ligand)...
        "(E_complex - E_receptor - E_ligand), still short of MM-GBSA/MD.",                # ...which is still an enthalpic proxy, short of MM-GBSA / full MD free energies.
        "- Affinity correlation is only meaningful once experimental labels are filled "   # Disclaimer 3: Warn the user that Spearman correlation requires...
        "and enough points remain; an underpowered ρ is reported as such, not spun.",    # ...actual data; reaffirming the pipeline's anti-fluke statistical integrity.
        "- Boltz affinity and DiffDock confidence are model self-estimates, validated "  # Disclaimer 4: Remind the reader that AI confidence scores...
        "here only against the small labelled subset.",                                  # ...are internal heuristics, not definitive physical properties.
    ]
    report = Path(results_dir) / "report.md"                                             # Define the final absolute path for the 'report.md' text file.
    report.write_text("\n".join(str(x) for x in lines))                                  # Join all lines with newline characters and write the text payload to disk.
    log.info("Report -> %s", report)                                                     # Log the successful completion and location of the report to the console.
    return report                                                                        # Return the file path to the orchestration script.