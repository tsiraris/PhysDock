#!/usr/bin/env python
"""
=============================================================================
PhysDock: Stage Aux — Experimental Affinity Acquisition (fetch_chembl.py)
=============================================================================
This auxiliary script serves as the biological data-sourcing engine for the 
PhysDock pipeline. In order to statistically validate the AI models (Stage 07), 
we must correlate their predicted affinity scores against physical reality. 

This script connects to the ChEMBL database (no 1 open-source repository of 
experimental bioactivity data) via its REST API. It dynamically pulls known 
lab measurements for drugs binding to the target protein (e.g., KRAS G12C, ChEMBL ID: CHEMBL2189121).

Crucially, it filters specifically for thermodynamic binding metrics (IC50, 
Ki, Kd) and extracts the standardized 'pChEMBL' value. This data is then used 
as the ground truth to populate the project's ligand manifest, enabling the 
Spearman rank correlation validation.
"""
import argparse, sys, time
from pathlib import Path
import requests

BASE = "https://www.ebi.ac.uk/chembl/api/data"                                           # Define the root base URL for all ChEMBL REST API endpoint requests


def fetch(target_chembl_id, limit):
    """
    Fetches standardized experimental binding affinities from the ChEMBL API.
    
    Queries the ChEMBL database for a specific protein target, filters for 
    high-quality binding assays, and paginates through the results to extract 
    the drug identity and binding strength.
    
    Constructs a REST API GET request targeting the `/activity` endpoint. 
    It uses a while-loop to handle API pagination (extracting the 'next' 
    URL from the metadata). It respects server rate limits using `time.sleep()`. 
    For each record, it strips away heavy metadata, saving only the SMILES, 
    ChEMBL ID, assay type, and the standardized pChEMBL value.
    
    Args:
        target_chembl_id (str): The specific alphanumeric ChEMBL ID for the target protein.
        limit (int): The maximum number of records to retrieve per API page.
        
    Returns:
        list[dict]: A list of dictionaries, where each dict represents a single drug's bioactivity profile.
        
    Example:
        >>> data = fetch("CHEMBL2189121", 20)
        >>> print(data[0])
        {'molecule_chembl_id': 'CHEMBL4299834', 'standard_type': 'IC50', 'pchembl_value': '8.3', 'canonical_smiles': '...'}
    """
    # Construct the initial API query URL with the specified target and filtering for standard binding metrics (IC50, Ki, Kd)
    url = (f"{BASE}/activity?target_chembl_id={target_chembl_id}"                        # Construct the initial query string targeting the specific biological protein ID...
           f"&standard_type__in=IC50,Ki,Kd&limit={limit}&format=json")                   # ...filtering strictly for standard binding metrics (IC50, Ki, Kd) and requesting JSON format
    out = []                                                                             # Initialize an empty list to accumulate the scraped activity records
    # Initiate a while-loop to handle pagination (keep looping as long as a 'next page' URL exists)
    while url:                                                                           # Loop as long as the 'next' page URL exists
        r = requests.get(url, timeout=60); r.raise_for_status(); j = r.json()            # Execute HTTP GET request (60s timeout), enforce success (throw error if 404/500), and parse the JSON payload
        for a in j["activities"]:                                                        # Iterate through the list of biological activity records returned in the current page block
            out.append({"molecule_chembl_id": a.get("molecule_chembl_id"),               # Extract the unique ChEMBL ID for the drug molecule and append to the results dictionary
                        "standard_type": a.get("standard_type"),                         # Extract the exact type of assay measurement performed (e.g., IC50, Kd)
                        "pchembl_value": a.get("pchembl_value"),                         # Extract the standardized negative log affinity score (pChEMBL), which acts as our ground truth
                        "canonical_smiles": a.get("canonical_smiles")})                  # Extract the 1D chemical string representation of the drug for later cross-referencing
        nxt = j["page_meta"].get("next")                                                 # Check the API pagination metadata block to see if there is a subsequent page of results
        url = ("https://www.ebi.ac.uk" + nxt) if nxt else None                           # Construct the next URL string if 'next' exists; otherwise, set to None to seamlessly break the while-loop
        time.sleep(0.2)                                                                  # Pause execution for 200 milliseconds to respect ChEMBL rate limits and prevent IP blacklisting
    return out                                                                           # Return the fully populated list of extracted activity dictionaries


if __name__ == "__main__":
    ap = argparse.ArgumentParser()                                                       # Initialize the standard command-line argument parser
    # Define the target argument, defaulting to the specific ID for mutant KRAS G12C
    ap.add_argument("--target", default="CHEMBL2189121")                                 
    # Define the pagination limit argument (number of records retrieved per single API page request), defaulting to 500 records per page in the API call
    ap.add_argument("--limit", type=int, default=500)                                    
    # Define the output file path destination for the scraped CSV data
    ap.add_argument("--out", default="data/ligands/chembl_kras_g12c.csv")                
    a = ap.parse_args()                                                                  # Parse the arguments provided by the user in the terminal
    import pandas as pd                                                                  # Import pandas locally to handle the tabular data conversion
    # Execute the fetch function and convert the returned list of dictionaries directly into a Pandas DataFrame
    df = pd.DataFrame(fetch(a.target, a.limit))                                          
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)                                # Ensure the target directory structure exists, safely creating parent folders if necessary
    # Save the DataFrame to disk as a CSV file, and print a success message detailing how many records were retrieved.
    df.to_csv(a.out, index=False)                                                        # Serialize the DataFrame to disk as a CSV, omitting the pandas row index column
    print(f"Wrote {len(df)} activities -> {a.out}. Map molecule_chembl_id/SMILES "       # Print a success message detailing how many records were retrieved...
          "to your manifest rows and fill pchembl.")                                     # ...and explicitly instruct the scientist to manually review and map the data