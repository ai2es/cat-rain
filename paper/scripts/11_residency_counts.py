"""Step 11: Counts for the residency-based samples in response-letter Table 4 (jcelis pipeline).

  A1 48,584  devices with >=1 ping on >=45 of 47 days in 2024 (04_california_residents.py, no spatial filter)
  A3 10,849  of those with a home ZIP code in the 568-ZIP study-area list (Paired_Analysis.ipynb)
  A4  7,178  of those that also appear anywhere in the 2023 extract (Common_UIDs.ipynb)
  B1 36,476  devices with >=1 ping on >=45 of 47 days in 2023
  C1  4,668  A1 and B1 (statewide);  C2  1,182  A3 and B1
The 45-day lists are read from jcelis's saved outputs; this script reproduces the downstream counts.

Usage (from the repo root; needs ~64 GB, run on casper):
    python paper/scripts/11_residency_counts.py -c paper/config/11_residency_counts.yaml
"""
import argparse
import json
import os
import re

import pandas as pd
import yaml


def main(config):
    nb = json.load(open(config["paired_analysis_notebook"]))
    src = [''.join(c["source"]) for c in nb["cells"] if "zipcodes_custom = [" in ''.join(c["source"])][0]
    zips = set(re.findall(r"\d{5}", src.split("zipcodes_custom = [")[1].split("]")[0]))

    res24 = pd.Index(pd.read_csv(config["residents_2024"]).iloc[:, 0].unique())
    res23 = pd.Index(pd.read_csv(config["residents_2023"]).iloc[:, 0].unique())
    home = pd.read_parquet(config["home_loc_file"], columns=["zipcode"])
    home = home[home.index.isin(res24)]
    roi = home.index[home["zipcode"].astype(str).isin(zips)].unique()
    key23 = pd.read_parquet(config["key_2023"]).index

    counts = {
        "study_area_zip_list": len(zips),
        "A1_residents_2024": len(res24),
        "A1_with_home_location": home.index.nunique(),
        "A3_home_in_study_area": len(roi),
        "A4_A3_in_2023_extract": int(roi.isin(key23).sum()),
        "B1_residents_2023": len(res23),
        "C1_A1_and_B1": int(res24.isin(res23).sum()),
        "C2_A3_and_B1": int(roi.isin(res23).sum()),
    }
    for k, v in counts.items():
        print(f"{k}: {v:,}")
    os.makedirs(os.path.dirname(config["out_file"]), exist_ok=True)
    pd.Series(counts).to_csv(config["out_file"], header=["devices"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, help="Path to config file")
    args = parser.parse_args()
    with open(args.config) as config_file:
        main(yaml.safe_load(config_file))
