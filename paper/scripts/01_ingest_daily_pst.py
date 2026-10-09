"""Step 1: Convert raw Veraset daily (UTC) parquet folders into one parquet file per Pacific-time day.

Each raw folder YYYY/MM/DD/ holds one UTC day, which spans two Pacific-time calendar days. Pings are
regrouped so that each output file veraset_YYYY-MM-DD_PST.parquet holds exactly one local day, indexed
by (caid, datetime). No pings are filtered.

Source: /glade/work/cbecker/cat-rain/scripts/create_intermediate_data.py (run separately for 2023 and
2024, with the start/end dates in the config).

Usage (from the repo root):
    python paper/scripts/01_ingest_daily_pst.py -c paper/config/01_ingest_2024.yaml
"""
import argparse
import multiprocessing
import os
import time
from glob import glob
from multiprocessing import Pool
from os.path import join

import numpy as np
import pandas as pd
import yaml

from src.catrain.preprocess import load_data


def main(config):
    base_path = config["raw_path"]
    out_path = config["out_path"]
    os.makedirs(out_path, exist_ok=True)
    n_workers = config.get("n_workers", -1)
    n_workers = n_workers if n_workers > 0 else multiprocessing.cpu_count()
    dates = pd.date_range(config["start_date"], config["end_date"], freq="1d")
    date_str = dates.strftime("%Y/%m/%d/")

    current_day_PST = None
    for i, date in enumerate(date_str):
        s = time.time()
        current_files = sorted(glob(join(base_path, date) + "*.parquet"))

        with Pool(processes=n_workers) as pool:
            current_df = pd.concat(pool.map(load_data, current_files), ignore_index=True)

        unique_dates = current_df["datetime"].dt.date.unique()

        if len(unique_dates) != 2:
            raise ValueError(f"There are not 2 unique days of PST time: please inspect the data for {date_str[i]}.")

        previous_day_PST = current_df[current_df["datetime"].dt.date == unique_dates.min()]
        date_str_PST = previous_day_PST.iloc[0]["datetime"].strftime("%Y-%m-%d")
        file_out_path = join(out_path, f"veraset_{date_str_PST}_PST.parquet")
        if i == 0:
            # first local day is only partially covered; written as-is
            previous_day_PST.set_index(["caid", "datetime"]).sort_index().to_parquet(file_out_path)
        else:
            # combine the late part of the local day (previous UTC folder) with the early part (this folder)
            merged_days_PST = pd.concat([current_day_PST, previous_day_PST]).set_index(["caid", "datetime"]).sort_index()
            merged_days_PST.to_parquet(file_out_path)

        current_day_PST = current_df[current_df["datetime"].dt.date == unique_dates.max()]
        print(f"Wrote {file_out_path} in {np.round((time.time() - s) / 60, 2)} minutes.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, help="Path to config file")
    args = parser.parse_args()
    with open(args.config) as config_file:
        main(yaml.safe_load(config_file))
