"""Step 2: Reorganize the daily Pacific-time files into device cohorts (parquet row groups).

1. Links the analysis days of each year (15 March - 29 April) into one directory.
2. Builds a key assigning every device ID (caid) seen on any of those days to a cohort of 1,000 devices
   (one parquet row group); 75,017,659 devices -> 75,018 cohorts for the paper.
3. Rewrites each daily file so that row group k holds the pings of cohort k.

Source: /glade/work/cbecker/cat-rain/scripts/chunk_data.py (key creation and chunking were run as
separate passes; here they run in sequence and existing outputs are skipped).

Usage (from the repo root):
    python paper/scripts/02_chunk_by_device.py -c paper/config/02_chunk_by_device.yaml
"""
import argparse
import glob
import os
from functools import partial
from multiprocessing import Pool
from os.path import join

import pandas as pd
import yaml

from src.catrain.preprocess import create_chunk_key, write_daily_chunked_data_parallel


def link_analysis_days(config):
    os.makedirs(config["link_path"], exist_ok=True)
    for period in config["periods"]:
        for date in pd.date_range(period["start_date"], period["end_date"], freq="1d"):
            name = f"veraset_{date:%Y-%m-%d}_PST.parquet"
            src, dst = join(period["daily_path"], name), join(config["link_path"], name)
            if not os.path.exists(src):
                raise FileNotFoundError(src)
            if not os.path.lexists(dst):
                os.symlink(src, dst)


def main(config):
    link_analysis_days(config)
    files = sorted(glob.glob(join(config["link_path"], "*.parquet")))
    print(f"{len(files)} daily files")

    os.makedirs(os.path.dirname(config["key_file"]), exist_ok=True)
    if not os.path.exists(config["key_file"]):
        create_chunk_key(files=files, out_path=config["key_file"], n_ids_per_chunk=config["n_ids_per_row_group"])

    os.makedirs(config["chunked_path"], exist_ok=True)
    todo = [f for f in files
            if not os.path.exists(join(config["chunked_path"], f"{os.path.basename(f).split('.')[0]}_chunked.parquet"))]
    f = partial(write_daily_chunked_data_parallel, key_out_path=config["key_file"], base_out_path=config["chunked_path"])
    # one file at a time by default: each daily file is loaded fully into memory
    with Pool(processes=config.get("n_workers", 1)) as pool:
        pool.map(f, todo)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, help="Path to config file")
    args = parser.parse_args()
    with open(args.config) as config_file:
        main(yaml.safe_load(config_file))
