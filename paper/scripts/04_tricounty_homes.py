"""Step 4: Keep devices whose home ZIP code is in Los Angeles, Ventura, or Santa Barbara County.

ZIP codes per county come from config/zips_counties.yaml. A ZIP code listed under more than one county
gets the last county in the file order (LA, SB, VE); the eight such ZIP codes all end up in Ventura.
Paper run: 42,039,428 -> 11,509,012 devices (LA 10,255,396; VE 924,575; SB 329,041).

Source: /glade/work/ggantos/cat-rain/notebooks/uid_overlap.ipynb cells 1-5 (output
/glade/work/ggantos/cat-rain/notebooks/home_loc_by_row_group_counties.parquet).

Usage (from the repo root):
    python paper/scripts/04_tricounty_homes.py -c paper/config/04_tricounty_homes.yaml
"""
import argparse
import os

import pandas as pd
import yaml


def main(config):
    df = pd.read_parquet(config["home_loc_file"])
    print(f"Before filter: {df.shape[0]:,}")

    with open(config["zips_counties_file"]) as stream:
        zips_counties = yaml.safe_load(stream)
    zips = [str(z) for z in zips_counties["zips_LA"] + zips_counties["zips_SB"] + zips_counties["zips_VE"]]
    zips_dict = {}
    for k, v in zips_counties.items():
        for x in v:
            zips_dict[str(x)] = k.split("_")[1]

    df = df[df.zipcode.isin(zips)].copy()
    df["county"] = df["zipcode"].map(zips_dict)
    print(f"After county filter: {df.shape[0]:,}")
    print(df.groupby("county").size())

    os.makedirs(os.path.dirname(config["out_file"]), exist_ok=True)
    df.to_parquet(config["out_file"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, help="Path to config file")
    args = parser.parse_args()
    with open(args.config) as config_file:
        main(yaml.safe_load(config_file))
