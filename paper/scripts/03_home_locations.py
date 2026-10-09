"""Step 3: Infer a home location for every device from all 92 analysis days pooled.

For each device cohort (row group), all daily files are read and concatenated ("full" aggregation).
Stay locations are detected with scikit-mobility (>= 10 min within 200 m; gaps > 24 h end a stay),
clustered at 100 m, and the cluster with the largest cumulative dwell time is the home. No ping-level
filters are applied. Each home is assigned the nearest ZIP code polygon. Devices with no stay location
get no home (42,039,428 of 75,017,659 devices have one in the paper run).

Output: <save_out_path>/home_loc_by_row_group.parquet with index uid and columns
duration, lat, lng, zipcode, row_group.

Source: /glade/work/cbecker/cat-rain/scripts/calc_home_location.py with config/calc_home.yml and the
calc_home_location / home_location_tao functions from /glade/work/cbecker/cat-rain/src/data.py
(run 2025-10-10, output /glade/derecho/scratch/cbecker/CAT_home_loc_20251010/). That home_location_tao
returns only the home cluster; src/catrain/data.py's version also returns a work location and renames the
columns, so the home-only version is kept here to reproduce the paper's file.

Usage (from the repo root):
    python paper/scripts/03_home_locations.py -c paper/config/03_home_locations.yaml
"""
import argparse
import multiprocessing
import os
import time
from functools import partial, reduce
from multiprocessing import Pool

import numpy as np
import pandas as pd
import yaml
from skmob.preprocessing import clustering, detection

from src.catrain.data import add_zipcode, data_class, get_row_group_trajectories, write_home_loc_by_row_group


def home_location_tao_home_only(tdf, stop_radius_factor, minutes_for_a_stop, spatial_radius_km,
                                leaving_time, no_data_for_minutes, min_speed_kmh, cluster_radius_km,
                                min_samples):
    """Home = stay-location cluster with the largest cumulative dwell time (Tao et al. 2021)."""
    stops = detection.stay_locations(tdf,
                                     stop_radius_factor=stop_radius_factor,
                                     minutes_for_a_stop=minutes_for_a_stop,
                                     spatial_radius_km=spatial_radius_km,
                                     leaving_time=leaving_time,
                                     no_data_for_minutes=no_data_for_minutes,
                                     min_speed_kmh=min_speed_kmh)
    if stops.shape[0] == 0:
        return None

    clusters = clustering.cluster(stops, cluster_radius_km=cluster_radius_km, min_samples=min_samples)
    clusters["duration"] = clusters["leaving_datetime"] - clusters["datetime"]

    _tdf1 = clusters.groupby(["uid", "cluster"])["duration"].sum()
    _tdf2 = clusters.groupby(["uid", "cluster"])[["lat", "lng"]].mean()
    _tdfhw = pd.concat((_tdf1.to_frame(), _tdf2), axis=1, join="inner")
    _tdfhw = _tdfhw.sort_values(by=["uid", "duration"], ascending=[False, False])
    return _tdfhw.groupby("uid").head(1).droplevel(1)


def calc_home_location(row_group, **kwargs):
    tdfs = get_row_group_trajectories(row_group)  # one pooled trajectory frame with aggregation_type "full"
    home_locations = [home_location_tao_home_only(tdf, **kwargs) for tdf in tdfs]
    home_locations = [h for h in home_locations if h is not None]
    if len(home_locations) == 0:
        return None
    tdf = reduce(lambda left, right: pd.merge(left, right, on="uid"), home_locations)
    return add_zipcode(tdf, all_matches=False)


def main(config):
    os.makedirs(config["save_out_path"], exist_ok=True)
    with open(os.path.join(config["save_out_path"], "process_conf.yaml"), "w") as outfile:
        yaml.dump(config, outfile, default_flow_style=False)
    data_class.set_attrs(config, load_key=True)
    n_processors = config["n_workers"] if config["n_workers"] > 0 else multiprocessing.cpu_count()
    f = partial(calc_home_location, **config["home_locations"]["home_location_tao"])
    s = time.time()
    with Pool(processes=n_processors) as pool:
        results = pool.map_async(f, data_class.row_groups).get()
    all_data = pd.concat([r for r in results if r is not None])
    print(f"{np.round((time.time() - s) / 60, 2)} minutes to complete home locations.")
    s = time.time()
    write_home_loc_by_row_group(all_data, config["save_out_path"])
    print(f"{np.round((time.time() - s) / 60, 2)} minutes to complete chunked write.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, help="Path to config file")
    args = parser.parse_args()
    with open(args.config) as config_file:
        main(yaml.safe_load(config_file))
