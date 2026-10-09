"""Step 6: Add flash flood warning and MRMS exposure columns to the daily metric files.

For each date:
  FFW  - True if the device's home is inside any NWS LOX flash flood warning polygon returned by the
         IEM "cow" API for that UTC day (00Z to 00Z next day). For 30 March 2024 these are the three
         overnight warnings issued 23:43 PDT 29 March - 03:58 PDT 30 March.
  precip_max/sum, streamflow_max/sum - MRMS daily (UTC-day) 1-h QPE and FLASH max unit streamflow at the
         MRMS grid cell nearest the home (KD-tree in a Lambert conformal projection).
The FFW step writes to <metavars_path>; the MRMS step reads those files and writes to <mrms_path>.

API responses are cached as JSON under <ffw_cache_path> (the paper's responses are in paper/data/ffw/),
so the step does not depend on the API returning the same answer later.

Source: notebooks/bool_bootstrap_calculation.ipynb cells 11-17 (run by ggantos/dgagne for 2024-03-27 to
2024-04-01, outputs analysis_ready_data_20251121_metavars/ and analysis_ready_data_20251121_mrms/).

Usage (from the repo root):
    python paper/scripts/06_exposure_flags.py -c paper/config/06_exposure_flags.yaml
"""
import argparse
import json
import multiprocessing
import os
from functools import partial
from itertools import product
from multiprocessing import Pool
from os.path import join

import numpy as np
import pandas as pd
import urllib3
import xarray as xr
import yaml
from pyproj import Proj
from scipy.spatial import cKDTree
from shapely.geometry import MultiPolygon, Point

FFW_URL = ("https://mesonet.agron.iastate.edu/api/1/cow.json?wfo=LOX&begints={s}T00:00:00Z&enable_shared_border=1"
           "&endts={e}T00:00:00Z&hailsize=1&wind=58&phenomena=FF&lsrtype=FF&lsrbuffer=30&warningbuffer=1")


def get_ffw_polygons(date, cache_path):
    s_date = pd.Timestamp(date).strftime("%Y-%m-%d")
    e_date = (pd.Timestamp(date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    cache_file = join(cache_path, f"cow_LOX_FF_{s_date}.json")
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            cow = json.load(f)
    else:
        cow = json.loads(urllib3.request("GET", FFW_URL.format(s=s_date, e=e_date)).data)
        os.makedirs(cache_path, exist_ok=True)
        with open(cache_file, "w") as f:
            json.dump(cow, f)
    return [MultiPolygon(w["geometry"]["coordinates"]) for w in cow["events"]["features"]]


def get_bool(row, polys):
    point = Point(row[["lng_home", "lat_home"]])
    return any([p.contains(point) for p in polys])


def add_ffw(date, config):
    name = f"veraset_{date}_PST_chunked.parquet"
    df_temp = pd.read_parquet(join(config["analysis_ready_path"], name))
    polygons = get_ffw_polygons(date, config["ffw_cache_path"])
    df_temp["FFW"] = df_temp.apply(get_bool, polys=polygons, axis=1)
    df_temp.to_parquet(join(config["metavars_path"], name))


def create_tree(mrms_ds):
    # grid and project MRMS lat-lon coords then create KDTree
    proj = Proj(proj="lcc", R=6371229, lat_0=25, lon_0=265, lat_1=25, lat_2=25)
    mrms_lon_grid = np.tile(mrms_ds["lon"].values - 360, (mrms_ds["lat"].values.shape[0], 1)).T
    mrms_lat_grid = np.tile(mrms_ds["lat"].values, (mrms_ds["lon"].values.shape[0], 1))
    proj_mrms_lon_grid, proj_mrms_lat_grid = np.array(proj(mrms_lon_grid, mrms_lat_grid))
    tree = cKDTree(np.c_[proj_mrms_lon_grid.ravel(), proj_mrms_lat_grid.ravel()])
    return tree, proj, mrms_lon_grid.shape


def get_mrms_var(df_temp, date, idx, mrms_ds, variable, variable_name):
    lon_idx_da = xr.DataArray(idx[:, 0], dims="z")
    lat_idx_da = xr.DataArray(idx[:, 1], dims="z")
    mrms_idx = mrms_ds[variable].sel(time=date).isel(lon=lon_idx_da, lat=lat_idx_da).values
    return pd.Series(mrms_idx.tolist(), name=variable_name, index=df_temp.index)


def add_mrms(date, config):
    name = f"veraset_{date}_PST_chunked.parquet"
    variables, aggregates = config["mrms_variables"], config["mrms_aggregates"]
    year = pd.Timestamp(date).year
    mrms_file = lambda v, a: config["mrms_daily_file"].format(variable=v, year=year, aggregate=a)
    mrms_ds = xr.open_dataset(mrms_file(variables[0], aggregates[0]), engine="netcdf4")
    tree, proj, mrms_lon_grid_shape = create_tree(mrms_ds)

    df_temp = pd.read_parquet(join(config["metavars_path"], name))
    day = df_temp["date"].iloc[0].to_numpy()
    _, points = tree.query(np.column_stack(proj(df_temp["lng_home"], df_temp["lat_home"])))
    idx = np.column_stack(np.unravel_index(points, mrms_lon_grid_shape))

    for variable, aggregate in product(variables, aggregates):
        mrms_ds = xr.open_dataset(mrms_file(variable, aggregate), engine="netcdf4")
        variable_name = f"streamflow_{aggregate}" if "STREAMFLOW" in variable else f"precip_{aggregate}"
        df_temp[variable_name] = get_mrms_var(df_temp, day, idx, mrms_ds, variable, variable_name)
    df_temp.to_parquet(join(config["mrms_path"], name))


def main(config):
    for key in ("metavars_path", "mrms_path"):
        os.makedirs(config[key], exist_ok=True)
    dates = pd.date_range(config["start_date"], config["end_date"], freq="1d").strftime("%Y-%m-%d")
    for date in dates:
        add_ffw(date, config)
        print("FFW", date)
    n_workers = config["n_workers"] if config["n_workers"] > 0 else multiprocessing.cpu_count()
    with Pool(processes=min(n_workers, len(dates))) as pool:
        pool.map(partial(add_mrms, config=config), dates)
    print("MRMS done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, help="Path to config file")
    args = parser.parse_args()
    with open(args.config) as config_file:
        main(yaml.safe_load(config_file))
