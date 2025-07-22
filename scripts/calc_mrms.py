import os
import time
import yaml
import argparse
import multiprocessing
import numpy as np
import pandas as pd
import xarray as xr
import pyarrow.parquet as pq
from tqdm import tqdm
from pyproj import Proj
from os.path import join
from functools import partial
from datetime import datetime
from multiprocessing import Pool
from scipy.spatial import cKDTree
import warnings
warnings.filterwarnings('ignore')


def query_mrms(row, mrms_ds, variable):
    return mrms_ds.isel(lon=row.lng_idx, lat=row.lat_idx)[variable].values

def add_mrms_features(row_group, home_loc_df, mrms_ds, variable):
    """
    Find indices of nearest lon/lat pair on a grid. Supports rectilinear and curilinear grids.
    Args:
        home_loc_df (pd.DataFrame): Pandas dataframe containing lat and lon columns
        lon_grid_array (np.array): Longitude values of coarse grid you are matching against
        lat_grid_array (np.array): Latitude values of coarse grid you are matching against
    Returns (list):
        List of i, j (Lon/Lat) indices for coarse grid.
    """
    # define projection
    proj = Proj(proj="lcc", R=6371229, lat_0=25, lon_0=265, lat_1=25, lat_2=25)
    
    # grid and project MRMS lat-lon coords then create KDTree
    mrms_lon_grid = np.tile(mrms_ds['lon'].values-360, (mrms_ds['lat'].values.shape[0], 1)).T
    mrms_lat_grid = np.tile(mrms_ds['lat'].values, (mrms_ds['lon'].values.shape[0], 1))
    proj_mrms_lon_grid, proj_mrms_lat_grid = np.array(proj(mrms_lon_grid, mrms_lat_grid))
    tree = cKDTree(np.c_[proj_mrms_lon_grid.ravel(), proj_mrms_lat_grid.ravel()])
    
    # query KDTree for row_group subset of dataframe
    df_sub = home_loc_df[home_loc_df.row_group == row_group]
    _, points = tree.query(np.column_stack(proj(df_sub['lng_home'], df_sub['lat_home'])))
    idx = np.column_stack(np.unravel_index(points, mrms_lon_grid.shape))
    
    # grab MRMS data using calculated index
    lon_idx_da = xr.DataArray(idx[:, 0], dims="lon")
    lat_idx_da = xr.DataArray(idx[:, 1], dims="lat")
    mrms_idx = mrms_ds.isel(lon=lon_idx_da).isel(lat=lat_idx_da)[variable].values
    mrms_idx = np.swapaxes(mrms_idx, 0, 2)
    mrms_idx = mrms_idx[np.arange(0, mrms_idx.shape[0]), np.arange(0, mrms_idx.shape[1]), :]
    precip = pd.Series(mrms_idx.tolist(), name="precip", index=df_sub.index)
    precip_cs = pd.Series(np.cumsum(mrms_idx, axis=1).tolist(), name="precip_cumsum", index=df_sub.index)
    df_sub = df_sub.merge(precip, left_index=True, right_index=True)
    df_sub = df_sub.merge(precip_cs, left_index=True, right_index=True)
    df_sub = df_sub.assign(precip_max=lambda x: np.max(x.precip[0]),
                           precip_mean=lambda x: np.mean(x.precip[0]),
                           precip_sum=lambda x: np.sum(x.precip[0]))
    return df_sub


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Path to config file")
    args = parser.parse_args()
    with open(args.config) as config_file:
        config = yaml.safe_load(config_file)
    os.makedirs(config["save_out_path"], exist_ok=True)
    with open(join(config["save_out_path"], "process_conf.yaml"), 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    home_loc_df = pd.read_parquet(config["home_loc_file"], engine='pyarrow')
    variable = config['variable']
    mrms_file = f"{config['mrms_file_root']}{variable}_{config['s_date_str']}_{config['e_date_str']}.nc"
    mrms_ds = xr.open_dataset(mrms_file, engine="netcdf4")

    row_groups = home_loc_df.row_group.unique()
    n_processors = config["n_workers"] if config["n_workers"] > 0 else multiprocessing.cpu_count()

    s = time.time()
    f = partial(add_mrms_features, home_loc_df=home_loc_df, mrms_ds=mrms_ds, variable=variable)
    print('Beginning data processing...')
    with Pool(processes=n_processors) as pool:
        result = pool.map_async(f, row_groups)
        results = result.get()
    print(f"{np.round((time.time() - s) / 60, 2)} minutes to complete parallel computation.")

    s = time.time()
    combined_data = pd.concat(results)
    date_str = datetime.today().strftime("%Y%m%d")
    file_path = f"veraset_statistics_mrms_{date_str}.parquet"
    combined_data.to_parquet(join(config['save_out_path'], file_path))
    print(f"{np.round((time.time() - s) / 60, 2)} minutes to complete save.")
