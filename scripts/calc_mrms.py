import os
import time
import yaml
import argparse
import multiprocessing
import numpy as np
import pandas as pd
import xarray as xr
# from tqdm import tqdm
from pyproj import Proj
from os.path import join
from functools import partial
from datetime import datetime
from multiprocessing import Pool
from itertools import product
from scipy.spatial import cKDTree
import warnings
warnings.filterwarnings('ignore')


def create_tree(mrms_ds):
    # define projection
    proj = Proj(proj="lcc", R=6371229, lat_0=25, lon_0=265, lat_1=25, lat_2=25)
    
    # grid and project MRMS lat-lon coords then create KDTree
    mrms_lon_grid = np.tile(mrms_ds['lon'].values-360, (mrms_ds['lat'].values.shape[0], 1)).T
    mrms_lat_grid = np.tile(mrms_ds['lat'].values, (mrms_ds['lon'].values.shape[0], 1))
    proj_mrms_lon_grid, proj_mrms_lat_grid = np.array(proj(mrms_lon_grid, mrms_lat_grid))
    tree = cKDTree(np.c_[proj_mrms_lon_grid.ravel(), proj_mrms_lat_grid.ravel()])
    return tree, proj, mrms_lon_grid.shape

def get_mrms_var(df_sub, idx, mrms_ds, variable, variable_name):

    # grab MRMS data using calculated index
    lon_idx_da = xr.DataArray(idx[:, 0], dims="lon")
    lat_idx_da = xr.DataArray(idx[:, 1], dims="lat")
    mrms_idx = mrms_ds.isel(lon=lon_idx_da).isel(lat=lat_idx_da)[variable].values
    mrms_idx = np.swapaxes(mrms_idx, 0, 2)
    mrms_idx = mrms_idx[np.arange(0, mrms_idx.shape[0]), np.arange(0, mrms_idx.shape[1]), :]
    series = pd.Series(mrms_idx.tolist(), name=variable_name, index=df_sub.index)
    return series

def cycle_through_mrms_variables(row_group, home_loc_df, variables, date_strs, variable_aggregates):

    mrms_fn = f"/glade/campaign/cisl/aiml/cat_flood/mrms/raw/{variables[0]}_{date_strs[0]}_daily_{variable_aggregates[0]}.nc"
    mrms_ds = xr.open_dataset(mrms_fn, engine="netcdf4")
    tree, proj, mrms_lon_grid_shape = create_tree(mrms_ds)
    
    df_sub = home_loc_df[home_loc_df.row_group == row_group]
    _, points = tree.query(np.column_stack(proj(df_sub['lng_home'], df_sub['lat_home'])))
    idx = np.column_stack(np.unravel_index(points, mrms_lon_grid_shape))

    for variable, date_str, variable_aggregate in product(variables, date_strs, variable_aggregates):
        mrms_fn = f"/glade/campaign/cisl/aiml/cat_flood/mrms/raw/{variable}_{date_str}_daily_{variable_aggregate}.nc"
        mrms_ds = xr.open_dataset(mrms_fn, engine="netcdf4")
        variable_name = f"streamflow_{variable_aggregate}" if "STREAMFLOW" in variable else f"precip_{variable_aggregate}"
        series = get_mrms_var(df_sub, idx, mrms_ds, variable, variable_name)
        if variable_name in df_sub.columns:
            df_sub[variable_name] = df_sub[variable_name] + series
        else:
            df_sub = df_sub.merge(series, left_index=True, right_index=True)
    return df_sub


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Path to config file")
    args = parser.parse_args()
    with open(args.config) as config_file:
        config = yaml.safe_load(config_file)
    os.makedirs(config["save_out_path"], exist_ok=True)
    date_str = datetime.today().strftime("%Y%m%d_%H%M")
    with open(join(config["save_out_path"], f"process_conf_{date_str}.yaml"), 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    home_loc_df_23 = pd.read_parquet(config["home_loc_file_23"], engine='pyarrow')
    home_loc_df = pd.read_parquet(config["home_loc_file_24"], engine='pyarrow')
    home_loc_df = home_loc_df[home_loc_df.index.isin(home_loc_df_23.index)]
    row_groups = home_loc_df.row_group.unique()
    
    n_processors = config["n_workers"] if config["n_workers"] > 0 else multiprocessing.cpu_count()
    f = partial(cycle_through_mrms_variables,
                home_loc_df=home_loc_df,
                variables=variables,
                date_strs=date_strs,
                variable_aggregates=variable_aggregates)
    
    s = time.time()
    print('Beginning data processing...')
    with Pool(processes=n_processors) as pool:
        result = pool.map_async(f, row_groups)
        results = result.get()
    print(f"{np.round((time.time() - s) / 60, 2)} minutes to complete parallel computation.")

    s = time.time()
    combined_data = pd.concat(results)
    file_path = f"veraset_users_mrms_{date_str}.parquet"
    combined_data.to_parquet(join(config['save_out_path'], file_path))
    print(f"{np.round((time.time() - s) / 60, 2)} minutes to complete save.")
