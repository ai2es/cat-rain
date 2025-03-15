import numpy as np
import multiprocessing
from multiprocessing import Pool
import time
import argparse
import yaml
from src.data import data_class,func_w_kwargs, write_home_loc_by_row_group, calc_home_location
import pandas as pd

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Path to config file")
    args = parser.parse_args()
    with open(args.config) as config_file:
        config = yaml.safe_load(config_file)
    data_class.set_attrs(config, load_key=True)  # set class attributes for file list, key_file, and metadata
    n_processors = config["n_workers"] if config["n_workers"] > 0 else multiprocessing.cpu_count()
    row_groups = data_class.row_groups
    f_kwargs = config['home_location_kwargs']
    f = func_w_kwargs(calc_home_location, f_kwargs)
    s = time.time()
    with Pool(processes=n_processors) as pool:
        result = pool.map_async(f, row_groups)
        results = result.get()
    all_data = pd.concat(results)
    print(f"{np.round((time.time() - s) / 60, 2)} minutes to complete home locations.")
    s = time.time()
    write_home_loc_by_row_group(all_data, config['save_out_path'])
    print(f"{np.round((time.time() - s) / 60, 2)} minutes to complete chunked write.")