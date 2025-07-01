import os
import sys
import time
import yaml
import argparse
import numpy as np
import pandas as pd
import multiprocessing
from functools import reduce
from multiprocessing import Pool

sys.path.insert(0, '..')
from src.data import data_class, function_pipe, func_w_kwargs, run_parallel_home_location, func_w_kwargs, save_summary, write_home_loc_by_row_group

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Path to config file")
    args = parser.parse_args()
    with open(args.config) as config_file:
        config = yaml.safe_load(config_file)
    os.makedirs(config["save_out_path"], exist_ok=True)
    data_class.set_attrs(config, load_key=True)  # set class attributes for file list, key_file, and metadata
    n_processors = config["n_workers"] if config["n_workers"] > 0 else multiprocessing.cpu_count()
    row_groups = data_class.row_groups
    f_kwargs = function_pipe(config)
    f = func_w_kwargs(run_parallel_home_location, f_kwargs)
    s = time.time()
    with Pool(processes=n_processors) as pool:
        result = pool.map_async(f, row_groups)
        results = result.get()
    print(f"{np.round((time.time() - s) / 60, 2)} minutes to complete home location calculations.")
    s = time.time()
    results = [r for r in results if r is not None]
    combined_data = pd.concat([r[0] for r in results])
    summary = reduce(lambda x, y: x.add(y), [r[1] for r in results])
    save_summary(summary)
    write_home_loc_by_row_group(combined_data, config['save_out_path'])
    print(f"{np.round((time.time() - s) / 60, 2)} minutes to complete home location write.")