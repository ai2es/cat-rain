import os
import sys
import time
import yaml
import argparse
import numpy as np
import pandas as pd
import multiprocessing
from functools import reduce
from datetime import datetime
from multiprocessing import Pool
from src.catrain.data import data_class, function_pipe, func_w_kwargs, run_parallel_uids, write_uids

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Path to config file")
    args = parser.parse_args()
    date_str = datetime.today().strftime("%Y%m%d_%H%M")
    with open(args.config) as config_file:
        config = yaml.safe_load(config_file)
    os.makedirs(config["save_out_path"], exist_ok=True)
    with open(os.path.join(config["save_out_path"], f"conf_{date_str}.yaml"), 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)
    data_class.set_attrs(config, load_key=True)  # set class attributes for file list, key_file, and metadata
    n_processors = config["n_workers"] if config["n_workers"] > 0 else multiprocessing.cpu_count()
    row_groups = data_class.row_groups
    f_kwargs = function_pipe(config)
    f = func_w_kwargs(run_parallel_uids, f_kwargs)
    s = time.time()
    with Pool(processes=n_processors) as pool:
        result = pool.map_async(f, row_groups)
        results = result.get()
    print(f"{np.round((time.time() - s) / 60, 2)} minutes to complete uid calculations.")
    s = time.time()
    results = [r for r in results if r is not None]
    combined_uids = [u for uid in results for u in uid]
    write_uids(combined_uids, config['save_out_path'], date_str)
    print(f"{np.round((time.time() - s) / 60, 2)} minutes to complete uid write.")