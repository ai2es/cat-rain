import numpy as np
import multiprocessing
from multiprocessing import Pool
import time
from datetime import datetime
import argparse
import yaml
from src.data import data_class, run_parallel, function_pipe, func_w_kwargs, calc_metrics
import pandas as pd
from os.path import join
import os

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Path to config file")
    args = parser.parse_args()
    with open(args.config) as config_file:
        config = yaml.safe_load(config_file)
    os.makedirs(config["save_out_path"], exist_ok=True)
    date_str = datetime.today().strftime("%Y%m%d")
    data_class.set_attrs(config) # set class attributes for file list, key_file, and metadata
    n_processors =  config["n_workers"] if config["n_workers"] > 0 else multiprocessing.cpu_count() ## try / 2
    row_groups = data_class.row_groups
    f_kwargs = function_pipe(config)
    f = func_w_kwargs(run_parallel, f_kwargs)
    s = time.time()
    print('testing...')
    with Pool(processes=n_processors) as pool:
        result = pool.map_async(f, row_groups)
        results = result.get()
    print(f"{np.round((time.time() - s) / 60, 2)} minutes to complete parallel computation.")
    s = time.time()

    if not config['grouped_analysis']["use"]:
        results = data_class.save_data_out(results)
    else:
        combined_data = pd.concat(results)

        group_cols = config["grouped_analysis"]["temporal_agg"] + config["grouped_analysis"]["spatial_agg"]
        grouped = combined_data.groupby(group_cols)
        groups = [g for g in grouped]
        metrics = [x for x in config['pipeline'] if x in config["individual_measures"]]
        group_kwargs = {"stats": config["grouped_analysis"]["stats"], "metric_cols": metrics, 'group_cols': group_cols}
        f = func_w_kwargs(calc_metrics, group_kwargs)
        print(f"{np.round((time.time() - s) / 60, 2)} minutes to complete parallel grouping.")
        s = time.time()
        with Pool(processes=n_processors) as p:
            group_result = p.map_async(f, groups)
            group_results = group_result.get()

        all_data = pd.concat(group_results)
        file_path = f"{data_class.aggregation_type}_statistics_{date_str}.parquet"
        all_data.to_parquet(join(config['save_out_path'], file_path))
        print(f"{np.round((time.time() - s) / 60, 2)} minutes to complete group computation.")
