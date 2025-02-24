import numpy as np
import multiprocessing
from multiprocessing import Pool
import time
import argparse
import yaml
from src.data import data_class, run_parallel, function_pipe, func_w_kwargs


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", help="Path to config file")
args = parser.parse_args()
with open(args.config) as config_file:
    config = yaml.safe_load(config_file)

data_class.set_attrs(config) # set class attributes for file list, key_file, and metadata
n_processors =  config["n_workers"] if config["n_workers"] > 0 else multiprocessing.cpu_count()
row_groups = data_class.row_groups
f_kwargs = function_pipe(config)
f = func_w_kwargs(run_parallel, f_kwargs)
s = time.time()
results = []
with Pool(processes=n_processors) as pool:
    result = pool.map_async(f, row_groups)
    results = result.get()

data_class.save_data_out(results)
print(f"{np.round((time.time() - s) / 60, 2)} minutes to complete parallel computation.")

