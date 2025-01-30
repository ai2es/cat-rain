import pandas as pd
import numpy as np
from os.path import join
from multiprocessing import Pool
import time
import argparse
import yaml
from glob import glob
from src.data import get_files, data_class, get_home_location, get_jump_lengths, get_max_distance_from_home
from src.data import func_w_kwargs, filter_tdf, compress_tdf, get_collective_random_entropy, get_radius_of_gyration
from src.data import get_real_entropy, get_uncorrelated_entropy, get_collective_uncorrelated_entropy
from src.data import get_mean_square_displacement

available_funcs = {"filter": filter_tdf,
                   "compress": compress_tdf,
                   "radius_of_gyration": get_radius_of_gyration,
                   "home_location": get_home_location,
                   "jump_lengths": get_jump_lengths,
                   "max_distance_from_home": get_max_distance_from_home,
                   "real_entropy": get_real_entropy,
                   "uncorrelated_entropy": get_uncorrelated_entropy,
                   "mean_square_displacement": get_mean_square_displacement,
                   "random_location_entropy": get_collective_random_entropy,
                   "uncorrelated_location_entropy": get_collective_uncorrelated_entropy}

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", help="Path to config file")
args = parser.parse_args()
with open(args.config) as config_file:
    config = yaml.safe_load(config_file)

base_path = config["base_data_path"]
prefix = config["prefix"]
out_path = config["save_out_path"]
key_file = config["key_file"]
# start = config["start_date"]
# end = config["end_date"]
# files = get_files(base_path, prefix, start, end, days_of_week="all")
files = sorted(glob(join(base_path, "v*.parquet")))
data_class.set_values(files, key_file) # set class attributes for file list, key_file, and metadata
data_class.parallel_agg = config["aggregation_type"]
func = available_funcs[config["function"]]
kwargs = config["kwargs"][config["function"]]

if config["row_group_start_end"]:
    row_groups = np.arange(config["row_group_start_end"][0], config["row_group_start_end"][1], 1)
else:
    row_groups = data_class.all_row_groups

f =func_w_kwargs(func, kwargs)

s = time.time()
results = []
with Pool() as pool:
    result = pool.starmap_async(f, row_groups)
    results.append(result.get())

# if data_class.parallel_agg == "full":
#
# elif data_class.parallel_agg == "daily":
#
#
data = pd.concat(results[0][0])

print(f"{np.round((time.time() - s) / 60, 2)} minutes to complete parallel computation.")
data.to_parquet(join(out_path, "home_locations_example_new.parquet"))

