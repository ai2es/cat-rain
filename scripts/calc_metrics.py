import pandas as pd
import numpy as np
from os.path import join
from multiprocessing import Pool
import time
from glob import glob
from src.data import get_files, data_class, get_home_location, get_jump_lengths, get_max_distance_from_home

# base_path = "<base path to veresat data>"
base_path = "/glade/derecho/scratch/cbecker/CAT_daily/chunked_data/"
prefix = 'veraset'
# out_path = "<output path to intermediate data>"
out_path = "/glade/derecho/scratch/cbecker/"
# key_file = "<path to key file>"
key_file = "/glade/derecho/scratch/cbecker/CAT_daily/chunked_data/id_row_group_key.parquet"


# start = "2024-03-15"
# end = "2024-03-30"
# files = get_files(base_path, prefix, start, end, days_of_week="all")
files = sorted(glob(join(base_path, "v*.parquet")))

data_class.set_values(files, key_file)
row_groups = data_class.all_row_groups # grab the first thousand row groups (1 million users)

s = time.time()
results = []
with Pool() as pool:
    result = pool.map_async(get_home_location, row_groups)
    results.append(result.get())
data = pd.concat(results[0])
print(f"{np.round((time.time() - s) / 60, 2)} minutes to complete parallel computation.")
data.to_parquet(join(out_path, "home_locations_example.parquet"))

