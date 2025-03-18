import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from glob import glob
from os.path import join
import multiprocessing
from multiprocessing import Pool
from skmob.core.trajectorydataframe import TrajDataFrame
from skmob.measures import individual

# ------------------------------
# Step 1: Load Metadata & Row Group Mapping
# ------------------------------
base_path_h # Path to save folder or name from .yaml file; import yaml if using that approach.
key_file # Path to save folder or name from .yaml file

# Load key file (maps `caid` to row groups)
key_df = pd.read_parquet(key_file)  # Columns: [caid, row_group]
print(f"✅ Loaded key file with {key_df.shape[0]:,} user-row group mappings.")

# Load daily files (47 files, 1 per day)
daily_files = sorted(glob(join(base_path_h, "veraset*")))
meta = [pq.ParquetFile(x).metadata for x in daily_files]

# ------------------------------
# 📌 Step 2: Count Unique Days Per User
# ------------------------------
def count_user_days(row_group):
    """Count unique days per user within a row group."""
    df = pd.concat([
        pq.ParquetFile(f, metadata=m).read_row_group(row_group).to_pandas()
        for f, m in zip(daily_files, meta)
    ])
    
    # Extract date and count unique days per user
    df['date'] = pd.to_datetime(df['datetime']).dt.date
    user_days = df.groupby('caid')['date'].nunique().reset_index(name="num_days")
    
    return user_days

# Identify total row groups
total_row_groups = meta[0].num_row_groups
row_groups = np.arange(0, total_row_groups)
args = [(rg,) for rg in row_groups]

# Parallel Processing
if __name__ == "__main__":
    with Pool(processes=multiprocessing.cpu_count())/2 as pool:
        results = pool.starmap(count_user_days, args)

    # Combine results across row groups
    user_day_counts = pd.concat(results)

    # Aggregate across row groups (if users appear in multiple row groups)
    user_day_counts = user_day_counts.groupby('caid')["num_days"].sum().reset_index()

    # Filter users present for 7+ days
    valid_users = set(user_day_counts[user_day_counts["num_days"] >= 7]["caid"])
    print(f"✅ Found {len(valid_users):,} California residents (users present for 7+ days).")

    # Save filtered user list
    save_folder # Path to save folder or name from .yaml file
    valid_users_path = join(save_folder, "valid_california_users.csv")
    pd.DataFrame(list(valid_users), columns=["caid"]).to_csv(valid_users_path, index=False)
    
    print(f"✅ Valid user list saved to: {valid_users_path}")
