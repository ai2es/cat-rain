import pandas as pd
import numpy as np
from glob import glob
from os.path import join
from multiprocessing import Pool
import multiprocessing
import time


base_path = # base_path to veresat data
out_path = "/glade/derecho/scratch/cbecker/CAT_daily/"
start = "2024-03-15"
end = "2024-04-30"
dates = pd.date_range(start, end, freq='1d')
date_str = dates.strftime("%Y/%m/%d/")

def load_data(file_path):

    """Loads a parquet file, converts datatypes, and converts to PST time."""

    columns = ['utc_timestamp', 'caid', 'id_type', 'latitude', 'longitude', 'horizontal_accuracy']
    df = pd.read_parquet(file_path, columns=columns)
    df = df.astype({'latitude': np.float32, 'longitude': np.float32})
    df['horizontal_accuracy'] = pd.to_numeric(df['horizontal_accuracy']).astype('int16')
    df['utc_timestamp'] = df['utc_timestamp'].dt.tz_localize('utc').dt.tz_convert('US/Pacific')
    # df['zipcode'] = [df['geo_fields'].iloc[x][2][1] for x in range(len(df))]
    # df['city'] = [df['geo_fields'].iloc[x][1][1] for x in range(len(df))]

    return df.rename(columns={'utc_timestamp': 'datetime'}).sort_values(["caid", "datetime"])


for i, date in enumerate(date_str):
    s = time.time()
    file_out_path = join(out_path, f"veraset_{dates[i].strftime('%Y-%m-%d')}_PST.parquet")
    current_files = sorted(glob(base_path + date + "*.parquet"))

    with Pool(processes=multiprocessing.cpu_count() - 8) as pool:

        current_df = pd.concat(pool.map(load_data, current_files), ignore_index=True)

    unique_days = current_df['datetime'].dt.day.unique()

    if len(unique_days) != 2:
        raise ValueError(f"There are not 2 unique days of PST time: please inspect the data for {date_str[i]}.")

    if i == 0:
        previous_day_PST = current_df[current_df['datetime'].dt.day == unique_days.min()]
        date_str_PST = previous_day_PST.iloc[0]['datetime'].strftime("%Y-%m-%d")
        file_out_path = join(out_path, f"veraset_{date_str_PST}_PST.parquet")
        previous_day_PST.set_index(["caid", "datetime"]).sort_index().to_parquet(file_out_path)

        current_day_PST = current_df[current_df['datetime'].dt.day == unique_days.max()]

        del previous_day_PST

    elif i < len(dates) - 1:

        ### get current and previous from same UTC date (in PST time)
        previous_day_PST = current_df[current_df['datetime'].dt.day == unique_days.min()]
        date_str_PST = previous_day_PST.iloc[0]['datetime'].strftime("%Y-%m-%d")
        file_out_path = join(out_path, f"veraset_{date_str_PST}_PST.parquet")
        merged_days_PST = pd.concat([current_day_PST, previous_day_PST]).set_index(['caid', 'datetime']).sort_index()
        merged_days_PST.to_parquet(file_out_path)

        current_day_PST = current_df[current_df['datetime'].dt.day == unique_days.max()]

    else:
        date_str_PST = current_day_PST.iloc[0]['datetime'].strftime("%Y-%m-%d")
        file_out_path = join(out_path, f"veraset_{date_str_PST}_PST.parquet")
        current_day_PST.set_index(["caid", "datetime"]).sort_index().to_parquet(file_out_path)

    print(f"Wrote {file_out_path} in {np.round((time.time() - s) / 60, 2)} minutes.")






