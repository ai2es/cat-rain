import pandas as pd
import numpy as np
from glob import glob
from os.path import join
from multiprocessing import Pool
import time, multiprocessing


base_path = "/glade/derecho/scratch/dgagne/veraset_cat/"
out_path = "/glade/derecho/scratch/ggantos/"
start = "2024-03-20"
end = "2024-03-31"
dates = pd.date_range(start, end, freq='1d')
date_str = dates.strftime("%Y/%m/%d/")

def load_data(file_path):
    """Loads a parquet file, converts datatypes, and converts to PST time."""

    columns = ['utc_timestamp', 'caid', 'id_type', 'latitude', 'longitude', 'horizontal_accuracy', 'geo_fields']
    df = pd.read_parquet(file_path, columns=columns)
    df = df.astype({'latitude': np.float32, 'longitude': np.float32})
    df['horizontal_accuracy'] = pd.to_numeric(df['horizontal_accuracy']).astype('int16')
    df['utc_timestamp'] = df['utc_timestamp'].dt.tz_localize('utc').dt.tz_convert('US/Pacific')
    df['zipcode'] = [df['geo_fields'].iloc[x][2][1] for x in range(len(df))]
    df['city'] = [df['geo_fields'].iloc[x][1][1] for x in range(len(df))]
    df.drop('geo_fields', axis=1, inplace=True)

    return df.rename(columns={'utc_timestamp': 'datetime'}).sort_values(["caid", "datetime"])



for i, date in enumerate(date_str):

    t0 = time.time()
    file_out_path = join(out_path, f"veraset_{dates[i].strftime('%Y-%m-%d')}_PST.parquet")
    current_files = sorted(glob(base_path + date + "*.parquet"))

    with Pool(processes=multiprocessing.cpu_count() - 8) as pool:

        current_df = pd.concat(pool.map(load_data, current_files), ignore_index=True)

    unique_days = current_df['datetime'].dt.day.unique()

    if len(unique_days) != 2:
        raise ValueError(f"There are not 2 unique days of PST time: please inspect the data for {date_str[i]}.")

    if i == 0:
        previous_day_PST = current_df[current_df['datetime'].dt.day == unique_days.min()].sort_values(
            ["caid", "datetime"])
        previous_day_PST.to_parquet(file_out_path)

        current_day_PST = current_df[current_df['datetime'].dt.day == unique_days.max()]

        del previous_day_PST

    elif i < len(dates) - 1:

        ### get current and previous from same UTC date
        previous_day_PST = current_df[current_df['datetime'].dt.day == unique_days.min()]
        merged_days_PST = pd.concat([current_day_PST, previous_day_PST])
        sorted_data = merged_days_PST.groupby('caid').apply(lambda x: x.sort_values(by="datetime"),
                                                            include_groups=False).droplevel(1)
        sorted_data.to_parquet(file_out_path)

        current_day_PST = current_df[current_df['datetime'].dt.day == unique_days.max()]

    else:

        current_day_PST.sort_values(["caid", "datetime"]).to_parquet(file_out_path)

    print(f"Wrote {file_out_path}")
    t1 = time.time()
    delta = t1-t0
    print(f"TIME PER DAY: {delta // 3600 % 24:02.0f}:{delta // 60 % 60:02.0f}:{delta % 60:02.0f}")
