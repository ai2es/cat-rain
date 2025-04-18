import pandas as pd
import numpy as np
from glob import glob
from os.path import join
import multiprocessing
from multiprocessing import Pool
import pyarrow.parquet as pq
from haversine import haversine, Unit
from skmob.core.trajectorydataframe import TrajDataFrame
from skmob.preprocessing import detection

# === Paths and setup ===
base_path_h = "/glade/derecho/scratch/jcelis/mobility/chunked_data"
daily_files = sorted(glob(join(base_path_h, "veraset*")))
meta = [pq.ParquetFile(x).metadata for x in daily_files]
output_folder = "/glade/work/jcelis/code/jcelis/2.mobility_code/Modular_approach/output_test/Daily_metrics"

# === Optional: Select a subset of row groups ===
# Leave as None to process ALL row groups
start_rg = None  # e.g., 0
end_rg = None    # e.g., 10

# === Utility functions ===
def get_row_groups(row_group):
    return pd.concat([
        pq.ParquetFile(f, metadata=m).read_row_group(row_group).to_pandas()
        for f, m in zip(daily_files, meta)
    ])

def get_row_group_trajectories(row_group):
    df = get_row_groups(row_group).reset_index()
    return TrajDataFrame(df, latitude='latitude', longitude='longitude', datetime='datetime', user_id='caid')

# === A. Total stops per user ===
def get_stop_counts(row_group):
    tdf = get_row_group_trajectories(row_group)
    stdf = detection.stay_locations(
        tdf, stop_radius_factor=0.5, minutes_for_a_stop=20.0,
        spatial_radius_km=0.2, leaving_time=True
    )
    return stdf.groupby("uid").size().reset_index(name="num_stops")

# === B. Distance per user per day ===
def get_daily_distance(row_group):
    df = get_row_groups(row_group).reset_index()
    df = df.sort_values(by=["caid", "datetime"])
    df["date"] = df["datetime"].dt.date
    df["lat_prev"] = df.groupby("caid")["latitude"].shift()
    df["lon_prev"] = df.groupby("caid")["longitude"].shift()
    df = df.dropna(subset=["lat_prev", "lon_prev"])
    df["distance"] = df.apply(lambda r: haversine(
        (r["lat_prev"], r["lon_prev"]), (r["latitude"], r["longitude"]), unit=Unit.MILES), axis=1)
    result = df.groupby(["caid", "date"])["distance"].sum().reset_index()
    return result.rename(columns={"caid": "uid", "distance": "distance_miles"})

# === C. Stops per user per day ===
def get_daily_stops(row_group):
    df = get_row_groups(row_group).reset_index()
    tdf = TrajDataFrame(df, latitude='latitude', longitude='longitude', datetime='datetime', user_id='caid')
    stdf = detection.stay_locations(
        tdf, stop_radius_factor=0.5, minutes_for_a_stop=20.0,
        spatial_radius_km=0.2, leaving_time=True
    )
    stdf["date"] = stdf["datetime"].dt.date
    return stdf.groupby(["uid", "date"]).size().reset_index(name="num_stops")

# === Run the script ===
if __name__ == "__main__":
    n_processors = 64
    #n_processors = int(multiprocessing.cpu_count() / 2)
    total_row_groups = meta[0].num_row_groups

    if start_rg is not None and end_rg is not None:
        selected_row_groups = np.arange(start_rg, end_rg)
    else:
        selected_row_groups = np.arange(0, total_row_groups)

    args = [(rg,) for rg in selected_row_groups]

    with Pool(processes=n_processors) as pool:
        # Output 1: Total number of stops
        stop_counts = pd.concat(pool.starmap(get_stop_counts, args))
        stop_counts.to_parquet(join(output_folder, "output1_total_stops_full.parquet"), index=False)
        stop_counts.to_csv(join(output_folder, "output1_total_stops_full.csv"), index=False)

        # Output 2: Total distance by date
        distance_by_day = pd.concat(pool.starmap(get_daily_distance, args))
        distance_by_day.to_parquet(join(output_folder, "output2_total_distance_by_day_full.parquet"), index=False)
        distance_by_day.to_csv(join(output_folder, "output2_total_distance_by_day_full.csv"), index=False)

        # Output 3: Total number of stops per day
        stops_by_day = pd.concat(pool.starmap(get_daily_stops, args))
        stops_by_day.to_parquet(join(output_folder, "output3_total_stops_by_day_full.parquet"), index=False)
        stops_by_day.to_csv(join(output_folder, "output3_total_stops_by_day_full.csv"), index=False)

        # Output 4: Combined distance + stops
        combined = pd.merge(distance_by_day, stops_by_day, how='outer', on=['uid', 'date'])
        combined["distance_miles"] = combined["distance_miles"].fillna(0)
        combined["num_stops"] = combined["num_stops"].fillna(0).astype(int)
        combined.to_parquet(join(output_folder, "output4_combined_distance_stops_full.parquet"), index=False)
        combined.to_csv(join(output_folder, "output4_combined_distance_stops_full.csv"), index=False)
