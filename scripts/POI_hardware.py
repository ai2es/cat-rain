# ============================================================
# POI VISITS + DWELL PIPELINE (Row-group based, Veraset chunked)
# - Runs on a list of row groups across ALL daily parquet files
# - Detects stays per device (uid/caid)
# - Spatial-joins stays to buffered POIs (50 m)
# - Aggregates to hourly + daily metrics
# - SAVES outputs (3 parquet files) + optionally attaches POI lat/lon AFTER saving
# ============================================================

import os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from multiprocessing import Pool
import pyarrow.parquet as pq
from glob import glob
from os.path import join

from skmob.core.trajectorydataframe import TrajDataFrame
from skmob.preprocessing import detection

# ======================
# CONFIG
# ======================
BASE_PATH = "/glade/campaign/cisl/aiml/cat_flood/intermediate_data/chunked_data/2024/"
POI_PATH  = "/glade/campaign/cisl/aiml/cat_flood/ancillary_data/POIs/California_filtered_pois.geojson"
COUNTY_SHP = (
    "/glade/work/jcelis/code/jcelis/2.mobility_code/"
    "Paper_figures_Code/county_shapes.shp"
)

COUNTIES = ["Los Angeles", "Ventura", "Santa Barbara"]

POI_TYPE = "hardware"      # "hardware" | "fuel" | "supermarket"
BUFFER_METERS = 50

# Time zone handling
TZ = "US/Pacific"

# Baseline window filter (optional; set to None to skip)
BASELINE_START = pd.Timestamp("2024-03-15").tz_localize(TZ)
BASELINE_END   = pd.Timestamp("2024-04-29 23:59:59").tz_localize(TZ)

# Event window (optional; set to None to skip)
EVENT_START = pd.Timestamp("2024-03-29").tz_localize(TZ)
EVENT_END   = pd.Timestamp("2024-04-01 23:59:59").tz_localize(TZ)

# Choose which window to run:
#   "event" -> uses EVENT_START/END
#   "baseline" -> uses BASELINE_START/END
#   None -> no time filtering
RUN_WINDOW = "baseline"

# Stay detection params
MIN_STOP_MINUTES = 5.0
SPATIAL_RADIUS_KM = 0.05
MAX_DWELL_MINUTES = 720.0  # cap durations to avoid crazy long stays

# Parallelism
NPROC = 50

# Output
OUT_DIR = "/glade/work/jcelis/code/jcelis/2.mobility_code/POI_analysis/POIs_output_v3"
RUN_TAG = f"{POI_TYPE}_{RUN_WINDOW or 'all'}_rg0_All"
SAVE_DIR = join(OUT_DIR, RUN_TAG)
os.makedirs(SAVE_DIR, exist_ok=True)

# ======================
# FILE LIST + METADATA
# ======================
daily_files = sorted(glob(join(BASE_PATH, "veraset*_chunked.parquet")))
if len(daily_files) == 0:
    raise FileNotFoundError(f"No files found in {BASE_PATH} matching veraset*_chunked.parquet")

meta = [pq.ParquetFile(x).metadata for x in daily_files]

# ======================
# HELPERS
# ======================
def load_counties():
    counties = gpd.read_file(COUNTY_SHP).to_crs("EPSG:4326")

    name_candidates = ["NAME", "Name", "COUNTY", "County", "county"]
    name_field = next((c for c in name_candidates if c in counties.columns), None)
    if name_field is None:
        raise ValueError(f"Could not find county name field. Columns: {list(counties.columns)}")

    counties = counties.rename(columns={name_field: "county"})
    counties = counties[counties["county"].isin(COUNTIES)][["county", "geometry"]].copy()
    return counties


def load_buffered_pois(poi_type: str, counties_gdf: gpd.GeoDataFrame, buffer_m: int = BUFFER_METERS):
    """
    Returns buffered POI polygons clipped to the 3 counties.
    Columns: poi_id, poi_type, county, geometry (polygon)
    """
    pois = gpd.read_file(POI_PATH).to_crs("EPSG:4326")
    pois = pois[pois["type"] == poi_type].copy()

    if "id" not in pois.columns:
        raise ValueError(f"POI file missing 'id'. Columns: {list(pois.columns)}")

    # Restrict POIs to the 3 counties
    pois = gpd.sjoin(pois, counties_gdf, predicate="within", how="inner").drop(columns=["index_right"])

    # Buffer in meters
    pois_3857 = pois.to_crs("EPSG:3857")
    pois_3857["geometry"] = pois_3857.buffer(buffer_m)
    pois_buf = pois_3857.to_crs("EPSG:4326")

    pois_buf = pois_buf.rename(columns={"id": "poi_id", "type": "poi_type"})
    return pois_buf[["poi_id", "poi_type", "county", "geometry"]].copy()


def _ensure_uid_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Your chunked files usually store device id as the index (caid) -> reset_index() yields 'caid' or 'index'.
    Normalize to 'uid'.
    """
    df = df.reset_index()
    if "uid" in df.columns:
        return df
    if "caid" in df.columns:
        return df.rename(columns={"caid": "uid"})
    if "index" in df.columns:
        return df.rename(columns={"index": "uid"})
    raise ValueError(f"Could not find uid/caid/index after reset_index(). Columns: {list(df.columns)[:25]}")


def normalize_points_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    - expects uid, datetime, latitude, longitude
    - timezone normalize to US/Pacific
    - apply selected window filter (optional)
    """
    if df is None or df.empty:
        return df

    required = {"uid", "datetime", "latitude", "longitude"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}. Columns: {list(df.columns)[:25]}")

    df = df.copy()

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["uid", "datetime", "latitude", "longitude"])

    # Ensure tz-aware in US/Pacific
    dt = df["datetime"]
    if getattr(dt.dt, "tz", None) is None:
        df["datetime"] = dt.dt.tz_localize(TZ, nonexistent="shift_forward", ambiguous="NaT")
    else:
        df["datetime"] = dt.dt.tz_convert(TZ)
    df = df.dropna(subset=["datetime"])

    # Window filter
    if RUN_WINDOW == "event":
        df = df[(df["datetime"] >= EVENT_START) & (df["datetime"] <= EVENT_END)]
    elif RUN_WINDOW == "baseline":
        df = df[(df["datetime"] >= BASELINE_START) & (df["datetime"] <= BASELINE_END)]

    return df


def load_row_group_all_days(rg: int, files: list) -> pd.DataFrame:
    """
    Read row-group 'rg' from ALL daily files and concat.
    """
    parts = []
    for f, m in zip(files, meta):
        try:
            tmp = pq.ParquetFile(f, metadata=m).read_row_group(rg).to_pandas()
            tmp = _ensure_uid_column(tmp)
            parts.append(tmp)
        except Exception as e:
            print(f"read error rg={rg} file={f} -> {e}")

    if not parts:
        return pd.DataFrame()

    df = pd.concat(parts, ignore_index=True)
    df = normalize_points_df(df)
    return df


def detect_stays_from_points(points_df: pd.DataFrame) -> pd.DataFrame:
    """
    Uses skmob stay detection on points_df (uid, lat, lon, datetime).
    Returns DataFrame with uid, datetime (arrival), leaving_datetime, lat, lng, duration_min.
    """
    if points_df is None or points_df.empty:
        return pd.DataFrame()

    points_df = points_df.sort_values(["uid", "datetime"])

    tdf = TrajDataFrame(
        points_df[["uid", "latitude", "longitude", "datetime"]].copy(),
        latitude="latitude",
        longitude="longitude",
        datetime="datetime",
        user_id="uid",
    )

    stdf = detection.stay_locations(
        tdf,
        stop_radius_factor=0.5,
        minutes_for_a_stop=MIN_STOP_MINUTES,
        spatial_radius_km=SPATIAL_RADIUS_KM,
        leaving_time=True
    )

    if stdf is None or len(stdf) == 0:
        return pd.DataFrame()

    stays = pd.DataFrame(stdf).copy()

    # duration minutes
    stays["datetime"] = pd.to_datetime(stays["datetime"])
    stays["leaving_datetime"] = pd.to_datetime(stays["leaving_datetime"])
    stays["duration_min"] = (stays["leaving_datetime"] - stays["datetime"]).dt.total_seconds() / 60.0

    stays = stays[(stays["duration_min"] > 0) & (stays["duration_min"] <= MAX_DWELL_MINUTES)].copy()

    # Normalize uid column name
    if "uid" not in stays.columns and "caid" in stays.columns:
        stays = stays.rename(columns={"caid": "uid"})

    # Ensure expected lat/lng names exist from skmob output
    # skmob typically outputs 'lat'/'lng'
    if "lat" not in stays.columns or "lng" not in stays.columns:
        # fallback: some versions output 'latitude'/'longitude'
        if "latitude" in stays.columns and "longitude" in stays.columns:
            stays = stays.rename(columns={"latitude": "lat", "longitude": "lng"})
        else:
            raise ValueError(f"Stay output missing lat/lng. Columns: {list(stays.columns)}")

    return stays[["uid", "datetime", "leaving_datetime", "lat", "lng", "duration_min"]].copy()


def match_stays_to_pois(stays_df: pd.DataFrame, pois_buf: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Spatial join stays points to buffered POI polygons.
    Returns: uid, datetime, leaving_datetime, duration_min, poi_id, poi_type, county
    """
    if stays_df is None or stays_df.empty:
        return pd.DataFrame()

    gdf = gpd.GeoDataFrame(
        stays_df.copy(),
        geometry=gpd.points_from_xy(stays_df["lng"], stays_df["lat"]),
        crs="EPSG:4326"
    )

    joined = gpd.sjoin(gdf, pois_buf, how="inner", predicate="within")
    if joined.empty:
        return pd.DataFrame()

    out = joined.drop(columns=["index_right", "geometry"], errors="ignore").copy()

    # avoid county_left/right artifacts: we only want POI county
    # pois_buf has 'county' so it should come through as 'county' unless conflicts happen
    if "county_right" in out.columns and "county" not in out.columns:
        out = out.rename(columns={"county_right": "county"})
    if "county_left" in out.columns:
        out = out.drop(columns=["county_left"], errors="ignore")

    keep = ["uid", "datetime", "leaving_datetime", "duration_min", "poi_id", "poi_type", "county"]
    keep = [c for c in keep if c in out.columns]
    return out[keep].copy()


def aggregate_hourly_daily(matched_df: pd.DataFrame):
    """
    Aggregates matched stays to hourly and daily frames:
    - num_devices = unique uid
    - avg_dwell / total_dwell are duration stats
    - num_stays = count of stay events
    """
    if matched_df is None or matched_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    df = matched_df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["hour"] = df["datetime"].dt.floor("H")
    df["day"]  = df["datetime"].dt.floor("D").dt.date

    group_cols_h = ["poi_id", "poi_type", "county", "hour"]
    group_cols_d = ["poi_id", "poi_type", "county", "day"]

    hourly = df.groupby(group_cols_h).agg(
        num_devices=("uid", "nunique"),
        avg_dwell=("duration_min", "mean"),
        total_dwell=("duration_min", "sum"),
        num_stays=("uid", "size"),
    ).reset_index()

    daily = df.groupby(group_cols_d).agg(
        num_devices=("uid", "nunique"),
        avg_dwell=("duration_min", "mean"),
        total_dwell=("duration_min", "sum"),
        num_stays=("uid", "size"),
    ).reset_index()

    return hourly, daily


# ======================
# WORKER + PIPELINE
# ======================
def _worker(rg: int, poi_type: str):
    try:
        points = load_row_group_all_days(rg, daily_files)
        if points is None or points.empty:
            return pd.DataFrame()

        stays = detect_stays_from_points(points)
        if stays is None or stays.empty:
            return pd.DataFrame()

        matched = match_stays_to_pois(stays, pois_buf_global)
        return matched

    except Exception as e:
        print(f"worker error rg={rg} poi_type={poi_type} -> {e}")
        return pd.DataFrame()


def run_pipeline(row_groups, poi_type="hardware", nproc=50):
    """
    Returns (matched_all, hourly_df, daily_df) for the selected row_groups.
    """
    args = [(rg, poi_type) for rg in row_groups]
    with Pool(processes=nproc) as pool:
        results = pool.starmap(_worker, args)

    matched_all = pd.concat([r for r in results if r is not None and not r.empty], ignore_index=True) \
        if any(r is not None and not r.empty for r in results) else pd.DataFrame()

    hourly_df, daily_df = aggregate_hourly_daily(matched_all)
    return matched_all, hourly_df, daily_df


# ======================
# OPTIONAL: POI LAT/LON LOOKUP (FROM ORIGINAL POI DB)
# (This is AFTER saving in this script; we add later as you asked.)
# ======================
def build_poi_lookup_from_source(poi_path: str, poi_type: str, counties_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    pois = gpd.read_file(poi_path).to_crs("EPSG:4326")
    pois = pois[pois["type"] == poi_type].copy()
    pois = gpd.sjoin(pois, counties_gdf, predicate="within", how="inner").drop(columns=["index_right"], errors="ignore")

    if "id" not in pois.columns:
        raise ValueError(f"POI file missing 'id'. Columns: {list(pois.columns)}")

    # geometry is POINT -> x/y is true lon/lat
    pois = pois.rename(columns={"id": "poi_id", "type": "poi_type"})
    pois["poi_lon"] = pois.geometry.x
    pois["poi_lat"] = pois.geometry.y

    return pois[["poi_id", "poi_type", "county", "poi_lat", "poi_lon"]].drop_duplicates()


def attach_poi_latlon(df: pd.DataFrame, poi_lookup: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    drop_cols = [c for c in ["poi_lat", "poi_lon"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df.merge(poi_lookup, on=["poi_id", "poi_type", "county"], how="left")


# ======================
# RUN
# ======================
counties_gdf = load_counties()
pois_buf_global = load_buffered_pois(POI_TYPE, counties_gdf, buffer_m=BUFFER_METERS)

print(f"POIs buffered: {len(pois_buf_global)} unique poi_id: {pois_buf_global['poi_id'].nunique()}")

# Example: first 1000 row groups
row_groups = list(range(0, 55213))

matched_all, hourly_df, daily_df = run_pipeline(row_groups, poi_type=POI_TYPE, nproc=NPROC)

print("TOTAL matched stays:", len(matched_all))
print("TOTAL unique uid:", matched_all["uid"].nunique() if not matched_all.empty else 0)
print("TOTAL unique poi_id:", matched_all["poi_id"].nunique() if not matched_all.empty else 0)
print("Hourly rows:", len(hourly_df), "Daily rows:", len(daily_df))

# ======================
# SAVE OUTPUTS (3 files)
# ======================
matched_path = join(SAVE_DIR, "matched_stays.parquet")
hourly_path  = join(SAVE_DIR, "hourly.parquet")
daily_path   = join(SAVE_DIR, "daily.parquet")

matched_all.to_parquet(matched_path, index=False)
hourly_df.to_parquet(hourly_path, index=False)
daily_df.to_parquet(daily_path, index=False)

print(" Saved 3 parquet files:")
print(" matched:", matched_path)
print(" hourly :", hourly_path)
print(" daily  :", daily_path)

# ======================
# OPTIONAL: ADD POI LAT/LON AFTER SAVING (creates *_with_poi_latlon.parquet)
# ======================
poi_lookup = build_poi_lookup_from_source(POI_PATH, POI_TYPE, counties_gdf)

matched_all2 = attach_poi_latlon(matched_all, poi_lookup)
hourly_df2   = attach_poi_latlon(hourly_df, poi_lookup)
daily_df2    = attach_poi_latlon(daily_df, poi_lookup)

matched_path2 = join(SAVE_DIR, "matched_stays_with_poi_latlon.parquet")
hourly_path2  = join(SAVE_DIR, "hourly_with_poi_latlon.parquet")
daily_path2   = join(SAVE_DIR, "daily_with_poi_latlon.parquet")

matched_all2.to_parquet(matched_path2, index=False)
hourly_df2.to_parquet(hourly_path2, index=False)
daily_df2.to_parquet(daily_path2, index=False)

print(" Saved 3 additional parquet files with POI lat/lon:")
print(" matched:", matched_path2)
print(" hourly :", hourly_path2)
print(" daily  :", daily_path2)
