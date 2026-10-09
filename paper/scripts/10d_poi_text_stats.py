"""Step 10d: Numbers quoted in the supermarket results text.

Writes to <out_path>:
  poi_totals.csv              - devices, stays, POIs, dwell mean/median, by county
  saturday_comparison.csv     - baseline-Saturday mean vs storm Saturday visits and mean dwell, by county
  hourly_windows.csv          - visits and dwell by local time window, baseline vs storm, by county
  store_ffw_comparison.csv    - store-level change on the storm Saturday, inside vs outside FFW polygons
  storm_day_rain_local.csv    - county-polygon MRMS hourly mean/max by local (PDT) hour on the storm day

Definitions follow Results_inspection.ipynb (Saturday comparison) and build_store_metrics() in the map
notebook (store-level percent changes clipped to +/-100%).

Source: checks run 2026-10-08/09 (saturday_stats.py, ffw_store_stats.py, hourly_check.py).

Usage (from the repo root):
    python paper/scripts/10d_poi_text_stats.py -c paper/config/10d_poi_text_stats.yaml
"""
import argparse
import json
import os
from os.path import join

import numpy as np
import pandas as pd
import shapefile
import xarray as xr
import yaml
from matplotlib.path import Path


def load_matched(poi_dir):
    m = pd.read_parquet(join(poi_dir, "matched_stays_with_poi_latlon.parquet"))
    m["datetime"] = pd.to_datetime(m["datetime"])
    if getattr(m["datetime"].dt, "tz", None) is not None:
        m["datetime"] = m["datetime"].dt.tz_localize(None)  # local (PDT) wall time
    m["day"] = m["datetime"].dt.floor("D")
    m["hour"] = m["datetime"].dt.hour
    return m


def county_paths(shapefile_path):
    reader = shapefile.Reader(shapefile_path)
    paths = {}
    for rec, shp in zip(reader.records(), reader.shapes()):
        parts = list(shp.parts) + [len(shp.points)]
        paths[rec["NAME"]] = [Path(np.array(shp.points[parts[i]:parts[i + 1]])) for i in range(len(parts) - 1)]
    return paths


def main(config):
    out = config["out_path"]
    os.makedirs(out, exist_ok=True)
    storm = pd.Timestamp(config["storm_day"])
    m = load_matched(config["poi_dir"])
    sat = m[m["day"].dt.weekday == 5]
    groups = [("All", sat)] + [(c, sat[sat["county"] == c]) for c in config["counties"]]

    totals = [{"county": "All", "devices": m["uid"].nunique(), "stays": len(m), "pois": m["poi_id"].nunique(),
               "dwell_mean_min": m["duration_min"].mean(), "dwell_median_min": m["duration_min"].median()}]
    for c, g in m.groupby("county"):
        totals.append({"county": c, "devices": g["uid"].nunique(), "stays": len(g), "pois": g["poi_id"].nunique(),
                       "dwell_mean_min": g["duration_min"].mean(), "dwell_median_min": g["duration_min"].median()})
    pd.DataFrame(totals).round(1).to_csv(join(out, "poi_totals.csv"), index=False)

    sat_rows, win_rows = [], []
    for name, d in groups:
        base, st = d[d["day"] != storm], d[d["day"] == storm]
        nb = base["day"].nunique()
        daily = d.groupby("day")["uid"].count()
        sat_rows.append({"county": name, "n_baseline_saturdays": nb,
                         "baseline_visits_mean": daily[daily.index != storm].mean(), "storm_visits": daily.get(storm, 0),
                         "baseline_dwell_mean": base["duration_min"].mean(), "storm_dwell_mean": st["duration_min"].mean()})
        for w, (h0, h1) in config["hour_windows"].items():
            bw, sw = base[base["hour"].between(h0, h1)], st[st["hour"].between(h0, h1)]
            win_rows.append({"county": name, "window": w, "baseline_visits_per_day": len(bw) / nb,
                             "storm_visits": len(sw), "pct_visits": 100 * (len(sw) / (len(bw) / nb) - 1),
                             "baseline_dwell": bw["duration_min"].mean(), "storm_dwell": sw["duration_min"].mean(),
                             "pct_dwell": 100 * (sw["duration_min"].mean() / bw["duration_min"].mean() - 1)})
    pd.DataFrame(sat_rows).round(1).to_csv(join(out, "saturday_comparison.csv"), index=False)
    pd.DataFrame(win_rows).round(1).to_csv(join(out, "hourly_windows.csv"), index=False)

    # store-level storm-Saturday change inside vs outside FFW polygons
    with open(config["ffw_json"]) as f:
        ffws = json.load(f)
    ffw_paths = [Path(np.array(f["geometry"]["coordinates"][0][0])) for f in ffws["events"]["features"]]
    base, st = sat[sat["day"] != storm], sat[sat["day"] == storm]
    nb = base["day"].nunique()
    keys = ["poi_id", "county", "poi_lat", "poi_lon"]
    s = st.groupby(keys).agg(storm_visits=("uid", "count"), storm_dwell=("duration_min", "mean"),
                             storm_dwell_sum=("duration_min", "sum"))
    b = base.groupby(keys).agg(base_total=("uid", "count"), base_dwell=("duration_min", "mean"),
                               base_dwell_sum=("duration_min", "sum"))
    stores = pd.concat([s, b], axis=1).reset_index()
    stores["storm_visits"] = stores["storm_visits"].fillna(0)
    stores["base_visits"] = stores["base_total"].fillna(0) / nb
    stores["pct_visits"] = (100 * (stores["storm_visits"] - stores["base_visits"]) / (stores["base_visits"] + 1e-6)).clip(-100, 100)
    stores["pct_dwell"] = (100 * (stores["storm_dwell"] - stores["base_dwell"]) / (stores["base_dwell"] + 1e-6)).clip(-100, 100)
    stores["in_ffw"] = [any(p.contains_point((lo, la)) for p in ffw_paths) for lo, la in zip(stores["poi_lon"], stores["poi_lat"])]

    def summarize(g):
        return pd.Series({
            "stores": len(g), "median_pct_visits": g["pct_visits"].median(),
            "median_pct_dwell": g["pct_dwell"].median(),
            "total_pct_visits": 100 * (g["storm_visits"].sum() / g["base_visits"].sum() - 1),
            "total_pct_dwell": 100 * ((g["storm_dwell_sum"].sum() / g["storm_visits"].sum())
                                      / (g["base_dwell_sum"].sum() / g["base_total"].sum()) - 1)})
    ffw_cmp = pd.concat([stores.groupby("in_ffw").apply(summarize).assign(county="All").reset_index(),
                         stores.groupby(["county", "in_ffw"]).apply(summarize).reset_index()])
    ffw_cmp.round(2).to_csv(join(out, "store_ffw_comparison.csv"), index=False)

    # storm-day rainfall by local hour over each county polygon (MRMS 1-h QPE, UTC hour ending)
    ds = xr.open_dataset(config["mrms_hourly_file"])
    var = list(ds.data_vars)[0]
    t0 = storm + pd.Timedelta(hours=1) - pd.Timestamp(storm).tz_localize("US/Pacific").utcoffset()
    tsel = slice(t0, t0 + pd.Timedelta(hours=23))
    rain = {}
    for name, paths in county_paths(config["county_shapefile"]).items():
        pts = np.vstack([p.vertices for p in paths])
        (x0, y0), (x1, y1) = pts.min(0), pts.max(0)
        da = ds[var].sel(time=tsel, lat=slice(y0, y1), lon=slice(x0 + 360, x1 + 360)).load()
        lo, la = np.meshgrid(da["lon"].values - 360, da["lat"].values)
        xy = np.column_stack([lo.ravel(), la.ravel()])
        inside = np.zeros(len(xy), dtype=bool)
        for p in paths:
            inside |= p.contains_points(xy)
        vals = da.values[:, inside.reshape(lo.shape)]
        rain[(name, "mean_mm")], rain[(name, "max_mm")] = vals.mean(axis=1), vals.max(axis=1)
    pd.DataFrame(rain, index=pd.Index(range(24), name="local_hour_start")).round(1).to_csv(join(out, "storm_day_rain_local.csv"))
    print(pd.read_csv(join(out, "saturday_comparison.csv")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, help="Path to config file")
    args = parser.parse_args()
    with open(args.config) as config_file:
        main(yaml.safe_load(config_file))
