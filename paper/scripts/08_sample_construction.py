"""Step 8: Sample-construction counts for paper Table 1 and response-letter Tables 2-3.

Writes to <out_path>:
  funnel_by_county.csv       - steps 2-5 of Table 1 by county, plus Saturday-presence counts
  exposure_by_county.csv     - exposure groups within the five-Saturday sample (Table 1)
  daily_coverage.csv         - per Saturday: tri-county devices with >=1 raw ping vs >=1 stay, iOS share,
                               median pings (response Table 2; reads the raw chunked pings)
  retained_vs_excluded.csv   - devices with a stay on the event day, retained vs excluded (response Table 3)
Totals for steps 0-1 come from parquet metadata (key file rows, home-location rows).

Source: checks run for the reviewer response on 2026-10-07 (funnel.py, raw_scan.py).

Usage (from the repo root; the raw scan needs ~64 GB, run on casper):
    python paper/scripts/08_sample_construction.py -c paper/config/08_sample_construction.yaml
"""
import argparse
import os
from os.path import join

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml


def read_day(config, day, cols):
    f = join(config["analysis_ready_path"], f"veraset_{day}_PST_chunked.parquet")
    return pd.read_parquet(f, columns=cols).set_index("uid")


def main(config):
    out = config["out_path"]
    os.makedirs(out, exist_ok=True)
    sats, event = config["saturdays"], config["event_day"]
    baseline = [d for d in sats if d != event]

    # steps 0-2
    totals = {
        "devices_in_extract": pq.ParquetFile(config["key_file"]).metadata.num_rows,
        "home_location_inferred": pq.ParquetFile(config["home_loc_file"]).metadata.num_rows,
    }
    home = pd.read_parquet(config["home_loc_counties_file"], columns=["county", "duration"])
    rows = {"home_zip_in_study_counties": home["county"].value_counts()}

    # steps 4-5 from the analysis-ready daily files
    cols = ["uid", "county", "zipcode", "number_of_visits", "distance_straight_line", "num_stops", "duration"]
    days = {d: read_day(config, d, cols) for d in sats}
    present = pd.DataFrame({d: pd.Series(True, index=days[d].index) for d in sats}).fillna(False).astype(bool)
    present["n_sat"] = present[sats].sum(axis=1)
    full = present.index[present["n_sat"] == len(sats)]
    rows["stay_on_event_day"] = days[event]["county"].value_counts()
    rows["stay_on_all_saturdays"] = days[event].loc[full, "county"].value_counts()

    # step 3 and Table 2 from the raw chunked pings
    coverage, raw_event = [], None
    for d in sats:
        t = pq.read_table(join(config["chunked_path"], f"veraset_{d}_PST_chunked.parquet"), columns=["caid", "id_type"])
        g = t.group_by(["caid", "id_type"]).aggregate([("caid", "count")]).to_pandas()
        g.columns = ["caid", "id_type", "pings"]
        g = g.drop_duplicates("caid").set_index("caid").join(home[["county"]], how="inner")
        g["kept"] = g.index.isin(days[d].index)
        ios = lambda x: (x["id_type"] == "idfa").mean() * 100
        coverage.append({"day": d, "devices_with_ping": len(g), "devices_with_stay": int(g["kept"].sum()),
                         "kept_pct": 100 * g["kept"].mean(), "ios_pct_kept": ios(g[g["kept"]]),
                         "ios_pct_dropped": ios(g[~g["kept"]]), "median_pings_kept": g.loc[g["kept"], "pings"].median(),
                         "median_pings_dropped": g.loc[~g["kept"], "pings"].median()})
        if d == event:
            raw_event = g
            rows["ping_on_event_day"] = g["county"].value_counts()
        del t
    pd.DataFrame(coverage).round(1).to_csv(join(out, "daily_coverage.csv"), index=False)

    funnel = pd.DataFrame(rows).T[["LA", "VE", "SB"]]
    funnel["Total"] = funnel.sum(axis=1)
    funnel = funnel.loc[["home_zip_in_study_counties", "ping_on_event_day", "stay_on_event_day", "stay_on_all_saturdays"]]
    for k, v in totals.items():
        funnel.loc[k, "Total"] = v
    funnel.to_csv(join(out, "funnel_by_county.csv"))
    present["n_sat"].value_counts().sort_index().rename("devices").to_csv(join(out, "saturday_presence.csv"))
    print(funnel)

    # exposure groups (Table 1)
    exp = pd.read_parquet(join(config["mrms_path"], f"veraset_{event}_PST_chunked.parquet"),
                          columns=["uid", "FFW", "precip_sum", "county"]).set_index("uid").loc[full]
    groups = {"FFW": exp["FFW"]}
    for thr in config["precip_thresholds_mm"]:
        groups[f"precip_ge_{thr}mm"] = exp["precip_sum"] >= thr
    pd.DataFrame({k: v.groupby(exp["county"]).sum() for k, v in groups.items()}).T.to_csv(join(out, "exposure_by_county.csv"))

    # retained vs excluded among devices with a stay on the event day (Table 3)
    a = days[event].join(pd.read_parquet(join(config["mrms_path"], f"veraset_{event}_PST_chunked.parquet"),
                                         columns=["uid", "FFW", "precip_sum"]).set_index("uid"))
    a["retained"] = a.index.isin(full)
    a["home_dwell_days"] = a["duration"].dt.total_seconds() / 86400
    base_dist = pd.DataFrame({d: days[d]["distance_straight_line"] for d in baseline}).mean(axis=1)
    a["baseline_mean_distance_km"] = base_dist.reindex(a.index)
    a["ios"] = raw_event["id_type"].reindex(a.index) == "idfa"
    summary = a.groupby("retained").agg(
        devices=("county", "size"), pings_median=("number_of_visits", "median"), pings_mean=("number_of_visits", "mean"),
        stays_median=("num_stops", "median"), home_dwell_days_median=("home_dwell_days", "median"),
        ios_pct=("ios", lambda x: 100 * x.mean()), distance_event_median_km=("distance_straight_line", "median"),
        baseline_distance_median_km=("baseline_mean_distance_km", "median"),
        ffw_pct=("FFW", lambda x: 100 * x.mean()), precip_median_mm=("precip_sum", "median")).T
    county_pct = 100 * pd.crosstab(a["county"], a["retained"], normalize="columns")
    pd.concat([summary, county_pct.add_prefix("county_pct_")]).round(2).to_csv(join(out, "retained_vs_excluded.csv"))
    print(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, help="Path to config file")
    args = parser.parse_args()
    with open(args.config) as config_file:
        main(yaml.safe_load(config_file))
