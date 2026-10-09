# Paper pipeline: atmospheric river mobility analysis

Scripts, configs, and notebooks that produce the analysis, figures, and tables in the CAT-rain paper
(30 March 2024 atmospheric river, Los Angeles, Ventura, and Santa Barbara Counties) and in the response to
reviewers. Several steps originally ran from collaborators' working copies outside this repository; each
script's docstring names its source.

All commands run from the repository root with `PYTHONPATH=.` (scripts import `src.catrain`).

## Environment and data

- Environment: `conda env create -f environment.yml` (`cat`). Steps 3, 5, and 9 need scikit-mobility;
  the other steps only need pandas, pyarrow, geopandas, xarray, pyproj, pyshp, and matplotlib.
- Heavy steps run as casper CPU jobs: `paper/pbs/run_step.pbs` (scripts) and `paper/pbs/run_notebook.pbs`
  (notebooks). Pass `CONDA_ENV=<name>` with `qsub -v` to use another environment.
- Data live under `/glade/campaign/cisl/aiml/cat_flood/`. Every config reads the **archived output of the
  previous step that the paper used**, so any single step can be rerun on its own. New outputs go to
  `/glade/campaign/cisl/aiml/cat_flood/paper_pipeline/` (large data) or `paper/output/` (figures and tables,
  ignored by git), so archived data are never overwritten. For a full rerun from raw data, point each step's
  input at the previous step's `paper_pipeline/` output.
- `paper/data/` holds the cached NWS flash flood warning (FFW) polygons from the IEM COW API used in the
  paper, so results do not depend on the API.

## Run order

| Step | Command | Main input (archived) | Output | Paper use |
|---|---|---|---|---|
| 0 | `python scripts/get_mrms_data.py` | NOAA MRMS on AWS | `mrms/raw/*_daily_{max,sum}.nc` (UTC-day aggregates) | rainfall exposure |
| 1 | `paper/scripts/01_ingest_daily_pst.py -c paper/config/01_ingest_{2023,2024}.yaml` | `movement_premium/` (Veraset, UTC days) | one parquet per Pacific-time day | |
| 2 | `paper/scripts/02_chunk_by_device.py -c paper/config/02_chunk_by_device.yaml` | `intermediate_data/{2023,2024}/` | device key (75,017,659 devices) and cohort-chunked daily files (75,018 row groups) | Table 1 step 0 |
| 3 | `paper/scripts/03_home_locations.py -c paper/config/03_home_locations.yaml` | `intermediate_data/chunked_data_2023_2024/` | home location for 42,039,428 devices | Table 1 step 1 |
| 4 | `paper/scripts/04_tricounty_homes.py -c paper/config/04_tricounty_homes.yaml` | `home_locations_20251010/home_loc_by_row_group.parquet` | 11,509,012 tri-county homes | Table 1 step 2 |
| 5 | `scripts/calc_metrics.py -c paper/config/05_daily_metrics.yaml` | step 2 chunks + `/glade/work/ggantos/cat-rain/notebooks/home_loc_by_row_group_counties.parquet` | daily metrics per device and day (`analysis_ready_data_20251121/`) | Table 1 step 4 |
| 6 | `paper/scripts/06_exposure_flags.py -c paper/config/06_exposure_flags.yaml` | `analysis_ready_data_20251121/` | FFW flag and MRMS at home (`analysis_ready_data_20251121_{metavars,mrms}/`) | exposure groups |
| 7 | `notebooks/ar_mobility_analysis.ipynb` | `analysis_ready_data_20251121{,_mrms}/` | five-Saturday percentile analysis (83,503 devices) | Figs. 4-5, Table 3 |
| 8 | `paper/scripts/08_sample_construction.py -c paper/config/08_sample_construction.yaml` | steps 2-6 | sample-construction counts (CSV) | Table 1; response Tables 1-3 |
| 9 | `paper/scripts/09_poi_stays.py -c paper/config/09_poi_stays.yaml` | step 2 chunks (2024) | supermarket stays (139,041 devices, 234,703 stays) | POI results |
| 10a | `paper/notebooks/10a_poi_hourly_figures.ipynb` | step 9 output | `visits_ffw_mrms_bars_final.pdf`, `dwell_ffw_mrms_bars_final.pdf` | Figs. 6-7 |
| 10b | `paper/notebooks/10b_poi_zip_maps.ipynb` | step 9 output | ZIP map and legend PNGs | Fig. 8 panels |
| 10c | `paper/scripts/10c_build_fig9.py --fig_dir paper/output/figures` | step 10b PNGs | `Fig9_POIs_dynamics_map.pdf` | Fig. 8 |
| 10d | `paper/scripts/10d_poi_text_stats.py -c paper/config/10d_poi_text_stats.yaml` | step 9 output, MRMS hourly | numbers quoted in the supermarket results | POI text |
| 11 | `paper/scripts/11_residency_counts.py -c paper/config/11_residency_counts.yaml` | jcelis residency lists | residency-sample counts | response Table 4 |
| - | `paper/notebooks/fig1_study_area.ipynb` | Census TIGER/ACS, OSM (downloaded) | study-area map | Fig. 1 |
| - | `paper/scripts/benchmark_row_group.py -c config/process_conf.yaml` | step 2 chunks | load and processing timings | pipeline-performance table |

Figure and table numbers follow the compiled manuscript (`Fig5_travel_freq.pdf` and
`Fig6_zscore_zip_code_sat.png` are Figs. 4-5; the hourly POI figures are Figs. 6-7; `Fig9_POIs_dynamics_map.pdf`
is Fig. 8).

Approximate run times on the original hardware: step 1 about 45 min per day; step 3 about 3.3 h; step 5
several hours; step 9 about 1.5 h with 36 workers; the remaining steps take minutes.

## What each step does

1. Converts each UTC day of raw pings to Pacific-time calendar days. No pings are filtered.
2. Links the 92 analysis days (15 March-29 April of 2023 and 2024), assigns every device to a cohort of
   1,000 devices, and rewrites each day so that row group *k* holds cohort *k*.
3. Detects stay locations (>= 10 min within 200 m; gaps > 24 h end a stay) over all 92 days pooled, clusters
   them at 100 m, and takes the cluster with the largest cumulative dwell as home. Assigns the nearest ZIP code.
4. Keeps devices with homes in the ZIP codes of `config/zips_counties.yaml`. The eight ZIP codes listed under
   two counties are assigned to Ventura.
5. Computes daily metrics (number of visits, straight-line distance, jump lengths, radius of gyration, maximum
   distance from home, number of stays >= 20 min within 200 m). Measures are merged with an inner join, so a
   device has metrics on a day only if it has at least one such stay. The config lists no filters.
6. Flags homes inside FFW polygons (NWS Los Angeles/Oxnard, UTC day) and adds MRMS daily precipitation and
   streamflow at the grid cell nearest each home.
7. Keeps devices with metrics on all five Saturdays (16 March-13 April 2024), ranks each device's 30 March
   travel distance against its four baseline Saturdays (percentiles), and compares exposed and unexposed
   devices within each county (permutation and Kolmogorov-Smirnov tests).
8. Counts devices at each sample-construction step, per-Saturday coverage (including raw pings and device
   OS from the advertising ID type), and retained vs excluded device characteristics.
9. Detects stays (>= 5 min within 50 m, up to 12 h) and matches them to 50 m buffers around supermarkets.
10. Makes the POI figures and the numbers quoted in the POI text. MRMS hours are converted from UTC (hour
    ending) to PDT (hour starting), and county maxima use the county polygons.
11. Reproduces the residency-sample counts used in the response letter.

## Provenance and known gaps

- Step 5 ran on 2025-11-21 with this repository's `scripts/calc_metrics.py` and `src/catrain/data.py` as of
  commit `ced1772`; its config is saved as `analysis_ready_data_20251121/process_conf.yaml`.
- The step 3 output used in the paper was written to `/glade/derecho/scratch/cbecker/CAT_home_loc_20251010/`
  and copied unchanged (same SHA-256) to `/glade/campaign/cisl/aiml/cat_flood/home_locations_20251010/` on
  2026-10-09, with the run's config (`calc_home.yml`).
- The 45-day residency lists read in step 11 come from jcelis's `04_california_residents.py`, which read a
  2024 chunking that is no longer on disk; step 11 reproduces only the counts downstream of those lists.
- Fig. 2 (`Fig2_CAT_data_characterization_log_log.png`) has no saving code in any of the searched
  repositories or notebooks. `Fig3_workflow_chart.png` (Fig. 3) and `Fig10_7_challenges.png` (Fig. 9) are diagrams.
- `paper/scripts/benchmark_row_group.py` originally ran against an older `src/data.py`; timings may differ.
- The `summary.parquet` files written by `calc_metrics.py` double count devices across worker tasks and invert
  pings per user; do not use them for sample sizes.
