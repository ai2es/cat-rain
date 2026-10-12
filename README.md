# CAT rain

Repository for CAT demonstration project for linking mobility data and behavior before, during, and after atmospheric rivers.
The objective of this repository is to enable fast and scalable processing of large mobility datasets.

## Repo structure
- `src/` core library code
- `scripts/` runnable entry points
- `notebooks/` exploration/validation
- `config/` config files
- `paper/` pipeline that reproduces the paper's analysis, figures, and tables
- `tests/` unit tests

## Reproducing the paper
The steps that produce the paper's results, in run order, with their configs, inputs, and outputs, are
documented in [paper/README.md](paper/README.md). In brief:

1. `scripts/get_mrms_data.py`: download MRMS precipitation and streamflow.
2. `paper/scripts/01_ingest_daily_pst.py`: convert raw Veraset pings to Pacific-time daily files.
3. `paper/scripts/02_chunk_by_device.py`: group devices into cohorts stored as parquet row groups.
4. `paper/scripts/03_home_locations.py`: infer home locations from all 92 days.
5. `paper/scripts/04_tricounty_homes.py`: keep homes in Los Angeles, Ventura, and Santa Barbara Counties.
6. `scripts/calc_metrics.py -c paper/config/05_daily_metrics.yaml`: daily mobility metrics.
7. `paper/scripts/06_exposure_flags.py`: flash flood warning and MRMS exposure at each home.
8. `notebooks/ar_mobility_analysis.ipynb`: five-Saturday percentile analysis (Figs. 4-5, Table 3).
9. `paper/scripts/08_sample_construction.py`: sample-construction table (Table 1).
10. `paper/scripts/09_poi_stays.py`: supermarket visits.
11. `paper/notebooks/10a_*`, `10b_*`, `paper/scripts/10c_*`, `10d_*`: supermarket figures and statistics.
