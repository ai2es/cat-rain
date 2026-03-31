# Mobility Pipeline - Points of interest (POIs) Component

## Goal
Convert raw mobility pings into POI visit + dwell-time metrics (hourly/daily), with optional geographic enrichment (ZIP/tract). Additionally, provide access to open source pois information.

## Inputs
- Mobility pings: Parquet (daily partitions; processed row-group-wise)
- POIs: GeoJSON/Parquet with lat/lon + category
- Boundaries (optional): ZIP/tract polygons for spatial join

## Outputs
- Hourly visits per POI
- Daily visits per POI
- Dwell-time stats per POI (per device aggregated)
- QA summary (counts, missingness, thresholds)

## Pipeline (4 steps)
1. Load pings (row groups)
2. Detect stays (minutes_for_a_stop, spatial_radius_km)
3. Match stays → POIs (match radius)
4. Aggregate metrics (hourly/daily) + export

## How to run
Run one day:
`python scripts/mobility_poi/run_day.py --date YYYY-MM-DD`

Backfill:
`python scripts/mobility_poi/backfill_range.py --start YYYY-MM-DD --end YYYY-MM-DD`

## Key parameters (paper defaults)

The metrics below follow the same filtering and parameters criteria used in the main section of the pipeline. 
- minutes_for_a_stop = __
- spatial_radius_km = __
- poi_match_radius_m = __
