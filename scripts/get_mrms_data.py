import s3fs
import pygrib
import gzip
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm
from itertools import product


def load_mrms_file(datetime, variable, grib_table, region="CONUS"):
    ts = pd.Timestamp(datetime)
    ts_date_str = ts.strftime("%Y%m%d")
    ts_datetime_str = ts.strftime("%Y%m%d-%H%M%S")
    s3 = s3fs.S3FileSystem(anon=True)
    filename = f"s3://noaa-mrms-pds/{region}/{variable}/{ts_date_str}/MRMS_{variable}_{ts_datetime_str}.grib2.gz"
    with s3.open(filename, "rb") as f:
        f_data = f.read()
    data = pygrib.fromstring(gzip.decompress(f_data))
    grib_attrs = (
        grib_table.loc[
            (grib_table["Category"] == data["parameterCategory"])
            & (grib_table["Parameter"] == data["parameterNumber"])
        ]
        .T.iloc[:, 0]
        .dropna()
        .to_dict()
    )
    xr_data = xr.DataArray(
        data=np.expand_dims(data.values[::-1], axis=0),
        coords={
            "time": [datetime],
            "lat": data.latitudes[:: data.Nx][::-1],
            "lon": data.longitudes[: data.Nx],
        },
        name=variable,
        attrs=grib_attrs,
    )
    return xr_data

def save_mrms_data(date_str, variable, variable_aggregate, grib_table_url):
    grib_table = pd.read_csv(grib_table_url)
    s_date_str, e_date_str = [s.replace("T", " ") for s in date_str.split("_")]
    dates = pd.date_range(s_date_str, e_date_str, freq="1d", inclusive="left")

    date_steps = []
    for date in tqdm(dates):
        time_steps = []
        datetimes = pd.date_range(date, date + pd.Timedelta(days=1), freq="1h", inclusive="left")
        for datetime in datetimes:
            try:
                mrms_full_step = load_mrms_file(datetime, variable, grib_table)
            except Exception as e:
                print(f"An unexpected error occurred: {e}. Skipping to next.")
                continue
            time_steps.append(
                mrms_full_step.sel(
                    lon=slice(360 - 124.7, 360 - 113.9, None), lat=slice(32.3, 42.1, None)
                )
            )
        combined_day = xr.concat(time_steps, dim="time")
        time = xr.DataArray(np.array([date]), [('time', np.array([date]))])
        if variable_aggregate == "max":
            combined_day = combined_day.max(dim="time", keep_attrs=True, skipna=True).expand_dims(time=time)
        elif variable_aggregate == "sum":
            combined_day = combined_day.sum(dim="time", keep_attrs=True, skipna=True).expand_dims(time=time)
        else:
            raise NotImplementedError("Variable aggregate method not implemented. Select 'max' or 'sum'")
        date_steps.append(combined_day)
    combined = xr.concat(date_steps, dim="time")
    combined.to_netcdf(
        f"/glade/campaign/cisl/aiml/cat_flood/mrms/raw/{variable}_{date_str}_daily_{variable_aggregate}.nc"
    )
    

date_strs = ["2024-03-15T00:00_2024-04-29T00:00", "2023-03-15T00:00_2023-04-29T00:00"]
variables = ["MultiSensor_QPE_01H_Pass2_00.00", "FLASH_HP_MAXUNITSTREAMFLOW_00.00"]
variable_aggregates = ["max", "sum"]
grib_table_url = "https://raw.githubusercontent.com/NOAA-National-Severe-Storms-Laboratory/mrms-support/refs/heads/main/GRIB2_TABLES/UserTable_MRMS_v12.2.csv"

for date_str, variable, variable_aggregate in product(date_strs, variables, variable_aggregates):
    save_mrms_data(date_str, variable, variable_aggregate, grib_table_url)
