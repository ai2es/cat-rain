import s3fs
import pygrib
import gzip
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm


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

def save_mrms_data(s_date_str, e_date_str, variable, grib_table_url):
    grib_table = pd.read_csv(grib_table_url)
    dates = pd.date_range(s_date_str, e_date_str, freq="1d")

    time_steps = []
    for date in tqdm(dates):
        mrms_full_step = load_mrms_file(date, variable, grib_table)
        time_steps.append(
            mrms_full_step.sel(
                lon=slice(360 - 124.7, 360 - 113.9, None), lat=slice(32.3, 42.1, None)
            )
        )
    combined = xr.concat(time_steps, dim="time")
    combined.to_netcdf(
        f"/glade/campaign/cisl/aiml/cat_flood/mrms/raw/{variable}_{s_date_str.replace(' ', 'T')}_{e_date_str.replace(' ', 'T')}.nc"
    )

s_date_str = "2023-03-15 00:00"
e_date_str = "2023-04-29 00:00"
# variable = "MultiSensor_QPE_01H_Pass2_00.00"
variable = "FLASH_HP_MAXUNITSTREAMFLOW_00.00"
grib_table_url = "https://raw.githubusercontent.com/NOAA-National-Severe-Storms-Laboratory/mrms-support/refs/heads/main/GRIB2_TABLES/UserTable_MRMS_v12.2.csv"

save_mrms_data(s_date_str, e_date_str, variable, grib_table_url)