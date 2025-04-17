import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
import os
import geopy.distance
from skmob.utils import utils, constants
from skmob.core.trajectorydataframe import TrajDataFrame
from skmob.measures import individual, collective
from skmob.preprocessing import filtering, compression, detection, clustering
from functools import partial, reduce
import geopandas as gpd
from shapely import STRtree, Point
from copy import copy
from os.path import join
# import pygeos


def calc_home_location(row_group, filters, individual_measures, home_location):
    
    tdfs = get_row_group_trajectories(row_group)
    
    home_locations = []
    for i, tdf in enumerate(tdfs):
        for filter in filters:
            tdf = filter(tdf)
        home_locations.append(home_location[0](tdf))

    tdf = reduce(lambda left, right: pd.merge(left, right, on='uid'), home_locations)
    tdf_w_geo = add_zipcode(tdf, all_matches=False)

    return tdf_w_geo


def write_home_loc_by_row_group(data, out_path):

    out_file = join(out_path, 'home_loc_by_row_group.parquet')
    if "uid" in data.index.names:
        data = data.merge(data_class.key, left_index=True, right_index=True, how="left")
    elif "uid" in data.columns:
        data = data.set_index('uid').merge(data_class.key, left_index=True, right_index=True, how="left")
    else:
        raise KeyError("UID not in index or columns.")

    schema = pa.Table.from_pandas(data).schema
    writer = pq.ParquetWriter(out_file, schema)
    for row_group in data_class.row_groups:
        row_group_data = pa.Table.from_pandas(data[data['row_group'] == row_group])
        writer.write_table(row_group_data)
    writer.close()


def add_date(tdf, date):

    tdf['date'] = date

    return tdf


def run_parallel(row_group, filters, individual_measures, home_location):
    """ Run filtering and measure calculations for a specific row group on a specific CPU worker.
    Args:
        row_group (int): The row group to run calculations on.
        filters (list): A list of filters (partial functions with kwargs) to apply to each row group.
        individual_measures (list): A list of individual measures (partial functions with kwargs) to apply to each
                                    row group.
    Returns:
        List of pandas DataFrames with merged metrics on filtered data.
    """
    tdfs = get_row_group_trajectories(row_group) ## add home data
    home = get_row_group_home(row_group)
    processed_tdfs = []
    for i, tdf in enumerate(tdfs):
        for filter in filters:
            tdf = filter(tdf)

        measure_dfs = [measure(tdf) for measure in individual_measures]
        if len(measure_dfs) > 0:
            tdf = reduce(lambda left, right: pd.merge(left, right, on='uid'), measure_dfs)
        if data_class.aggregation_type == "full":
            date_str = data_class.dates[0].strftime("%Y-%m-%d_") + data_class.dates[-1].strftime("%Y-%m-%d")
            tdf = add_date(tdf, date_str)
        else:
            tdf = add_date(tdf, data_class.dates[i])
        processed_tdfs.append(tdf)

    merged_data = pd.merge(home, pd.concat(processed_tdfs), on='uid')

    return merged_data


def function_pipe(conf):
    """ Create partials (functions with predefined kwargs) for each filter and measurement. """

    filter_funcs, ind_measure_funcs, home_location_funcs = [], [], []
    for f in conf['pipeline']:
        if f in conf["filters"]:
            kwargs = conf["filters"][f]
            filter_funcs.append(func_w_kwargs(available_funcs[f], kwargs))
        elif f in conf["individual_measures"]:
            kwargs = conf["individual_measures"][f]
            ind_measure_funcs.append(func_w_kwargs(available_funcs[f], kwargs))
        elif f in conf["home_locations"]:
            kwargs = conf["home_locations"][f]
            home_location_funcs.append(func_w_kwargs(available_funcs[f], kwargs))
        else:
            raise NameError("Function not found.")

    return {"filters": filter_funcs, "individual_measures": ind_measure_funcs, "home_location": home_location_funcs}


def ensure_list(obj):
    """Ensures the returned object is always a list of DataFrames for consistency."""
    return [obj] if isinstance(obj, pd.DataFrame) else obj


def func_w_kwargs(func, kwargs=None):
    """ Return specified funciton with static key word arguments for parallel processing. Also supports no kwargs."""
    if kwargs is not None:
        return partial(func, **kwargs)
    else:
        return partial(func)


class DataProcessor(object):
    """ Class that supports gathering all data for any number of row_groups in serial or parallel mode. Additionally,
    can calculate a wide variety of metric from a scikit-mobility Trajectory data frame.
    """
    def __init__(self, base_data_path="", home_loc_file=None, key_file="", geo_file="", data_prefix="", start_date="",
                 end_date="", meta=None, home_meta=None, key_meta=None, n_rows=None, n_row_groups=None,
                 row_group_start_end=None, aggregation_type="", save_out_path="", save_out_files="", n_workers=-1,
                 days_of_week="", pipeline=None, filters=None):

        self.base_data_path = base_data_path
        self.home_loc_file = home_loc_file
        self.key_file = key_file
        self.geo_file = geo_file
        self.data_prefix = data_prefix
        self.start_date = start_date
        self.end_date = end_date
        self.meta = meta
        self.home_meta = home_meta
        self.key_meta = key_meta
        self.n_rows = n_rows
        self.n_row_groups = n_row_groups
        self.row_group_start_end = row_group_start_end
        self.aggregation_type = aggregation_type
        self.save_out_path = save_out_path
        self.save_out_files = save_out_files
        self.n_workers = n_workers
        self.days_of_week = days_of_week
        self.pipeline = pipeline
        self.filters = filters

    def set_attrs(self, config, load_key=False):

        for key, value in config.items():
            setattr(self, key, value)

        self.data_files, self.dates = self.get_files()

        self.meta = []
        for f in self.data_files:
            with pq.ParquetFile(f) as pf:
                self.meta.append(copy(pf.metadata))
        self.key_meta = pq.ParquetFile(self.key_file).metadata
        if self.home_loc_file:
            self.home_meta = pq.ParquetFile(self.home_loc_file).metadata
        self.n_rows = self.key_meta.num_rows
        self.n_row_groups = self.key_meta.num_row_groups
        if self.row_group_start_end:
            self.row_groups = np.arange(self.row_group_start_end[0], self.row_group_start_end[1], 1)
        else:
            self.row_groups = np.arange(self.n_row_groups)

        if self.aggregation_type == "full":
            fn = f"veraset_{self.start_date}_{self.end_date}.parquet"
            self.save_out_files = os.path.join(self.save_out_path, fn)
        elif self.aggregation_type == "daily":
            self.save_out_files = [os.path.join(self.save_out_path, f"{f.split('/')[-1]}")
                                   for f in self.data_files]

        ## this should be changed to be flexible with different types of geo-files and not just zip-code
        self.geo_data = self.load_geo_file()
        self.zip_codes = self.geo_data['ZIP_CODE'].values
        self.lookup_tree = self.create_lookup_tree(self.geo_data.geometry)
        if load_key:
            self.key = self.load_key_file()

    def get_files(self):
        """
        Retrieve the file names for a specified date range. Supports specification of
        days of the week.

        Returns (list):
            List of matching file names that are confirmed to exist.
        """
        date_range = pd.date_range(start=self.start_date, end=self.end_date, freq='d')

        if self.days_of_week == "all":
            data_files = date_range.strftime(f"{self.base_data_path}{self.data_prefix}_%Y-%m-%d_PST_chunked.parquet")
        else:
            days = pd.date_range(start=self.start_date, end=self.end_date, freq=f"W-{self.days_of_week[0]}")
            for d in self.days_of_week[1:]:
                days = days.union(pd.date_range(start=self.start_date, end=self.end_date, freq=f"W-{d}"))
            data_files = days.strftime(f"{self.base_data_path}{self.data_prefix}_%Y-%m-%d_PST_chunked.parquet")

        files, dates = [], []
        for i, f in enumerate(data_files):
            if os.path.isfile(f):
                files.append(f)
                dates.append(date_range[i])

        return files, dates


    def load_key_file(self):
        """ Load key file into memory for easier user_ID lookup"""

        return pd.read_parquet(self.key_file)

    def load_geo_file(self):

        return  gpd.read_file(self.geo_file)

    def create_lookup_tree(self, geom):

        return STRtree(geom)

    def save_data_out(self, processed_data):

        if self.aggregation_type == "full":
            combined_data = pd.concat([data for data in processed_data])
            combined_data.to_parquet(self.save_out_files)
        elif self.aggregation_type == "daily":
            combined_data = [pd.concat(item[f] for item in processed_data) for f in range(len(self.data_files))]
            for data, file_name in zip(combined_data, self.save_out_files):
                data.to_parquet(file_name)
        else:
            raise ValueError("Unsupported aggregation type")
        return combined_data

###################### FILTERING ###########################
def get_row_group_home(row_group):
    """ Read a specific row group across all files. """

    home_file = data_class.home_loc_file
    home_meta = data_class.home_meta
    home_location = pq.ParquetFile(home_file, metadata=home_meta).read_row_group(row_group).to_pandas().reset_index()

    return home_location


def get_row_group_df(row_group):
    """ Read a specific row group across all files. """
    files = data_class.data_files
    meta = data_class.meta
    df_row_groups = [pq.ParquetFile(f, metadata=m).read_row_group(row_group).to_pandas() for f, m in zip(files, meta)]
    if data_class.aggregation_type == 'full':
        return ensure_list(pd.concat(df_row_groups))
    elif data_class.aggregation_type == 'daily':
        return df_row_groups
    else:
        raise ValueError("Unsupported aggregation type")


def get_row_group_trajectories(row_group):
    """ Read the row group from all files and return a skmobility Trajectory DataFrame """
    dfs = get_row_group_df(row_group)
    tdf = [TrajDataFrame(x.reset_index(), latitude='latitude', longitude='longitude', datetime='datetime',
                        user_id='caid') for x in dfs]

    # assert set(tdf[0]['uid'].values) == set(home['uid'].values)
    return tdf


##############  MISCELLANEOUS  ################

def add_zipcode(tdf, **kwargs):

    points = [Point(lon, lat) for lon, lat in zip(tdf["lng"], tdf["lat"])] ### code to create list of Points to query
    indices = data_class.lookup_tree.query_nearest(points, **kwargs)[1]
    tdf['zipcode'] = data_class.zip_codes[indices]
    return tdf


def filter_by_min_pings_per_daterange(tdf, start_date='2024-03-14', end_date='2024-04-30', min_pings=1):

    # filter by date range, inclusive    
    _tdf = tdf.copy(deep=True)
    _tdf.loc[:, constants.DATETIME] = _tdf[constants.DATETIME].dt.normalize()
    _tdf = _tdf[(_tdf[constants.DATETIME] >= start_date) & (_tdf[constants.DATETIME] <= end_date)]
    threshold = (len(_tdf[constants.DATETIME].unique()) - 1) * min_pings
    
    # if 'uid' column in not present in the TrajDataFrame
    if constants.UID not in _tdf.columns:
        _tdf = _tdf.set_index(constants.DATETIME, drop=True)
        _tdf = _tdf.loc[:, constants.LATITUDE].groupby(level=0).count()
        if len(_tdf[_tdf > 1]) > threshold:
            return tdf
    else:
        _tdf = _tdf.set_index([constants.UID, constants.DATETIME], drop=True)
        _tdf = _tdf.loc[:, constants.LATITUDE].groupby(level=[0,1]).count()
        _tdf[_tdf > 1] = 1
        _tdf = _tdf.groupby(level=0).count()
        uids = _tdf[_tdf > threshold].index
        
        return tdf[tdf[constants.UID].isin(uids)]


def filter_by_horizontal_accuracy_and_speed(tdf, max_horizontal_accuracy=800, max_speed_kmh=128.7):

    columns = [constants.LATITUDE, constants.LONGITUDE]
    distfunc = geopy.distance.distance
    
    tdf.reset_index(inplace=True)
    for i in tdf[tdf.horizontal_accuracy >= max_horizontal_accuracy].index:
        if i-1 in tdf.index and tdf.loc[i-1, "uid"] == tdf.loc[i, "uid"]:
            start = i - 1
            end = i
        elif i+1 in tdf.index and tdf.loc[i+1, "uid"] == tdf.loc[i, "uid"]:
            start = i
            end = i + 1
        else:
            continue
        
        dt = utils.diff_seconds(tdf.loc[start, constants.DATETIME], tdf.loc[end, constants.DATETIME])
        distance = distfunc(tdf.loc[start, columns].tolist(), tdf.loc[end, columns].tolist()).km
        
        if distance / dt * 3600. > max_speed_kmh:
            tdf.drop(i, inplace=True)
            
    return tdf.set_index("index", drop=True)


def filter_by_horizontal_accuracy_and_speed(tdf, max_horizontal_accuracy=800, max_speed_kmh=128.7):

    columns = [constants.LATITUDE, constants.LONGITUDE]
    distfunc = geopy.distance.distance
    
    tdf.reset_index(inplace=True)
    for i in tdf[tdf.horizontal_accuracy >= max_horizontal_accuracy].index:
        if i-1 in tdf.index and tdf.loc[i-1, "uid"] == tdf.loc[i, "uid"]:
            start = i - 1
            end = i
        elif i+1 in tdf.index and tdf.loc[i+1, "uid"] == tdf.loc[i, "uid"]:
            start = i
            end = i + 1
        else:
            continue
        
        dt = utils.diff_seconds(tdf.loc[start, constants.DATETIME], tdf.loc[end, constants.DATETIME])
        distance = distfunc(tdf.loc[start, columns].tolist(), tdf.loc[end, columns].tolist()).km
        
        if distance / dt * 3600. > max_speed_kmh:
            tdf = tdf.drop(i)
    
    return tdf.set_index("index", drop=True)


def calc_metrics(tdf_group, stats, metric_cols, group_cols):

    group, data = tdf_group
    metric_data = data[metric_cols]
    stat_funcs = [grouped_stats[s] for s in stats]
    stat_data = metric_data.agg(stat_funcs).reset_index().rename(columns={'index': 'metric'})
    for i, g in enumerate(group_cols):
        stat_data[g] = group[i]
    index = group_cols + ['metric']
    stat_data.set_index(index, inplace=True)

    return stat_data

def custom_filter(tdf, horizontal_acc, id_type):

    tdf = tdf[tdf['horizontal_accuracy'] <= horizontal_acc]
    tdf = tdf[tdf['id_type'] == id_type]

    return tdf

def home_location_tao(tdf, stop_radius_factor, minutes_for_a_stop, spatial_radius_km,
                     leaving_time, no_data_for_minutes, min_speed_kmh, cluster_radius_km,
                     min_samples):

    print("Users before home location calculation", tdf.uid.unique().shape)
    stops = detection.stay_locations(tdf,
                                     stop_radius_factor=stop_radius_factor,
                                     minutes_for_a_stop=minutes_for_a_stop,
                                     spatial_radius_km=spatial_radius_km,
                                     leaving_time=leaving_time,
                                     no_data_for_minutes=no_data_for_minutes,
                                     min_speed_kmh=min_speed_kmh)
    print("Users after stay_locations", stops.uid.unique().shape)
    if stops.shape[0] == 0:
        pass
    clusters = clustering.cluster(stops, cluster_radius_km=cluster_radius_km, min_samples=min_samples)
    clusters["duration"] = clusters["leaving_datetime"] - clusters["datetime"]
    print("Users after clustering", clusters.uid.unique().shape)

    _tdf1 = clusters.groupby(["uid", "cluster"])["duration"].sum()
    _tdf2 = clusters.groupby(["uid", "cluster"])[["lat", "lng"]].mean()
    _tdfhw = pd.concat((_tdf1.to_frame(), _tdf2), axis=1, join="inner")
    _tdfhw = _tdfhw.sort_values(by = ['uid', 'duration'], ascending = [False, False])
    _tdfhw = _tdfhw.groupby("uid").head(2)

    home = _tdfhw.groupby("uid").head(1).droplevel(1)
    home = home.rename(columns={"duration": "duration_home"})
    work = _tdfhw.groupby("uid").tail(1).droplevel(1)
    work = work.rename(columns={"lat": "lat_work", "lng": "lng_work", "duration": "duration_work"})
    print("Users before return", home.index.unique().shape)
    
    return home.merge(work, left_index=True, right_index=True)


available_funcs = {"mobility_filter": filtering.filter,
                   "mobility_filter_copy1": filtering.filter,
                   "mobility_filter_copy2": filtering.filter,
                   "pings_per_date_range_filter": filter_by_min_pings_per_daterange,
                   "horizontal_accuracy_and_speed_filter": filter_by_horizontal_accuracy_and_speed,
                   "compress": compression.compress,
                   "radius_of_gyration": individual.radius_of_gyration,
                   "home_location_skm": individual.home_location,
                   "home_location_tao": home_location_tao,
                   "jump_lengths": individual.jump_lengths,
                   "real_entropy": individual.real_entropy,
                   "uncorrelated_entropy": individual.uncorrelated_entropy,
                   "distance_straight_line": individual.distance_straight_line,
                   "mean_square_displacement": collective.mean_square_displacement,
                   "random_location_entropy": collective.random_location_entropy,
                   "uncorrelated_location_entropy": collective.uncorrelated_location_entropy,
                   "add_zipcode": add_zipcode,
                   "custom_filter": custom_filter}

grouped_stats = {"mean": "mean",
                 "median": "median",
                 "max": "max",
                 "min": "min",
                 "sum": "sum"}

# Due to serialization issues resulting in slow downs when processing in parallel, this class instance is initialized
# here and imported directly in calc_metrics.py where it can be modified.
data_class = DataProcessor()

