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
from haversine import haversine, Unit
from functools import partial, reduce
import geopandas as gpd
from shapely import STRtree, Point
from copy import copy
from os.path import join
from geopy.distance import distance
import plotly
import plotly.express as px
import plotly.graph_objects as go
# import pygeos


## MOVE TO PLOTTING FILE WHEN FINISHED
def plot_summary(summary, title):
    length = summary.shape[0]//3
    labels = ["Base"] + [f"Filter {i}" for i in range(length)]
    colors = px.colors.qualitative.Plotly[:length]
    mode_size = [8]*length
    line_size = [2]*length
    
    x_data = np.vstack([summary.columns[:-1]]*length)
    y_data = summary.loc[:, :, title][summary.columns[:-1]].values
    
    fig = go.Figure()
    
    for i in range(0, length):
        fig.add_trace(go.Scatter(x=x_data[i], y=y_data[i], mode='lines',
                                 name=labels[i],
                                 line=dict(color=colors[i], width=line_size[i]),
                                 connectgaps=True))
        # endpoints & min/max
        idx_max = np.argmax(y_data[i])
        idx_min = np.argmin(y_data[i]) 
        fig.add_trace(go.Scatter(x=[x_data[i][0], x_data[i][-1], x_data[i][idx_max], x_data[i][idx_min]],
                                 y=[y_data[i][0], y_data[i][-1], y_data[i][idx_max], y_data[i][idx_min]],
                                 mode='markers',
                                 marker=dict(color=colors[i], size=mode_size[i]+2)))
    
    fig.update_layout(
        xaxis=dict(showline=True,
                   showgrid=False,
                   showticklabels=True,
                   linecolor='rgb(204, 204, 204)',
                   linewidth=2,
                   ticks='outside',
                   tickfont=dict(family='Arial',
                                 size=12,
                                 color='rgb(82, 82, 82)')),
        yaxis=dict(showgrid=False,
                   zeroline=False,
                   showline=False,
                   showticklabels=False),
        autosize=False,
        margin=dict(autoexpand=False,
                    l=50,
                    r=300,
                    t=110),
        showlegend=False,
        plot_bgcolor='white')
    
    annotations = []
    for y_trace, x_trace, label, color in zip(y_data, x_data, labels, colors):
        idx_max = np.argmax(y_trace)
        idx_min = np.argmin(y_trace)    
        if isinstance(y_trace[0], int):
            ymax = f'{y_trace[idx_max]:,}'
            ymin = f'{y_trace[idx_min]:,}'
            ymean = f'{y_trace.mean():,}'
        else:
            ymax = f'{y_trace[idx_max]:.2f}'
            ymin = f'{y_trace[idx_min]:.2f}'
            ymean = f'{y_trace.mean():.2f}'
        xmax = x_trace[idx_max]
        xmin = x_trace[idx_min]
        
        annotations.append(dict(xref='paper', x=0.05, y=y_trace[0],
                                      xanchor='right', yanchor='middle',
                                      text=label,
                                      font=dict(family='Arial',
                                                size=16),
                                      showarrow=False))
        annotations.append(dict(xref='paper', x=1.05, y=ymax,
                                      xanchor='left', yanchor='middle',
                                      text=f"Max {xmax.month}/{xmax.day}: {ymax}",
                                      font=dict(family='Arial',
                                                size=16),
                                      showarrow=True))
        annotations.append(dict(xref='paper', x=1.05, y=ymin,
                                      xanchor='left', yanchor='middle',
                                      text=f"Min {xmin.month}/{xmin.day}: {ymin}",
                                      font=dict(family='Arial',
                                                size=16),
                                      showarrow=True))
        annotations.append(dict(xref='paper', x=1.05, y=ymean,
                                      xanchor='left', yanchor='middle',
                                      text=f"Avg: {ymean}",
                                      font=dict(family='Arial',
                                                size=16),
                                      showarrow=False))
    
    annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,
                            xanchor='left', yanchor='bottom',
                            text=title,
                            font=dict(family='Arial',
                                      size=30,
                                      color='rgb(37,37,37)'),
                            showarrow=False))
    fig.update_layout(annotations=annotations)
    filename = os.path.join(data_class.save_out_path, f"{title.replace(' ', '_')}.png")
    plotly.offline.plot(fig, filename=filename.replace(".png", ".html"))
    fig.write_image(filename, scale=1)
    return


###################### CREATE FILTERING SUMMARY ###########################

def create_summary(ix_funcs, ix_kwargs, conf):
    index = pd.MultiIndex.from_arrays([ix_funcs,
                                       ix_kwargs,
                                       ["Unique Users", "Total Pings", "Pings per User"]*(len(ix_funcs)//3)],
                                      names=("Filter", "Filter Arguments", "Metric"))
    date_range = pd.date_range(start=conf["start_date"], end=conf["end_date"], freq='d')
    summary = pd.DataFrame(data=0.0, index=index, columns=[dt.date() for dt in date_range])
    return summary

def data_monitor(tdf, filter, args, summary):
    date = tdf[constants.DATETIME].dt.date.iloc[0]
    pings = tdf.uid.size
    nusers = tdf.uid.unique().size
    summary.loc[(filter, args, "Total Pings"), date] += pings
    summary.loc[(filter, args, "Unique Users"), date] += nusers
    return summary

def save_summary(summary):
    summary.loc[:, "Total"] = summary.sum(axis=1)
    for i in range(0, summary.shape[0], 3):
        summary.iloc[i+2] = summary.iloc[i] / summary.iloc[i+1]
    summary.to_parquet(os.path.join(data_class.save_out_path, "summary.parquet"))
    for title in ["Unique Users", "Pings per User"]:
        plot_summary(summary, title)
    return


###################### PARALLEL PROCESSING ###########################

def run_parallel(row_group, filters, individual_measures, collective_measures, home_location, summary):
    """ Run filtering and measure calculations for a specific row group on a specific CPU worker.
    Args:
        row_group (int): The row group to run calculations on.
        filters (list): A list of filters (partial functions with kwargs) to apply to each row group.
        individual_measures (list): A list of individual measures (partial functions with kwargs) to apply to each
                                    row group.
    Returns:
        List of pandas DataFrames with merged metrics on filtered data.
    """
    tdfs = get_row_group_trajectories(row_group)
    home = get_row_group_home(row_group).rename(columns={'lng':'home_lng', 'lat':'home_lat'})
    processed_tdfs = []

    if min_pings_per_date_filter in [f.func for f in filters]:
        filter = filters[[f.func for f in filters].index(min_pings_per_date_filter)]
        uids = []
        for tdf in tdfs:
            uids.append(filter(tdf))
        uids = [u for u in uids if u != []]
        uids_int = list(set.intersection(*map(set, uids)))
    
    for i, tdf in enumerate(tdfs):
        summary = data_monitor(tdf, "None", "None", summary)
        
        if len(filters) > 0:
            for j,filter in enumerate(filters):
                if min_pings_per_date_filter in [filter.func]:
                    tdf = tdf[tdf[constants.UID].isin(uids_int)]
                elif tdf.shape[0] != 0:
                    tdf = filter(tdf)
                if tdf.shape[0] != 0:
                    filter_name = summary.index.get_level_values(0)[j*3+3]
                    filter_args = summary.index.get_level_values(1)[j*3+3]
                    summary = data_monitor(tdf, filter_name, filter_args, summary)
        
        if tdf.shape[0] == 0:
            continue
            
        if max_distance_from_home_static in [x.func for x in individual_measures]: ### Custom logic for function that requires home loc
            individual_measures = [x for x in individual_measures if x.func != max_distance_from_home_static]
            individual_measures.append(partial(max_distance_from_home_static, home_tdf=home))

        if len(individual_measures) > 0:
            measure_dfs = [measure(tdf) for measure in individual_measures]
            tdf = reduce(lambda left, right: pd.merge(left, right, on='uid'), measure_dfs)

        if data_class.aggregation_type == "full":
            date_str = data_class.dates[0].strftime("%Y-%m-%d_") + data_class.dates[-1].strftime("%Y-%m-%d")
            tdf = add_date(tdf, date_str)
        else:
            tdf = add_date(tdf, data_class.dates[i])

        processed_tdfs.append(tdf)
        
    if len(processed_tdfs) != 0:
        merged_data = pd.merge(home, pd.concat(processed_tdfs), on='uid')
        return merged_data, summary


def function_pipe(conf):
    """ Create partials (functions with predefined kwargs) for each filter and measurement. """

    filter_funcs, ind_measure_funcs, home_location_funcs, col_measure_funcs = [], [], [], []
    ix_funcs, ix_kwargs = ["None"]*3, ["None"]*3
    for f in conf['pipeline']:
        if f in conf["filters"]:
            kwargs = conf["filters"][f]
            filter_funcs.append(func_w_kwargs(available_funcs[f], kwargs))
            ix_funcs += [f]*3
            ix_kwargs += [str(kwargs)]*3
        elif f in conf["individual_measures"]:
            kwargs = conf["individual_measures"][f]
            ind_measure_funcs.append(func_w_kwargs(available_funcs[f], kwargs))
        elif f in conf["home_locations"]:
            kwargs = conf["home_locations"][f]
            home_location_funcs.append(func_w_kwargs(available_funcs[f], kwargs))
        elif f in conf["collective_measures"]:
            kwargs = conf["collective_measures"][f]
            col_measure_funcs.append(func_w_kwargs(available_funcs[f], kwargs))
        else:
            raise NameError("Function not found.")
        summary = create_summary(ix_funcs, ix_kwargs, conf)

    return {"filters": filter_funcs, "individual_measures": ind_measure_funcs, "collective_measures": col_measure_funcs, "home_location": home_location_funcs, "summary": summary}


def ensure_list(obj):
    """Ensures the returned object is always a list of DataFrames for consistency."""
    return [obj] if isinstance(obj, pd.DataFrame) else obj


def func_w_kwargs(func, kwargs=None):
    """ Return specified funciton with static key word arguments for parallel processing. Also supports no kwargs."""
    if kwargs is not None:
        return partial(func, **kwargs)
    else:
        return partial(func)


###################### DATA PROCESSOR OBJECT ###########################

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
            processed_data.to_parquet(self.save_out_files)
        elif self.aggregation_type == "daily":
        
            for date, file in zip(self.dates, self.save_out_files):
                daily_data = processed_data[processed_data['date'] == date]
                daily_data.to_parquet(file)
        else:
            raise ValueError("Unsupported aggregation type")
        return

###################### LOADING CLUSTERED DATA ###########################

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

    return tdf


###################### CUSTOM FILTERS ###########################

def min_pings_per_date_filter(tdf, start_date="2024-03-14", end_date="2024-04-30", min_pings=1):
    date = tdf[constants.DATETIME].iloc[0]
    if (date >= pd.Timestamp(start_date, tz='US/Pacific')) & (date <= pd.Timestamp(end_date, tz='US/Pacific')):
        counts = tdf[constants.UID].value_counts()
        uids = list(counts[counts >= min_pings].index)
    else:
        uids = []
    return uids


def horizontal_accuracy_and_speed_filter(tdf, max_horizontal_accuracy=800, max_speed_kmh=128.7):

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


##############  UTILS  #################################################

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


def add_date(tdf, date):

    tdf['date'] = date
    return tdf


def add_zipcode(tdf, **kwargs):

    points = [Point(lon, lat) for lon, lat in zip(tdf["lng"], tdf["lat"])] ### code to create list of Points to query
    indices = data_class.lookup_tree.query_nearest(points, **kwargs)[1]
    tdf['zipcode'] = data_class.zip_codes[indices]
    return tdf


##############  HOME LOCATION  #################################################

def calc_home_location(row_group, **kwargs):

    tdfs = get_row_group_trajectories(row_group)
    f = available_funcs['home_location']
    home_locations = [f(tdf, **kwargs) for tdf in tdfs]
    tdf = reduce(lambda left, right: pd.merge(left, right, on='uid'), home_locations)
    tdf_w_geo = add_zipcode(tdf, all_matches=False)

    return tdf_w_geo


def home_location_tao(tdf, stop_radius_factor, minutes_for_a_stop, spatial_radius_km,
                     leaving_time, no_data_for_minutes, min_speed_kmh, cluster_radius_km,
                     min_samples):

    stops = detection.stay_locations(tdf,
                                     stop_radius_factor=stop_radius_factor,
                                     minutes_for_a_stop=minutes_for_a_stop,
                                     spatial_radius_km=spatial_radius_km,
                                     leaving_time=leaving_time,
                                     no_data_for_minutes=no_data_for_minutes,
                                     min_speed_kmh=min_speed_kmh)
    if stops.shape[0] == 0:
        pass
    clusters = clustering.cluster(stops, cluster_radius_km=cluster_radius_km, min_samples=min_samples)
    clusters["duration"] = clusters["leaving_datetime"] - clusters["datetime"]

    _tdf1 = clusters.groupby(["uid", "cluster"])["duration"].sum()
    _tdf2 = clusters.groupby(["uid", "cluster"])[["lat", "lng"]].mean()
    _tdfhw = pd.concat((_tdf1.to_frame(), _tdf2), axis=1, join="inner")
    _tdfhw = _tdfhw.sort_values(by = ['uid', 'duration'], ascending = [False, False])
    _tdfhw = _tdfhw.groupby("uid").head(2)

    home = _tdfhw.groupby("uid").head(1).droplevel(1)
    home = home.rename(columns={"duration": "duration_home"})
    work = _tdfhw.groupby("uid").tail(1).droplevel(1)
    work = work.rename(columns={"lat": "lat_work", "lng": "lng_work", "duration": "duration_work"})
    
    return home.merge(work, left_index=True, right_index=True)


def write_home_loc_by_row_group(data, out_path):

    out_file = join(out_path, 'home_loc_by_row_group.parquet')
    data = data.set_index('uid').merge(data_class.key, left_index=True, right_index=True, how="left")
    schema = pa.Table.from_pandas(data).schema
    writer = pq.ParquetWriter(out_file, schema)
    for row_group in data_class.row_groups:
        row_group_data = pa.Table.from_pandas(data[data['row_group'] == row_group])
        writer.write_table(row_group_data)
    writer.close()


##############  CUSTOM METRICS  #################################################

def _max_distance_from_home_static_individual(df):

    df_samp = df.iloc[0:1]
    df_samp['max_distance_from_home'] = 0
    reference = (df_samp['home_lat'].values[0], df_samp['home_lng'].values[0])
    points = list(zip(df[constants.LATITUDE].values, df[constants.LONGITUDE].values))
    max_dist = 0
    for point in points:
        dist = distance(reference, point).km
        if dist > max_dist:
            max_dist = dist
        df_samp['max_distance_from_home'] = max_dist
    return df_samp[['max_distance_from_home']]


def max_distance_from_home_static(tdf, home_tdf, **kwargs):
    df_combined = tdf.merge(home_tdf, how='inner', on=constants.UID)
    return df_combined.groupby(constants.UID).apply(_max_distance_from_home_static_individual)


def stay_locations_count(tdf, **kwargs):
    tdf = detection.stay_locations(tdf, **kwargs)
    tdf = pd.DataFrame(tdf.groupby(constants.UID).size())
    return tdf.reset_index().rename(columns={0: "num_stops"})
    

available_funcs = {"mobility_filter": filtering.filter,
                   "min_pings_per_date_filter": min_pings_per_date_filter,
                   "horizontal_accuracy_and_speed_filter": horizontal_accuracy_and_speed_filter,
                   "compress": compression.compress,
                   "home_location_skm": individual.home_location,
                   "home_location_tao": home_location_tao,
                   "number_of_visits": individual.number_of_visits,
                   "distance_straight_line": individual.distance_straight_line,
                   "jump_lengths": individual.jump_lengths,
                   "radius_of_gyration": individual.radius_of_gyration,
                   "waiting_times": individual.waiting_times,
                   "real_entropy": individual.real_entropy,
                   "uncorrelated_entropy": individual.uncorrelated_entropy,
                   "max_distance_from_home_static": max_distance_from_home_static,
                   "stay_locations_count": stay_locations_count,
                   "add_zipcode": add_zipcode,
                   "mean_square_displacement": collective.mean_square_displacement,
                   "random_location_entropy": collective.random_location_entropy,
                   "uncorrelated_location_entropy": collective.uncorrelated_location_entropy}

grouped_stats = {"mean": "mean",
                 "median": "median",
                 "max": "max",
                 "min": "min",
                 "sum": "sum"}

# Due to serialization issues resulting in slow downs when processing in parallel, this class instance is initialized
# here and imported directly in calc_metrics.py where it can be modified.
data_class = DataProcessor()
