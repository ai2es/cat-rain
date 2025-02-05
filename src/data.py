import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import os
from skmob.core.trajectorydataframe import TrajDataFrame
from skmob.measures import individual, collective
from skmob.preprocessing import filtering, compression
from functools import partial, reduce


def run_parallel(row_group, filters, individual_measures):
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
    for filter in filters:
        tdfs = filter(tdfs)

    processed_meassures = []
    for tdf in tdfs:
        measure_dfs = [measure(ensure_list(tdf))[0] for measure in individual_measures]
        merged_df = reduce(lambda left, right: pd.merge(left, right, on='uid'), measure_dfs)
        processed_meassures.append(merged_df)

    return processed_meassures


def function_pipe(conf):
    """ Create partials (functions with predefined kwargs) for each filter and measurement. """

    filter_funcs, ind_measure_funcs = [], []
    for f in conf['pipeline']:
        if f in conf["filters"]:
            kwargs = conf["filters"][f]
            filter_funcs.append(func_w_kwargs(available_funcs[f], kwargs))
        elif f in conf["individual_measures"]:
            kwargs = conf["individual_measures"][f]
            ind_measure_funcs.append(func_w_kwargs(available_funcs[f], kwargs))
        else:
            raise NameError("Function not found.")

    return {"filters": filter_funcs, "individual_measures": ind_measure_funcs}


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
    def __init__(self, base_data_path="", key_file="", data_prefix="", start_date="", end_date="", meta=None,
                 key_meta=None, n_rows=None, n_row_groups=None, row_group_start_end=None, aggregation_type="",
                 save_out_path="", save_out_files="", n_workers=-1, days_of_week="", pipeline=None, filters=None):

        self.base_data_path = base_data_path
        self.key_file = key_file
        self.data_prefix = data_prefix
        self.start_date = start_date
        self.end_date = end_date
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

    def set_attrs(self, config):

        for key, value in config.items():
            setattr(self, key, value)

        self.data_files = self.get_files()
        self.meta = [pq.ParquetFile(x).metadata for x in self.data_files]
        self.key_meta = pq.ParquetFile(self.key_file).metadata
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

        return [file for file in data_files if os.path.isfile(file)]


    def load_key_file(self):
        """ Load key file into memory for easier user_ID lookup"""

        self.key_data = pd.read_parquet(self.key_file)

    def save_data_out(self, processed_data):

        if self.aggregation_type == "full":
            combined_data = pd.concat(processed_data[0])
            combined_data.to_parquet(self.save_out_files)
        elif self.aggregation_type == "daily":
            combined_data = [pd.concat(processed_data[:][f]) for f in range(len(self.data_files))]
            for data, file_name in zip(combined_data, self.save_out_files):
                data.to_parquet(file_name)


###################### FILTERING ###########################
def get_row_groups(row_group):
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
    dfs = get_row_groups(row_group)
    tdf = [TrajDataFrame(x.reset_index(), latitude='latitude', longitude='longitude', datetime='datetime',
                        user_id='caid') for x in dfs]

    return tdf

def filter_tdf(tdfs, **kwargs):
    """ Filter TrajDataFrames. See below for kwargs
    https://scikit-mobility.github.io/scikit-mobility/reference/preprocessing.html#skmob.preprocessing.filtering.filter
    """
    return [filtering.filter(tdf, **kwargs) for tdf in tdfs]

def compress_tdf(tdfs, **kwargs):
    """ Compress / filter TrajDataFrames. See below for kwargs
    https://scikit-mobility.github.io/scikit-mobility/reference/preprocessing.html#skmob.preprocessing.compression.compress
    """
    return [compression.compress(tdf, **kwargs) for tdf in tdfs]

################### INDIVIDUAL MEASURES ##############################

def get_home_location(tdfs, **kwargs):
    """ Read the row group from all files and return the home location of each user. """

    return [individual.home_location(tdf, **kwargs) for tdf in tdfs]

def get_jump_lengths(tdfs, **kwargs):
    """ Read the row group from all files and return the home location of each user. """

    return [individual.jump_lengths(tdf, **kwargs) for tdf in tdfs]


def get_radius_of_gyration(tdfs, **kwargs):
    """ Read the row group from all files and return the radius of gyration of each user.
    https://scikit-mobility.github.io/scikit-mobility/reference/individual_measures.html
    """
    return [individual.radius_of_gyration(tdf, **kwargs) for tdf in tdfs]


def get_max_distance_from_home(tdfs, **kwargs):
    """ Read the row group from all files and return the max distance from home of each user.
    https://scikit-mobility.github.io/scikit-mobility/reference/individual_measures.html
    """
    return [individual.max_distance_from_home(tdf, **kwargs) for tdf in tdfs]


def get_real_entropy(tdfs, **kwargs):
    """ Read the row group from all files and return the real entropy of each user.
    https://scikit-mobility.github.io/scikit-mobility/reference/individual_measures.html
    """
    return [individual.real_entropy(tdf, **kwargs) for tdf in tdfs]


def get_uncorrelated_entropy(tdfs, **kwargs):
    """ Read the row group from all files and return the uncorrelated entropy of each user.
    https://scikit-mobility.github.io/scikit-mobility/reference/individual_measures.html
    """
    return [individual.uncorrelated_entropy(tdf, **kwargs) for tdf in tdfs]

############### COLLECTIVE MEASURES ##################

def get_collective_uncorrelated_entropy(tdfs, **kwargs):
    """ Read the row group from all files and return the uncorrelated collective entropy of each location. This is
    likely best suited for data that has been previously compressed into coarser locations.
    https://scikit-mobility.github.io/scikit-mobility/reference/collective_measures.html
    """
    return [collective.uncorrelated_location_entropy(tdf, **kwargs) for tdf in tdfs]


def get_collective_random_entropy(tdfs, **kwargs):
    """ Read the row group from all files and return the random collective entropy of each location. This is
    likely best suited for data that has been previously compressed into coarser locations.
    https://scikit-mobility.github.io/scikit-mobility/reference/collective_measures.html
    """
    return [collective.random_location_entropy(tdf, **kwargs) for tdf in tdfs]


def get_mean_square_displacement(tdfs, **kwargs):
    """ Read the row group from all files and return the mean square displacement. The mean squared displacement is a
    measure of the deviation of the position of an object with respect to a reference position over time.
    https://scikit-mobility.github.io/scikit-mobility/reference/collective_measures.html
    """
    return [collective.mean_square_displacement(tdf, **kwargs) for tdf in tdfs]


available_funcs = {"mobility_filter": filter_tdf,
                   "compress": compress_tdf,
                   "radius_of_gyration": get_radius_of_gyration,
                   "home_location": get_home_location,
                   "jump_lengths": get_jump_lengths,
                   "max_distance_from_home": get_max_distance_from_home,
                   "real_entropy": get_real_entropy,
                   "uncorrelated_entropy": get_uncorrelated_entropy,
                   "mean_square_displacement": get_mean_square_displacement,
                   "random_location_entropy": get_collective_random_entropy,
                   "uncorrelated_location_entropy": get_collective_uncorrelated_entropy}

# Due to serialization issues resulting in slow downs when processing in parallel, this class instance is initialized
# here and imported directly in calc_metrics.py where it can be modified.
data_class = DataProcessor()
