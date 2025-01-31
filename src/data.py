import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import os
from skmob.core.trajectorydataframe import TrajDataFrame
from skmob.measures import individual, collective
from skmob.preprocessing import filtering, compression
from functools import partial


def ensure_list(obj):
    """Ensures the returned object is always a list of DataFrames for consistency."""
    return [obj] if isinstance(obj, pd.DataFrame) else obj


def func_w_kwargs(func, kwargs=None):
    """ Return specified funciton with static key word arguments for parallel processing. Also supports no kwargs."""
    if kwargs is not None:
        return partial(func, **kwargs)
    else:
        return partial(func)


class DataProcessor():
    """ Class that supports gathering all data for any number of row_groups in serial or parallel mode. Additionally,
    can calculate a wide variety of metric from a scikit-mobility Trajectory data frame.

    Example Usage for small number of row groups:

        processor = DataProcessor(list_of_files, path_to_key_file)
        row_groups_of_interest = [11, 16, 288]
        tdf = processor.get_row_group_trajectories(row_groups_of_interest)
        individual.home_location(tdf, show_progress=False) # gets home location for each user
    """
    # @classmethod
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

        if isinstance(self.function, str):
            self.func_str = self.function
        else:
            self.func_str = '_'.join(self.function)

        if self.aggregation_type == "full":
            fn = f"veraset_{self.function}_{self.start_date}_{self.end_date}.parquet"
            self.save_out_files = os.path.join(self.save_out_path, fn)
        elif self.aggregation_type == "daily":
            self.save_out_files = [os.path.join(self.save_out_path, f"{self.function}_{f.split('/')[-1]}")
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
            combined_data = pd.concat(processed_data[0][0])
            combined_data.to_parquet(self.save_out_files)
        elif self.aggregation_type == "daily":
            combined_data = [pd.concat(processed_data[0][:][f]) for f in range(len(self.data_files))]
            for data, file_name in zip(combined_data, self.save_out_files):
                data.to_parquet(file_name)

    # # To do:
    # # def get_specified_user_data(self, user_id):
    # #     """ Get all data for a specified user_id """
    # #     """

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


def get_row_group_trajectories(row_group):
    """ Read the row group from all files and return a skmobility Trajectory DataFrame """
    dfs = get_row_groups(row_group)
    tdf = [TrajDataFrame(x.reset_index(), latitude='latitude', longitude='longitude', datetime='datetime',
                        user_id='caid') for x in dfs]

    return tdf

def filter_tdf(row_group, **kwargs):
    """ Filter TrajDataFrames. See below for kwargs
    https://scikit-mobility.github.io/scikit-mobility/reference/preprocessing.html#skmob.preprocessing.filtering.filter
    """
    tdfs = get_row_group_trajectories(row_group)

    return [filtering.filter(tdf, **kwargs) for tdf in tdfs]

def compress_tdf(row_group, **kwargs):
    """ Compress / filter TrajDataFrames. See below for kwargs
    https://scikit-mobility.github.io/scikit-mobility/reference/preprocessing.html#skmob.preprocessing.compression.compress
    """
    tdfs = get_row_group_trajectories(row_group)

    return [compression.compress(tdf, **kwargs) for tdf in tdfs]

################### INDIVIDUAL MEASURES ##############################

def get_home_location(row_group, **kwargs):
    """ Read the row group from all files and return the home location of each user. """

    tdfs = get_row_group_trajectories(row_group)

    return [individual.home_location(tdf, **kwargs) for tdf in tdfs]

def get_jump_lengths(row_group, **kwargs):
    """ Read the row group from all files and return the home location of each user. """

    tdfs = get_row_group_trajectories(row_group)

    return [individual.jump_lengths(tdf, **kwargs) for tdf in tdfs]


def get_radius_of_gyration(row_group, **kwargs):
    """ Read the row group from all files and return the radius of gyration of each user.
    https://scikit-mobility.github.io/scikit-mobility/reference/individual_measures.html
    """

    tdfs = get_row_group_trajectories(row_group)

    return [individual.radius_of_gyration(tdf, **kwargs) for tdf in tdfs]


def get_max_distance_from_home(row_group, **kwargs):
    """ Read the row group from all files and return the max distance from home of each user.
    https://scikit-mobility.github.io/scikit-mobility/reference/individual_measures.html
    """

    tdfs = get_row_group_trajectories(row_group)

    return [individual.max_distance_from_home(tdf, **kwargs) for tdf in tdfs]


def get_real_entropy(row_group, **kwargs):
    """ Read the row group from all files and return the real entropy of each user.
    https://scikit-mobility.github.io/scikit-mobility/reference/individual_measures.html
    """

    tdfs = get_row_group_trajectories(row_group)

    return [individual.real_entropy(tdf, **kwargs) for tdf in tdfs]


def get_uncorrelated_entropy(row_group, **kwargs):
    """ Read the row group from all files and return the uncorrelated entropy of each user.
    https://scikit-mobility.github.io/scikit-mobility/reference/individual_measures.html
    """

    tdfs = get_row_group_trajectories(row_group)

    return [individual.uncorrelated_entropy(tdf, **kwargs) for tdf in tdfs]

############### COLLECTIVE MEASURES ##################

def get_collective_uncorrelated_entropy(row_group, **kwargs):
    """ Read the row group from all files and return the uncorrelated collective entropy of each location. This is
    likely best suited for data that has been previously compressed into coarser locations.
    https://scikit-mobility.github.io/scikit-mobility/reference/collective_measures.html
    """

    tdfs = get_row_group_trajectories(row_group)

    return [collective.uncorrelated_location_entropy(tdf, **kwargs) for tdf in tdfs]


def get_collective_random_entropy(row_group, **kwargs):
    """ Read the row group from all files and return the random collective entropy of each location. This is
    likely best suited for data that has been previously compressed into coarser locations.
    https://scikit-mobility.github.io/scikit-mobility/reference/collective_measures.html
    """

    tdfs = get_row_group_trajectories(row_group)

    return [collective.random_location_entropy(tdf, **kwargs) for tdf in tdfs]


def get_mean_square_displacement(row_group, **kwargs):
    """ Read the row group from all files and return the mean square displacement. The mean squared displacement is a
    measure of the deviation of the position of an object with respect to a reference position over time.
    https://scikit-mobility.github.io/scikit-mobility/reference/collective_measures.html
    """

    tdfs = get_row_group_trajectories(row_group)

    return [collective.mean_square_displacement(tdf, **kwargs) for tdf in tdfs]

 # Due to serialization issues resulting in slow downs when processing in parallel, this class instance is initialized
 # here and imported directly in calc_metrics.py where it can be modified.
data_class = DataProcessor()
