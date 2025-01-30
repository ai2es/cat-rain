import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from skmob.core.trajectorydataframe import TrajDataFrame
from skmob.measures import individual, collective
from skmob.preprocessing import filtering, compression
from functools import partial

def get_files(data_path, data_prefix, start_date, end_date, days_of_week="all"):
    """
    Retrieve the file names for a specified date range. Supports specification of
    days of the week.

    Args:
        data_path (str): Base path to daily chunked parquet files
        data_prefix (str): Prefix string for file names
        start_date (str): Start date of file format of "YYYY-MM-DD"
        end_date (str): End date of file format of "YYYY-MM-DD"
        days_of_week (str / list): Days of week. Supports "all" and individual days of the week.
            The following formats are required:
                "MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"

            days_of_week=["TUE", "WED", "FRI"] is the correct format for all Tuesdays, Wednesdays, and Fridays
            between the start and end dates (inclusive).
    Returns (list):
        List of matching file names
    """
    date_range = pd.date_range(start=start_date, end=end_date, freq='d')

    if days_of_week == "all":
        data_files = date_range.strftime(f"{data_path}{data_prefix}_%Y-%m-%d_PST_chunked.parquet")
    else:
        days = pd.date_range(start=start_date, end=end_date, freq=f"W-{days_of_week[0]}")
        for d in days_of_week[1:]:
            days = days.union(pd.date_range(start=start_date, end=end_date, freq=f"W-{d}"))
        data_files = days.strftime(f"{data_path}{data_prefix}_%Y-%m-%d_PST_chunked.parquet")

    return data_files


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
    @classmethod
    def set_values(self, data_files=None, key_file=None):

        self.data_files = data_files
        self.key_file = key_file
        self.meta = [pq.ParquetFile(x).metadata for x in data_files]
        self.key_meta = pq.ParquetFile(key_file).metadata
        self.n_rows = self.key_meta.num_rows
        self.n_row_groups = self.key_meta.num_row_groups
        self.all_row_groups = np.arange(self.n_row_groups)
        self.row_groups = None
        self.parallel_agg = 'full'  # Currently supports "full" (over all data) and "daily" (per file).

    # depending on execution time, maybe do this upon init
    def load_key_file(self):
        """ Load key file into memory for easier user_ID lookup"""

        self.key_data = pd.read_parquet(self.key_file)


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
    if data_class.parallel_agg == 'full':
        return ensure_list(pd.concat(df_row_groups))
    elif data_class.parallel_agg == 'daily':
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

    return [individual.home_location(tdf, show_progress=False, **kwargs) for tdf in tdfs]

def get_jump_lengths(row_group, **kwargs):
    """ Read the row group from all files and return the home location of each user. """

    tdfs = get_row_group_trajectories(row_group)

    return [individual.jump_lengths(tdf, show_progress=False, **kwargs) for tdf in tdfs]


def get_radius_of_gyration(row_group, **kwargs):
    """ Read the row group from all files and return the radius of gyration of each user.
    https://scikit-mobility.github.io/scikit-mobility/reference/individual_measures.html
    """

    tdfs = get_row_group_trajectories(row_group)

    return [individual.radius_of_gyration(tdf, show_progress=False, **kwargs) for tdf in tdfs]


def get_max_distance_from_home(row_group, **kwargs):
    """ Read the row group from all files and return the max distance from home of each user.
    https://scikit-mobility.github.io/scikit-mobility/reference/individual_measures.html
    """

    tdfs = get_row_group_trajectories(row_group)

    return [individual.max_distance_from_home(tdf, show_progress=False, **kwargs) for tdf in tdfs]


def get_real_entropy(row_group, **kwargs):
    """ Read the row group from all files and return the real entropy of each user.
    https://scikit-mobility.github.io/scikit-mobility/reference/individual_measures.html
    """

    tdfs = get_row_group_trajectories(row_group)

    return [individual.real_entropy(tdf, show_progress=False, **kwargs) for tdf in tdfs]


def get_uncorrelated_entropy(row_group, **kwargs):
    """ Read the row group from all files and return the uncorrelated entropy of each user.
    https://scikit-mobility.github.io/scikit-mobility/reference/individual_measures.html
    """

    tdfs = get_row_group_trajectories(row_group)

    return [individual.uncorrelated_entropy(tdf, show_progress=False, **kwargs) for tdf in tdfs]

############### COLLECTIVE MEASURES ##################

def get_collective_uncorrelated_entropy(row_group, **kwargs):
    """ Read the row group from all files and return the uncorrelated collective entropy of each location. This is
    likely best suited for data that has been previously compressed into coarser locations.
    https://scikit-mobility.github.io/scikit-mobility/reference/collective_measures.html
    """

    tdfs = get_row_group_trajectories(row_group)

    return [collective.uncorrelated_location_entropy(tdf, show_progress=False, **kwargs) for tdf in tdfs]


def get_collective_random_entropy(row_group, **kwargs):
    """ Read the row group from all files and return the random collective entropy of each location. This is
    likely best suited for data that has been previously compressed into coarser locations.
    https://scikit-mobility.github.io/scikit-mobility/reference/collective_measures.html
    """

    tdfs = get_row_group_trajectories(row_group)

    return [collective.random_location_entropy(tdf, show_progress=False, **kwargs) for tdf in tdfs]


def get_mean_square_displacement(row_group, **kwargs):
    """ Read the row group from all files and return the mean square displacement. The mean squared displacement is a
    measure of the deviation of the position of an object with respect to a reference position over time.
    https://scikit-mobility.github.io/scikit-mobility/reference/collective_measures.html
    """

    tdfs = get_row_group_trajectories(row_group)

    return [collective.mean_square_displacement(tdf, show_progress=False, **kwargs) for tdf in tdfs]

 # Due to serialization issues resulting in slow downs when processing in parallel, this class instance is initialized
 # here and imported directly in calc_metrics.py where it can be modified.
data_class = DataProcessor()
