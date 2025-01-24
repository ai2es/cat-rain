import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from skmob.core.trajectorydataframe import TrajDataFrame
from skmob.measures import individual

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
    date_range = pd.date_range(start=start_date, end=end_date, freq='day')

    if days_of_week == "all":
        data_files = date_range.strftime(f"{data_path}{data_prefix}_%Y-%m-%d_PST_chunked.parquet")
    else:
        days = pd.date_range(start=start_date, end=end_date, freq=f"W-{days_of_week[0]}")
        for d in days_of_week[1:]:
            days = days.union(pd.date_range(start=start_date, end=end_date, freq=f"W-{d}"))
        data_files = days.strftime(f"{data_path}{data_prefix}_%Y-%m-%d_PST_chunked.parquet")

    return data_files


class DataProcessor():
    """ Class that supports gathering all data for any number of row_groups in serial or parallel mode. Additionally,
    can calculate a wide variety of metric from a scikit-mobility Trajectory data frame.

    Example Usage for small number of row groups:

        processor = DataProcessor(list_of_files, path_to_key_file)
        row_groups_of_interest = [11, 16, 288]
        tdf = processor.get_row_group_trajectories(row_groups_of_interest)
        individual.home_location(tdf, show_progress=False) # gets home location for each user
    """
    def __init__(self, data_files, key_file):

        self.data_files = data_files
        self.key_file = key_file
        self.meta = [pq.ParquetFile(x).metadata for x in data_files]
        self.key_meta = pq.ParquetFile(key_file).metadata
        self.n_rows = self.key_meta.num_rows
        self.n_row_groups = self.key_meta.num_row_groups
        self.all_row_groups = np.arange(self.n_row_groups)
        self.row_groups = None
    # depending on execution time, maybe do this upon init
    def load_key_file(self):
        """ Load key file into memory for easier user_ID lookup"""

        self.key_data = pd.read_parquet(self.key_file)


    def get_row_groups_data(self, row_groups):
        """Returns all data from specified row groups. Supports singular or list of row groups.
        Args:
            row_groups (int / list): List of row groups
        Returns:
            pd.DataFrame of specified row groups
        """
        if not isinstance(row_groups, list):
            row_groups = [row_groups]
        return pd.concat([pq.ParquetFile(f, metadata=m).read_row_groups(row_groups).to_pandas()
                          for f, m in zip(self.data_files, self.meta)])


    def get_row_group_trajectories(self, row_groups):
        """ Read the row group from all files and return a skmobility Trajectory DataFrame
        Args:
            row_groups (int / list): List of row groups
        Returns:
            pd.DataFrame of specified row groups
        """
        df = self.get_row_groups_data(row_groups)
        return TrajDataFrame(df.reset_index(),
                             latitude='latitude',
                             longitude='longitude',
                             datetime='datetime',
                             user_id='caid')

    ### Metrics -- many are just wrappers around scikit-mobility metrics for easier parallel computation ####

    def get_home_location(self, row_groups):
        """ Read the row group from all files and return the home location of each user.
        https://scikit-mobility.github.io/scikit-mobility/reference/individual_measures.html
        """

        tdf = self.get_row_group_trajectories(row_groups)

        return individual.home_location(tdf, show_progress=False)


    def get_jump_lengths(self, row_groups):
        """ Read the row group from all files and return the jump lengths of each user.
        https://scikit-mobility.github.io/scikit-mobility/reference/individual_measures.html
        """

        tdf = self.get_row_group_trajectories(row_groups)

        return individual.jump_lengths(tdf, show_progress=False, merge=False)

    def get_radius_of_gyration(self, row_groups):
        """ Read the row group from all files and return the radius of gyration of each user.
        https://scikit-mobility.github.io/scikit-mobility/reference/individual_measures.html
        """

        tdf = self.get_row_group_trajectories(row_groups)

        return individual.radius_of_gyration(tdf, show_progress=False)

    def get_max_distance_from_home(self, row_groups):
        """ Read the row group from all files and return the max distance from home of each user.
        https://scikit-mobility.github.io/scikit-mobility/reference/individual_measures.html
        """

        tdf = self.get_row_group_trajectories(row_groups)

        return individual.max_distance_from_home(tdf, show_progress=False)


    def get_real_entropy(self, row_groups):
        """ Read the row group from all files and return the real entropy of each user.
        https://scikit-mobility.github.io/scikit-mobility/reference/individual_measures.html
        """

        tdf = self.get_row_group_trajectories(row_groups)

        return individual.real_entropy(tdf, show_progress=False)

    def get_uncorrelated_entropy(self, row_groups):
        """ Read the row group from all files and return the uncorrelated entropy of each user.
        https://scikit-mobility.github.io/scikit-mobility/reference/individual_measures.html
        """

        tdf = self.get_row_group_trajectories(row_groups)

        return individual.uncorrelated_entropy(tdf, show_progress=False)


    # # To do:
    # # def get_specified_user_data(self, user_id):
    # #     """ Get all data for a specified user_id """
    # #     """




