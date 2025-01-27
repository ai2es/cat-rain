import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from skmob.core.trajectorydataframe import TrajDataFrame
from skmob.preprocessing import detection
from skmob.measures import individual


def get_row_groups(row_group, files):
    """ Read a specific row group across all files. """

    return pd.concat([pq.ParquetFile(f, metadata=m).read_row_group(row_group).to_pandas() for f, m in zip(files, meta)])


def get_row_group_trajectories(row_group, files):
    """ Read the row group from all files and return a skmobility Trajectory DataFrame """
    df = get_row_groups(row_group, files)
    tdf = TrajDataFrame(df.reset_index(), latitude='latitude', longitude='longitude', datetime='datetime',
                        user_id='caid')

    return tdf


def get_home_location(row_group, files):
    """ Read the row group from all files and return the home location of each user. """

    tdf = get_row_group_trajectories(row_group, files)

    return individual.home_location(tdf, show_progress=False)