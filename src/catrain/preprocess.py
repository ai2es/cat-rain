import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import time
from os.path import join


def load_data(file_path):

    """Loads a parquet file, converts datatypes, and converts to PST time.
    Args:
        file_path (str): Path to the parquet file.
        """

    columns = ['utc_timestamp', 'caid', 'id_type', 'latitude', 'longitude', 'horizontal_accuracy']
    df = pd.read_parquet(file_path, columns=columns)
    df = df.astype({'latitude': np.float32, 'longitude': np.float32})
    df['horizontal_accuracy'] = pd.to_numeric(df['horizontal_accuracy']).astype('int16')
    df['utc_timestamp'] = df['utc_timestamp'].dt.tz_localize('utc').dt.tz_convert('US/Pacific')
    # df['zipcode'] = [df['geo_fields'].iloc[x][2][1] for x in range(len(df))]
    # df['city'] = [df['geo_fields'].iloc[x][1][1] for x in range(len(df))]

    return df.rename(columns={'utc_timestamp': 'datetime'}).sort_values(["caid", "datetime"])


def create_chunk_key(files, out_path, n_ids_per_chunk=1000):
    """ Create a master key with User ID and corresponding row group number.
    Args:
        files (list): list of file paths
        out_path (str): output path
        n_ids_per_chunk (int, optional): number of row groups to use per chunk. Defaults to 1000.
    """
    all_unique_ids = []
    s = time.time()
    for f in files:
        print(f)
        ids = pd.read_parquet(f, columns=['caid'])
        unique_ids = sorted(set(ids.index.get_level_values(0)))
        all_unique_ids.append(unique_ids)
    print(np.round((time.time() - s) / 60, 2), "minutes to read all unique IDs.")
    s = time.time()
    all_uniques = sorted(set(np.hstack(all_unique_ids)))
    print(np.round((time.time() - s) / 60, 2), "minutes to combine and sort all unique IDs.")

    key_df = pd.DataFrame(data={'row_group': np.zeros(len(all_uniques))}, index=all_uniques)
    n_ids = len(all_uniques)
    n_row_groups = int(np.ceil(n_ids / n_ids_per_chunk))
    row_group_keys = np.hstack([np.repeat(x, n_ids_per_chunk) for x in range(n_row_groups)])[:n_ids].astype('int32')
    key_df['row_group'] = row_group_keys
    key_df.to_parquet(out_path, row_group_size=n_ids_per_chunk)


def write_daily_chunked_data(daily_files, key_file, base_out_path):
    """ Write out daily files with row groups corresponding to each set of unique IDs.
    Args:
        daily_files (list): list of daily file paths
        key_file (pandas.DataFrame): Master key files with row groups corresponding to each set of unique IDs
        base_out_path (str): output directory path
    """
    key = pq.ParquetFile(key_file)
    n_row_groups = key.metadata.num_row_groups

    for f in daily_files:
        print(f)
        out_file = join(base_out_path, f.split('/')[-1].split('.')[0] + "_chunked.parquet")
        print(out_file)
        table = pq.read_table(f)
        data = table.to_pandas().reset_index().set_index('caid')
        schema = pa.Table.from_pandas(data).schema
        daily_ids = set(data.index)
        writer = pq.ParquetWriter(out_file, schema)

        for row_group in np.arange(n_row_groups):
            ids = set(key.read_row_group(row_group).to_pandas().index)
            found_ids = list(daily_ids.intersection(ids))
            row_group = pa.Table.from_pandas(data.loc[found_ids])
            writer.write_table(row_group)
        writer.close()
