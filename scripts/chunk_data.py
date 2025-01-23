import pandas as pd
import numpy as np
import glob
import time
from src.preprocess import create_chunk_key, write_daily_chunked_data

base_path = "<path to daily veraset files>"
key_out_path = "<out path for master key file>"
chunked_out_path = "<out path for chunked data>"
n_ids_per_row_group = 1000

files = sorted(glob.glob(join(base_path, "*.parquet")))

create_chunk_key(files=files,
                 out_path=key_out_path,
                 n_ids_per_chunk=n_ids_per_row_group)

write_daily_chunked_data(daily_files=files,
                         key_file=key_out_path,
                         base_out_path=chunked_out_path)