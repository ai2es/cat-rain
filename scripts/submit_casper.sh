#!/bin/bash -l
#PBS -N cat
#PBS -A NAML0001
#PBS -l walltime=02:00:00
#PBS -j oe
#PBS -q casper
#PBS -l select=1:ncpus=34:mem=354GB
#PBS -m a
#PBS -M ggantos@ucar.edu
module load conda
conda activate cat

# python -u create_intermediate_data.py
# python -u chunk_data.py
# python -u calc_metrics.py -c ../config/process_conf.yaml
# python -u calc_home_location.py -c ../config/calc_home_loc.yaml
# python -u get_mrms_data.py
# python -u calc_mrms.py -c ../config/mrms.yaml
python -u calc_uids.py -c ../config/calc_uids.yaml