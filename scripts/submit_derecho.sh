#!/bin/bash -l
#PBS -N cat
#PBS -A NAML0001
#PBS -l walltime=02:00:00
#PBS -j oe
#PBS -q main
#PBS -l select=1
#PBS -m a
#PBS -M cbecker@ucar.edu
module load conda
conda activate cat

#python -u create_intermediate_data.py
#python -u chunk_data.py
#python calc_home_location.py -c ../config/calc_home_loc.yml
python -u calc_metrics.py -c ../config/process_conf.yaml