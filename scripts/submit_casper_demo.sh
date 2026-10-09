#!/bin/bash -l
#PBS -N cat
#PBS -A NAML0001
#PBS -l walltime=00:04:00
#PBS -j oe
#PBS -q casper
#PBS -l select=1:ncpus=8:mem=32GB
#PBS -m a
#PBS -M ggantos@ucar.edu
module load conda
conda activate cat-demo


python -u calc_metrics.py -c ../config/demo.yaml
# python -u calc_home_location.py -c ../config/calc_home_loc.yaml
# python -u calc_uids.py -c ../config/calc_uids.yaml
# python -u get_mrms_data.py
# python -u calc_mrms.py -c ../config/mrms.yaml