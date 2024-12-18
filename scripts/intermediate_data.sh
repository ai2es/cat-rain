#!/bin/bash -l
#PBS -N cat_data_04
#PBS -l select=1:ncpus=128:ngpus=0
#PBS -l walltime=12:00:00
#PBS -A NAML0001
#PBS -q main
#PBS -j oe

source ~/.bashrc
conda activate cat
python -u create_intermediate_data.py