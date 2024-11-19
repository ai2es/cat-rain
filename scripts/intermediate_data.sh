#!/bin/bash -l
#PBS -N cat_data
#PBS -l select=1:ncpus=36:ngpus=0:mem=128GB
#PBS -l walltime=14:00:00
#PBS -A NAML0001
#PBS -q casper
#PBS -j oe
#PBS -o ../logs/$PBS_JOBID.out

source ~/.bashrc
conda activate cat
python -u create_intermediate_data.py