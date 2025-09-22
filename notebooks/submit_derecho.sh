#!/bin/bash -l
#PBS -N cat_mrms
#PBS -A NAML0001
#PBS -l walltime=00:30:00
#PBS -j oe
#PBS -q main
#PBS -l select=1:ompthreads=2:mem=235GB
#PBS -m a
#PBS -M ggantos@ucar.edu
module load conda
conda activate cat

python -u mrms_aws.py
