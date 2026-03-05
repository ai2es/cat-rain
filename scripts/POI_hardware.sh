#!/bin/bash -l
#PBS -N JC_Fuel_POIs
#PBS -A NAML0001
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -q main
#PBS -l select=1:ompthreads=2
#PBS -m a
#PBS -M jcelis@ou.edu
module load conda
conda activate pycat

# Change to the scripts directory
cd /glade/u/home/jcelis/POIs_filtered

# Export the project root to PYTHONPATH so that src can be found
export PYTHONPATH=/glade/u/home/jcelis/POIs_filtered
python POI_hardware.py -c ../config/process_conf.yaml
