#!/bin/bash
#PBS -l walltime=00:05:00

# Load conda environement
module load singularity/3.2.0
echo "starting"
# Navigate to working dir
cd $PBS_O_WORKDIR

export PYTHONPATH="${PYTHONPATH}:/lyceum/jp6g18/marl_disaster_relief/src"
singularity exec image.sif python src/marl-disaster.py train

echo "Finishing job"