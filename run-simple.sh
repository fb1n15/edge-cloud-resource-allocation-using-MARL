#!/bin/bash
#PBS -l walltime=12:00:00
#PBS –l nodes=1:ppn=16
#PBS -m ae -M jp6g18@soton.ac.uk
#PBS -N marl-disaster-response

echo "Starting Job"

# Load conda environement
module load singularity/3.2.0

# Navigate to working dir
cd $PBS_O_WORKDIR

export PYTHONPATH="${PYTHONPATH}:/lyceum/jp6g18/git/marl_disaster_relief/src"
singularity exec image.sif python src/marl-disaster.py train

echo "Finishing job"