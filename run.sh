#!/bin/bash
echo "Loading conda environment"
module load conda/4.4.0
source activate ip_env
echo "Loaded conda environment"

python src/learning/training.py
echo "Finishing job"