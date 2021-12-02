#!/bin/bash

#SBATCH --output=/mainfs/home/fb1n15/MARL-ReverseAuction/marl-edge-cloud/iridis-reports/%j.out
#SBATCH --ntasks=2  # Number of Tasks (up-to 32 nodes running at the same time)
#SBATCH --cpus-per-task=40  # use multiple cores each for multithreading
#SBATCH --time=2:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=fan_bi@icloud.com

# (https://stackoverflow.com/a/67537416/7060068)
# #SBATCH --nodes=2  # Number of nodes requested
# #SBATCH --exclusive          # I don't want to share my compute node with anyone
# #SBATCH --ntasks-per-node=4  # Tasks per node  (https://stackoverflow.com/a/51141287/7060068)

cd "$HOME"/MARL-ReverseAuction/marl-edge-cloud/ || exit  # cd to the project location
n_tasks=2

CONFIG_FILE1="/mainfs/home/fb1n15/MARL-ReverseAuction/marl-edge-cloud/configs/5-actions_first-price_no-history_revenue.yaml"

echo "Starting Job"

module load conda/py3-latest
source activate jack
module load openmpi/3.0.0/intel
export PYTHONPATH="${PYTHONPATH}:${SLURM_SUBMIT_DIR}/src"

mpirun -np $n_tasks --bind-to none python ./src/marl.py train --config $CONFIG_FILE1

echo "Finishing job"