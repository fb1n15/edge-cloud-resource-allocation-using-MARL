#!/bin/bash

#SBATCH --requeue
#SBATCH --partition=scavenger
#SBATCH --output=/mainfs/home/fb1n15/MARL-ReverseAuction/marl-edge-cloud/iridis-reports/%j.out  # change the output log destination
#SBATCH --ntasks=1  # Number of Tasks (up-to 32 jobs running at the same time)
#SBATCH --cpus-per-task=10  # use 10 CPU cores for each task
#SBATCH --time=00:30:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=fan_bi@icloud.com

# (https://stackoverflow.com/a/67537416/7060068)
# #SBATCH --nodes=2  # Number of nodes requested
# #SBATCH --exclusive          # I don't want to share my compute node with anyone
# #SBATCH --ntasks-per-node=4  # Tasks per node  (https://stackoverflow.com/a/51141287/7060068)

cd "$HOME"/MARL-ReverseAuction/marl-edge-cloud/ || exit  # cd to the project location
n_tasks=1

CONFIG_FILE1="/mainfs/home/fb1n15/MARL-ReverseAuction/marl-edge-cloud/configs/config_local_first-price.yaml"

CONFIG_FILE2="/mainfs/home/fb1n15/MARL-ReverseAuction/marl-edge-cloud/configs/config_local_second-price.yaml"

echo "Starting Job"

module load conda/py3-latest
source activate jack
module load openmpi/3.0.0/intel
export PYTHONPATH="${PYTHONPATH}:${SLURM_SUBMIT_DIR}/src"
export PYTHONPATH="${PYTHONPATH}:/local/software/cplex/12.8/cplex/python/3.6/x86-64_linux"


mpirun -np $n_tasks --bind-to none python ./src/marl.py train --config $CONFIG_FILE1
mpirun -np $n_tasks --bind-to none python ./src/marl.py train --config $CONFIG_FILE2

echo "Job Finished"