#!/bin/bash

#SBATCH --output=/mainfs/home/fb1n15/MARL-ReverseAuction/marl-edge-cloud/iridis-reports/%j.out  # change the output log destination
#SBATCH --ntasks=10  # Number of Tasks (up-to 32 jobs running at the same time)
#SBATCH --cpus-per-task=10  # use multiple cores each for multithreading
#SBATCH --time=2:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=fan_bi@icloud.com

# (https://stackoverflow.com/a/67537416/7060068)
# #SBATCH --nodes=2  # Number of nodes requested
# #SBATCH --exclusive          # I don't want to share my compute node with anyone
# #SBATCH --ntasks-per-node=4  # Tasks per node  (https://stackoverflow.com/a/51141287/7060068)

cd "$HOME"/MARL-ReverseAuction/marl-edge-cloud/ || exit  # cd to the project location
n_jobs=10

CONFIG_FILE1="/mainfs/home/fb1n15/MARL-ReverseAuction/marl-edge-cloud/configs/experiments/edge_cloud/hyperparameters/cpu_ppo_fc_independent_with-history_5-actions_env-1.yaml"

CONFIG_FILE2="/mainfs/home/fb1n15/MARL-ReverseAuction/marl-edge-cloud/configs/experiments/edge_cloud/hyperparameters/cpu_ppo_fc_independent_with-history_5-actions_env-2.yaml"

CONFIG_FILE3="/mainfs/home/fb1n15/MARL-ReverseAuction/marl-edge-cloud/configs/experiments/edge_cloud/hyperparameters/cpu_ppo_fc_independent_with-history_5-actions_env-3.yaml"

echo "Starting Job, number of jobs:"
echo $n_jobs

module load conda/py3-latest
source activate jack
module load openmpi/3.0.0/intel
export PYTHONPATH="${PYTHONPATH}:${SLURM_SUBMIT_DIR}/src"

mpirun -np $n_jobs --bind-to none python ./src/marl.py train --config $CONFIG_FILE1
#mpirun -np $n_jobs --bind-to none python ./src/marl.py train --config $CONFIG_FILE2
#mpirun -np $n_jobs --bind-to none python ./src/marl.py train --config $CONFIG_FILE3

echo "Finishing job"