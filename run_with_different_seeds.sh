#!/bin/bash

# #SBATCH --nodes=2  # Number of nodes requested
#SBATCH --ntasks=20  # Number of Tasks (up-to 32 jobs running at the same time)
#SBATCH --ntasks-per-node=4  # Tasks per node
#SBATCH --partition=batch
#SBATCH --time=2:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=fan_bi@icloud.com
#SBATCH --output=/mainfs/home/fb1n15/MARL-ReverseAuction/marl-edge-cloud/iridis-reports/%j.out
# (https://stackoverflow.com/a/67537416/7060068)


cd "$HOME"/MARL-ReverseAuction/marl-edge-cloud/ || exit  # cd to the project location

case $SLURM_ARRAY_TASK_ID in
# Scaling up experiment
  1)
    CONFIG_FILE="/mainfs/home/fb1n15/MARL-ReverseAuction/marl-edge-cloud/configs/experiments/edge_cloud/hyperparameters/cpu_ppo_fc_independent_with_history.yaml"
    ;;
  2)
    CONFIG_FILE="/mainfs/home/fb1n15/MARL-ReverseAuction/marl-edge-cloud/configs/experiments/edge_cloud/hyperparameters/cpu_ppo_fc_independent_without_history.yaml"
    ;;
esac

echo "Starting Job"

module load conda/py3-latest
source activate jack
module load openmpi/3.0.0/intel
export PYTHONPATH="${PYTHONPATH}:${SLURM_SUBMIT_DIR}/src"
#for j in {1..10}
#do
#  python ./src/marl.py train --config $CONFIG_FILE
#done

mpirun -np 20 python ./src/marl.py train --config $CONFIG_FILE

echo "Finishing job"