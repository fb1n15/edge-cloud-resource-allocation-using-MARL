#!/bin/bash


#SBATCH --ntasks-per-node=4  # Tasks per node
#SBATCH --nodes=2  # Number of nodes requested
#SBATCH --partition=batch
#SBATCH --time=2:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=fan_bi@icloud.com

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
export PYTHONPATH="${PYTHONPATH}:${SLURM_SUBMIT_DIR}/src"
for j in {1..10}
do
  python ./src/marl.py train --config $CONFIG_FILE
done


echo "Finishing job"