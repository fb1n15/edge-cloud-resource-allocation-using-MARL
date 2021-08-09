#!/bin/bash

## maximum count of tasks per node
#SBATCH --ntasks-per-node=14
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=fan_bi@icloud.com

case $SLURM_ARRAY_TASK_ID in
# Scaling up experiment
  1)
    CONFIG_FILE="/mainfs/home/fb1n15/MARL-Jack/marl-disaster-relief/configs/experiments/edge_cloud/hyperparameters/ppo_fc_centralised_critic_lr.yaml"
    ;;
  2)
    CONFIG_FILE="/mainfs/home/fb1n15/MARL-Jack/marl-disaster-relief/configs/experiments/edge_cloud/hyperparameters/ppo_fc_independent_lr.yaml"
    ;;
esac

echo "Starting Job"

module load conda/py3-latest
source activate jack
export PYTHONPATH="${PYTHONPATH}:${SLURM_SUBMIT_DIR}/src"
conda env list
python --version
python src/marl-disaster.py train --config $CONFIG_FILE

echo "Finishing job"