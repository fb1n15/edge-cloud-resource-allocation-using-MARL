#!/bin/bash
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=fan_bi@icloud.com

case $SLURM_ARRAY_TASK_ID in
# Scaling up checkpoint
  1)
    CONFIG_FILE="configs/experiments/scaling_up/ppo_fc_independent.yaml"
    ;;
  2)
    CONFIG_FILE="configs/experiments/scaling_up/ppo_convolutional_independent.yaml"
    ;;
  3)
    CONFIG_FILE="configs/experiments/scaling_up/ppo_convolutional_centralised/1agents.yaml"
    ;;
  4)
    CONFIG_FILE="configs/experiments/scaling_up/ppo_convolutional_centralised/2agents.yaml"
    ;;
  5)
    CONFIG_FILE="configs/experiments/scaling_up/ppo_convolutional_centralised/4agents.yaml"
    ;;
  6)
    CONFIG_FILE="configs/experiments/scaling_up/ppo_convolutional_centralised/8agents.yaml"
    ;;
  7)
    CONFIG_FILE="configs/experiments/scaling_up/qmix/1agents.yaml"
    ;;
  8)
    CONFIG_FILE="configs/experiments/scaling_up/qmix/2agents.yaml"
    ;;
  9)
    CONFIG_FILE="configs/experiments/scaling_up/qmix/4agents.yaml"
    ;;
  10)
    CONFIG_FILE="configs/experiments/scaling_up/qmix/8agents.yaml"
    ;;
esac

echo "Starting Job"

module load conda/py3-latest
source activate jack
export PYTHONPATH="${PYTHONPATH}:${SLURM_SUBMIT_DIR}/src"
python src/marl-disaster.py train --config $CONFIG_FILE

echo "Finishing job"