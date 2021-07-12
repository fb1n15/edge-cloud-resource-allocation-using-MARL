#!/bin/bash
#SBATCH --ntasks-per-node=14
#SBATCH --nodes=1
#SBATCH --partition=lycium
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=fan_bi@icloud.com

case $SLURM_ARRAY_TASK_ID in
# Depth experiment
  1)
    CONFIG_FILE="configs/experiments/fc_depth/ppo_fc_independent_depth1.yaml"
    ;;
  2)
    CONFIG_FILE="configs/experiments/fc_depth/ppo_fc_independent_depth2.yaml"
    ;;
  3)
    CONFIG_FILE="configs/experiments/fc_depth/ppo_fc_independent_depth3.yaml"
    ;;
  4)
    CONFIG_FILE="configs/experiments/fc_depth/ppo_fc_independent_depth4.yaml"
    ;;
  5)
    CONFIG_FILE="configs/experiments/fc_depth/ppo_fc_independent_depth5.yaml"
    ;;
  6)
    CONFIG_FILE="configs/experiments/fc_depth/ppo_fc_independent_lr_depth2.yaml"
    ;;
esac

echo "Starting Job"

module load python/3.6.4
module load cuda/10.2
source venv/bin/activate
export PYTHONPATH="${PYTHONPATH}:${SLURM_SUBMIT_DIR}/src"
python src/marl-disaster.py train --config $CONFIG_FILE

echo "Finishing job"