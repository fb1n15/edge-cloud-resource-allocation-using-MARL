#!/bin/bash
#SBATCH --ntasks-per-node=14
#SBATCH --nodes=1
#SBATCH --partition=lycium
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jp6g18@soton.ac.uk

case $SLURM_ARRAY_TASK_ID in
# Scaling up checkpoint
  1)
    CONFIG_FILE="configs/experiments/hyperparameters/ppo_fc_independent_lr.yaml"
    ;;
  2)
    CONFIG_FILE="/mainfs/home/fb1n15/MARL-Jack/marl-disaster-relief/configs/experiments/edge_cloud/hyperparameters/ppo_fc_centralised_critic_lr.yaml"
    ;;
esac

echo "Starting Job"

module load python/3.6.4
module load cuda/10.2
source venv/bin/activate
export PYTHONPATH="${PYTHONPATH}:${SLURM_SUBMIT_DIR}/src"
python src/marl-disaster.py train --config $CONFIG_FILE

echo "Finishing job"