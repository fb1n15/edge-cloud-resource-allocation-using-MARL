#!/bin/bash
#SBATCH --ntasks-per-node=28
#SBATCH --nodes=1
#SBATCH --partition=lycium
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jp6g18@soton.ac.uk

case $SLURM_ARRAY_TASK_ID in
  1)
    CONFIG_FILE="configs/gridworld_obstacles/experiments/ppo_vision_net_independent.yaml"
    ;;
esac

echo "Starting Job"

module load python/3.6.4
module load cuda/10.2
source venv/bin/activate
export PYTHONPATH="${PYTHONPATH}:${SLURM_SUBMIT_DIR}/src"
python src/marl-disaster.py train --config $CONFIG_FILE

echo "Finishing job"