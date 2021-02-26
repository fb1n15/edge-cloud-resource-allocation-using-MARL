#!/bin/bash
#SBATCH --ntasks-per-node=7
#SBATCH --nodes=1
#SBATCH --partition=lycium
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jp6g18@soton.ac.uk
#SBATCH --output=test-srun.out

echo "Starting Job"

module load python/3.6.4
module load cuda/10.2
source venv/bin/activate
export PYTHONPATH="${PYTHONPATH}:${SLURM_SUBMIT_DIR}/src"
python src/marl-disaster.py train

echo "Finishing job"