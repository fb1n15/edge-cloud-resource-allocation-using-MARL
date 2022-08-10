#!/bin/bash

#SBATCH --partition=batch
#SBATCH --nodes=1  # Number of nodes requested
#SBATCH --job-name=model_depth
#SBATCH --output=/mainfs/home/fb1n15/MARL-ReverseAuction/marl-edge-cloud/iridis-reports/%j.out  # change the output log destination
#SBATCH --ntasks=1  # Number of Tasks (up-to 32 jobs running at the same time)
#SBATCH --ntasks-per-node=1  # Tasks per node  (https://stackoverflow.com/a/51141287/7060068)
#SBATCH --cpus-per-task=40  # use 10 CPU cores for each task
# #SBATCH --time=1:00:00
#SBATCH --time=60:00:00
#SBATCH --exclusive          # I don't want to share my compute node with anyone

#SBATCH --mail-type=ALL
#SBATCH --mail-user=fb1n15@soton.ac.uk.com

# (https://stackoverflow.com/a/67537416/7060068)

n_tasks=1  # Number of Tasks (up-to 32 jobs running at the same time)
cd "$HOME"/MARL-ReverseAuction/marl-edge-cloud/ || exit  # cd to he project location
CONFIG_FILE="/mainfs/home/fb1n15/MARL-ReverseAuction/marl-edge-cloud/simulations/hyperparamter_optimisation__model_depth/config_HPO_iridis5.yaml"
echo "Starting Job"

module load conda/py3-latest
source activate edge-cloud-resource-allocation
ulimit -c 0  # disable core dumps
module load openmpi/3.0.0/intel
export PYTHONPATH="${PYTHONPATH}:${SLURM_SUBMIT_DIR}/src"
export PYTHONPATH="${PYTHONPATH}:/local/software/cplex/12.8/cplex/python/3.6/x86-64_linux"

START=$1
END=$(($1+9))
# run 10 trials
for SEED in $(seq "$START" "$END")
do
  echo "Starting a Job with SEED=$SEED"  # print the seed value
  mpirun -np $n_tasks --bind-to none python ./src/marl.py train --config $CONFIG_FILE  --env_seed "$SEED"
done

echo "Job Finished"  # print Job Finished