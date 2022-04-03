#!/bin/bash

#SBATCH --requeue
#SBATCH --partition=scavenger
#SBATCH --output=/mainfs/home/fb1n15/MARL-ReverseAuction/marl-edge-cloud/iridis-reports/%j.out  # change the output log destination
#SBATCH --nodes=1  # Number of nodes requested
# #SBATCH --cpus-per-task=40  # use 40 CPU cores for each task
#SBATCH --time=00:10:00
#SBATCH --exclusive          # I don't want to share my compute node with anyone
### Give all resources to a single Ray task, ray can manage the resources internally
#SBATCH --ntasks-per-node=1  # Number of tasks to be launched per Node
# #SBATCH --gpus-per-task=0  # Number of GPUs per task

#SBATCH --mail-type=ALL
#SBATCH --mail-user=fb1n15@soton.ac.uk.com

# (https://stackoverflow.com/a/67537416/7060068)
# #SBATCH --cpus-per-task=10  # use 10 CPU cores for each task
# #SBATCH --ntasks=1  # Number of Tasks (up-to 32 jobs running at the same time)

cd "$HOME"/MARL-ReverseAuction/marl-edge-cloud/ || exit # cd to the project location
n_tasks=1
CONFIG_FILE="/mainfs/home/fb1n15/MARL-ReverseAuction/marl-edge-cloud/simulations/hyperparamter_optimisation__learning_rate/config_HPO_iridis5.yaml"

module load conda/py3-latest
source activate edge-cloud-resource-allocation
ulimit -c 0  # disable core dumps
module load openmpi/3.0.0/intel
export PYTHONPATH="${PYTHONPATH}:${SLURM_SUBMIT_DIR}/src"
export PYTHONPATH="${PYTHONPATH}:/local/software/cplex/12.8/cplex/python/3.6/x86-64_linux"
# https://stackoverflow.com/a/37868546/7060068
export LC_ALL=en_US.utf-8
export LANG=en_US.utf-8
################# DON NOT CHANGE THINGS HERE UNLESS YOU KNOW WHAT YOU ARE DOING ###############
# This script is a modification to the implementation suggest by gregSchwartz18 here:
# https://github.com/ray-project/ray/issues/826#issuecomment-522116599
redis_password=$(uuidgen)
export redis_password

nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST) # Getting the node names
nodes_array=($nodes)

node_1=${nodes_array[0]}
ip=$(srun --nodes=1 --ntasks=1 -w $node_1 hostname --ip-address) # making redis-address

if [[ $ip == *" "* ]]; then
  IFS=' ' read -ra ADDR <<<"$ip"
  if [[ ${#ADDR[0]} > 16 ]]; then
    ip=${ADDR[1]}
  else
    ip=${ADDR[0]}
  fi
  echo "We detect space in ip! You are using IPV6 address. We split the IPV4 address as $ip"
fi

port=6379
ip_head=$ip:$port
export ip_head
echo "IP Head: $ip_head"

echo "STARTING HEAD at $node_1"
# srun --nodes=1 --ntasks=1 -w $node_1 start-head.sh $ip $redis_password &
srun --nodes=1 --ntasks=1 -w $node_1 \
  ray start --head --node-ip-address=$ip --port=6379 --redis-password=$redis_password --block &
sleep 30

worker_num=$(($SLURM_JOB_NUM_NODES - 1)) #number of nodes other than the head node
for ((i = 1; i <= $worker_num; i++)); do
  node_i=${nodes_array[$i]}
  echo "STARTING WORKER $i at $node_i"
  srun --nodes=1 --ntasks=1 -w $node_i ray start --address $ip_head --redis-password=$redis_password --block &
  sleep 5
done

##############################################################################################

#mpirun -np 1 --bind-to none python ./src/marl.py train --config "/mainfs/home/fb1n15/MARL-ReverseAuction/marl-edge-cloud/simulations/hyperparamter_optimisation__model_depth/config_HPO_iridis5.yaml"  --env_seed 1

python ./src/marl.py train --config "/mainfs/home/fb1n15/MARL-ReverseAuction/marl-edge-cloud/simulations/hyperparamter_optimisation__model_depth/config_HPO_iridis5.yaml"  --env_seed 1

#START=1
#END=2
#for SEED in $(seq "$START" "$END"); do
#  echo "Starting a Job with SEED=$SEED" # print the seed value
#  python ./src/marl.py train --config $CONFIG_FILE --env_seed "$SEED"
#done
#
