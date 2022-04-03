#!/bin/bash

# submit parallel simulations jobs
for i in $(seq 10 100 3000)  # 0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000
do
  sbatch "/mainfs/home/fb1n15/MARL-ReverseAuction/marl-edge-cloud/simulations/hyperparamter_optimisation__learning_rate/batch_partition_submit_a_job__learning_rate.sh" "$i"
done