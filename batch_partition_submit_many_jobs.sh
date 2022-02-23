#!/bin/bash

# submit parallel simulations jobs
for i in $(seq 1 100 3000)  # 0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000
do
  sbatch ./batch_partition_submit_a_job.sh "$i"
done