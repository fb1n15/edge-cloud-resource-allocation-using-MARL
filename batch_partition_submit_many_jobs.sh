#!/bin/bash

# submit parallel simulations jobs
# run five trials for each hyperparameter setting
for i in $(seq 7 100 500)  # 100, 200, 300, ..., 1000
do
  echo $i
  sbatch ./simulations/hyperparamter_optimisation__entropy_coeff/batch_partition_submit_a_job.sh $i
  sbatch ./simulations/hyperparamter_optimisation__learning_rate/batch_partition_submit_a_job.sh $i
  sbatch ./simulations/hyperparamter_optimisation__model_depth/batch_partition_submit_a_job.sh $i
  sbatch ./simulations/hyperparamter_optimisation__PPO_clip_param/batch_partition_submit_a_job.sh $i
  sbatch ./simulations/hyperparamter_optimisation__train_batch_size/batch_partition_submit_a_job.sh $i
  sbatch ./simulations/hyperparamter_optimisation__auction_type/batch_partition_submit_a_job.sh $i
done



