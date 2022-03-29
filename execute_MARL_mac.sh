#!/bin/bash

## run just one trial
#CONFIG_FILE="/Users/fan/OneDrive - University of Southampton/My-Projects/Edge-Cloud-Resource-Allocation/marl-edge-cloud/configs/config_HPO_execution_local_M.yaml"  # config file for RLlib
#CHECKPOINT_DIR="/Users/fan/OneDrive - University of Southampton/My-Projects/Edge-Cloud-Resource-Allocation/marl-edge-cloud/results/Hyperparameter_Optimization__train_batch_size/PPO_EdgeCloudEnv1_01aa3_00002_2_clip_param=0.3,entropy_coeff=0.01,auction_type=first-price,lambda=0.95,lr=0.0001,layers=[256],obse_2022-03-28_21-39-29/checkpoint_000033/checkpoint-33"  # checkpoint dir for RLlib
## execute the trained agent/policy
#python "./src/marl.py" run --config "$CONFIG_FILE" --checkpoint "$CHECKPOINT_DIR" --env_seed 1


## execute the trained agent/policy (RC = 1)
#CONFIG_FILE="/Users/fan/OneDrive - University of Southampton/My-Projects/Edge-Cloud-Resource-Allocation/marl-edge-cloud/configs/config_HPO_execution_local_M.yaml"  # config file for RLlib
#CHECKPOINT_DIR="/Users/fan/OneDrive - University of Southampton/My-Projects/Edge-Cloud-Resource-Allocation/marl-edge-cloud/results/resource_coefficient=1/PPO_EdgeCloudEnv1_42ae1_00000_0_2022-03-29_17-21-17/checkpoint_000033/checkpoint-33"  # checkpoint dir for RLlib

# execute the trained agent/policy (RC = 2)
CONFIG_FILE="/Users/fan/OneDrive - University of Southampton/My-Projects/Edge-Cloud-Resource-Allocation/marl-edge-cloud/configs/config_HPO_execution_local_XL.yaml"  # config file for RLlib
CHECKPOINT_DIR="/Users/fan/OneDrive - University of Southampton/My-Projects/Edge-Cloud-Resource-Allocation/marl-edge-cloud/results/resource_coefficient=2/PPO_EdgeCloudEnv1_d1508_00000_0_2022-03-29_18-29-42/checkpoint_000033/checkpoint-33"  # checkpoint dir for RLlib

python "./src/marl.py" run --config "$CONFIG_FILE" --checkpoint "$CHECKPOINT_DIR" --env_seed 1
