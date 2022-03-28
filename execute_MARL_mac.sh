#!/bin/bash

# run just one trial
CONFIG_FILE="/Users/fan/OneDrive - University of Southampton/My-Projects/Edge-Cloud-Resource-Allocation/marl-edge-cloud/configs/config_HPO_execution_local.yaml"  # config file for RLlib
CHECKPOINT_DIR="/Users/fan/OneDrive - University of Southampton/My-Projects/Edge-Cloud-Resource-Allocation/marl-edge-cloud/results/Hyperparameter_Optimization__cooperative/PPO_EdgeCloudEnv1_f26f6_00000_0_clip_param=0.3,entropy_coeff=0.01,auction_type=first-price,cooperative=False,lambda=0.95,lr=0.0001_2022-03-27_17-42-53/checkpoint_000097/checkpoint-97"  # checkpoint dir for RLlib
# execute the trained agent/policy
python "./src/marl.py" run --config "$CONFIG_FILE" --checkpoint "$CHECKPOINT_DIR" --env_seed 1
