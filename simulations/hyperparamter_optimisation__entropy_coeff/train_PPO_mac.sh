#!/bin/bash

# run just one trial
CONFIG_FILE="/Users/fan/OneDrive - University of Southampton/My-Projects/Edge-Cloud-Resource-Allocation/marl-edge-cloud/simulations/hyperparamter_optimisation__entropy_coeff/config_HPO_mac.yaml"
python ../../src/marl.py train --config "$CONFIG_FILE" --env_seed 1










#python "./src/marl.py" train --config "$CONFIG_FILE" --env_seed 2
#python "./src/marl.py" train --config "$CONFIG_FILE" --env_seed 3













#for CONFIG_FILE in "./configs/experiments/edge_cloud/hyperparameters/cpu_ppo_fc_independent_with_history.yaml" "./configs/experiments/edge_cloud/hyperparameters/cpu_ppo_fc_independent_without_history.yaml" "Users/fan/OneDrive - University of Southampton/My-Projects/MARL-Jack/marl-disaster-relief/configs/experiments/edge_cloud/hyperparameters/cpu_DQN_fc_independent_with_history.yaml" "./configs/experiments/edge_cloud/hyperparameters/cpu_DQN_fc_independent_without_history.yaml"
#do
#    python "./src/marl.py" train --config "$CONFIG_FILE"
#done

## run just one job (ppo with history)
#CONFIG_FILE="./configs/experiments/edge_cloud/hyperparameters/cpu_ppo_fc_independent_with-history_5-actions_env-2_local.yaml"
#python "./src/marl.py" train --config "$CONFIG_FILE"



## run just one job PPO without history
#CONFIG_FILE="./configs/experiments/edge_cloud/hyperparameters/cpu_ppo_fc_independent_without_history.yaml"
#python "./src/marl.py" train --config "$CONFIG_FILE"

## run just one job DQN with history
#CONFIG_FILE="./configs/experiments/edge_cloud/hyperparameters/cpu_DQN_fc_independent_with_history.yaml"
#python "./src/marl.py" train --config "$CONFIG_FILE"

## run just one job DQN without history
#CONFIG_FILE="./configs/experiments/edge_cloud/hyperparameters/cpu_DQN_fc_independent_without_history.yaml"
#python "./src/marl.py" train --config "$CONFIG_FILE"
