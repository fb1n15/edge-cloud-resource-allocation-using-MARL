#!/bin/bash

# run just one trial
CONFIG_FILE="./configs/local_config_1.yaml"
python "./src/marl.py" train --config "$CONFIG_FILE"













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
