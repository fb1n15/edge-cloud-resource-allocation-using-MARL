#!/bin/bash

# train for different auction types
#CONFIG_FILE="/Users/fan/OneDrive - University of Southampton/My-Projects/Edge-Cloud-Resource-Allocation/marl-edge-cloud/simulations/hyperparamter_optimisation__auction_type/config_HPO_mac.yaml"
#python "./src/marl.py" train --config "$CONFIG_FILE" --env_seed 1

## train for different train_batch_size
#CONFIG_FILE="/Users/fan/OneDrive - University of Southampton/My-Projects/Edge-Cloud-Resource-Allocation/marl-edge-cloud/simulations/hyperparamter_optimisation__train_batch_size/config_HPO_mac.yaml"
#python "./src/marl.py" train --config "$CONFIG_FILE" --env_seed 1

## train for different number_of_actions
#CONFIG_FILE="/Users/fan/OneDrive - University of Southampton/My-Projects/Edge-Cloud-Resource-Allocation/marl-edge-cloud/simulations/hyperparamter_optimisation__number_of_actions/config_HPO_mac.yaml"
#python "./src/marl.py" train --config "$CONFIG_FILE" --env_seed 1


## train for different learning_rate
#CONFIG_FILE="/Users/fan/OneDrive - University of Southampton/My-Projects/Edge-Cloud-Resource-Allocation/marl-edge-cloud/simulations/hyperparamter_optimisation__learning_rate/config_HPO_mac.yaml"
#python "./src/marl.py" train --config "$CONFIG_FILE" --env_seed 1
#
## train for different clip_param
#CONFIG_FILE="/Users/fan/OneDrive - University of Southampton/My-Projects/Edge-Cloud-Resource-Allocation/marl-edge-cloud/simulations/hyperparamter_optimisation__PPO_clip_param/config_HPO_mac.yaml"
#python "./src/marl.py" train --config "$CONFIG_FILE" --env_seed 1


## train for different model_depths
#CONFIG_FILE="/Users/fan/OneDrive - University of Southampton/My-Projects/Edge-Cloud-Resource-Allocation/marl-edge-cloud/simulations/hyperparamter_optimisation__model_depth/config_HPO_mac.yaml"
#python "./src/marl.py" train --config "$CONFIG_FILE" --env_seed 1

## train for different resource coefficients
#CONFIG_FILE="/Users/fan/OneDrive - University of Southampton/My-Projects/Edge-Cloud-Resource-Allocation/marl-edge-cloud/simulations/resource_coefficient/config_HPO_execution_local_S.yaml"
#python "./src/marl.py" train --config "$CONFIG_FILE" --env_seed 1

#CONFIG_FILE="/Users/fan/OneDrive - University of Southampton/My-Projects/Edge-Cloud-Resource-Allocation/marl-edge-cloud/simulations/resource_coefficient/config_execution_n_tasks_local_M.yaml"
#python "./src/marl.py" train --config "$CONFIG_FILE" --env_seed 1

#CONFIG_FILE="/Users/fan/OneDrive - University of Southampton/My-Projects/Edge-Cloud-Resource-Allocation/marl-edge-cloud/simulations/resource_coefficient/config_HPO_execution_local_L.yaml"
#python "./src/marl.py" train --config "$CONFIG_FILE" --env_seed 1
#
#CONFIG_FILE="/Users/fan/OneDrive - University of Southampton/My-Projects/Edge-Cloud-Resource-Allocation/marl-edge-cloud/simulations/resource_coefficient/config_HPO_execution_local_XL.yaml"
#python "./src/marl.py" train --config "$CONFIG_FILE" --env_seed 1






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
