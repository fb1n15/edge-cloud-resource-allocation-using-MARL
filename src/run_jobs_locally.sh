#!/bin/bash

#for CONFIG_FILE in "Users/fan/OneDrive - University of Southampton/My-Projects/MARL-Jack/marl-disaster-relief/configs/experiments/edge_cloud/hyperparameters/cpu_DQN_fc_independent_with_history.yaml" "/Users/fan/OneDrive - University of Southampton/My-Projects/MARL-Jack/marl-disaster-relief/configs/experiments/edge_cloud/hyperparameters/cpu_DQN_fc_independent_without_history.yaml" "/Users/fan/OneDrive - University of Southampton/My-Projects/MARL-Jack/marl-disaster-relief/configs/experiments/edge_cloud/hyperparameters/cpu_ppo_fc_independent_with_history.yaml" "/Users/fan/OneDrive - University of Southampton/My-Projects/MARL-Jack/marl-disaster-relief/configs/experiments/edge_cloud/hyperparameters/cpu_ppo_fc_independent_without_history.yaml"
#do
#    python src/marl.py train --config "$CONFIG_FILE"
#done

# run just one job
CONFIG_FILE="/Users/fan/OneDrive - University of Southampton/My-Projects/MARL-Jack/marl-disaster-relief/configs/experiments/edge_cloud/hyperparameters/cpu_ppo_fc_independent_with_history.yaml"
python marl.py train --config "$CONFIG_FILE"