#!/bin/bash

### Train the model (3 fog nodes)
#CONFIG_FILE="/Users/fan/Library/Mobile Documents/com~apple~CloudDocs/GitHub_Projects/edge-cloud-resource-allocation-using-MARL/simulations/number_of_fog_nodes/config_HPO_execution_local_3_nodes.yaml"
## execute the trained agent/policy
##for SEED in {1..10}
##do
##  python "./src/marl.py" train --config "$CONFIG_FILE" --env_seed "$SEED"
##done
#
#SEED=1
#python "./src/marl.py" train --config "$CONFIG_FILE" --env_seed "$SEED"


### Train the model (6 fog nodes)
#CONFIG_FILE="/Users/fan/Library/Mobile Documents/com~apple~CloudDocs/GitHub_Projects/edge-cloud-resource-allocation-using-MARL/simulations/number_of_fog_nodes/config_HPO_execution_local_6_nodes.yaml"
## execute the trained agent/policy
#for SEED in {1..10}
#do
#  python "./src/marl.py" train --config "$CONFIG_FILE" --env_seed "$SEED"
#done
#
### Train the model (9 fog nodes)
#CONFIG_FILE="/Users/fan/Library/Mobile Documents/com~apple~CloudDocs/GitHub_Projects/edge-cloud-resource-allocation-using-MARL/simulations/number_of_fog_nodes/config_HPO_execution_local_9_nodes.yaml"
## execute the trained agent/policy
#for SEED in {1..10}
#do
#  python "./src/marl.py" train --config "$CONFIG_FILE" --env_seed "$SEED"
#done
#

## Train the model (12 fog nodes)
CONFIG_FILE="/Users/fan/Library/Mobile Documents/com~apple~CloudDocs/GitHub_Projects/edge-cloud-resource-allocation-using-MARL/simulations/number_of_fog_nodes/config_HPO_execution_local_12_nodes.yaml"
# execute the trained agent/policy
#for SEED in {1..10}
#do
#  python "./src/marl.py" train --config "$CONFIG_FILE" --env_seed "$SEED"
#done

SEED=1 # seed for the experiment
echo "fog nodes number: 12"
echo "SEED: $SEED"
python "./src/marl.py" train --config "$CONFIG_FILE" --env_seed "$SEED"


### execute the trained agent/policy (RC = 0.7)
#CONFIG_FILE="/Users/fan/OneDrive - University of Southampton/My-Projects/Edge-Cloud-Resource-Allocation/marl-edge-cloud/simulations/resource_coefficient/config_HPO_execution_local_S.yaml"  # config file for RLlib
#CHECKPOINT_DIR="/Users/fan/OneDrive - University of Southampton/My-Projects/Edge-Cloud-Resource-Allocation/marl-edge-cloud/results/resource_coefficient=0.7/PPO_EdgeCloudEnv1_baf33_00000_0_2022-03-29_21-06-33/checkpoint_000033/checkpoint-33"  # checkpoint dir for RLlib
## execute the trained agent/policy
#python "./src/marl.py" run --config "$CONFIG_FILE" --checkpoint "$CHECKPOINT_DIR" --env_seed 1


## execute the trained agent/policy (RC = 1)
#CONFIG_FILE="/Users/fan/OneDrive - University of Southampton/My-Projects/Edge-Cloud-Resource-Allocation/marl-edge-cloud/configs/config_execution_n_tasks_local_M.yaml"  # config file for RLlib
#CHECKPOINT_DIR="/Users/fan/OneDrive - University of Southampton/My-Projects/Edge-Cloud-Resource-Allocation/marl-edge-cloud/results/resource_coefficient=1/PPO_EdgeCloudEnv1_42ae1_00000_0_2022-03-29_17-21-17/checkpoint_000033/checkpoint-33"  # checkpoint dir for RLlib
#python "./src/marl.py" run --config "$CONFIG_FILE" --checkpoint "$CHECKPOINT_DIR" --env_seed 1

## execute the trained agent/policy (RC = 1.3)
#CONFIG_FILE="/Users/fan/OneDrive - University of Southampton/My-Projects/Edge-Cloud-Resource-Allocation/marl-edge-cloud/simulations/resource_coefficient/config_HPO_execution_local_L.yaml"  # config file for RLlib
#CHECKPOINT_DIR="/Users/fan/OneDrive - University of Southampton/My-Projects/Edge-Cloud-Resource-Allocation/marl-edge-cloud/results/resource_coefficient=1.3/PPO_EdgeCloudEnv1_06d60_00000_0_2022-03-29_21-37-18/checkpoint_000033/checkpoint-33"  # checkpoint dir for RLlib
#python "./src/marl.py" run --config "$CONFIG_FILE" --checkpoint "$CHECKPOINT_DIR" --env_seed 1
#
#
## execute the trained agent/policy (RC = 1.6)
#CONFIG_FILE="/Users/fan/OneDrive - University of Southampton/My-Projects/Edge-Cloud-Resource-Allocation/marl-edge-cloud/simulations/resource_coefficient/config_HPO_execution_local_XL.yaml"  # config file for RLlib
#CHECKPOINT_DIR="/Users/fan/OneDrive - University of Southampton/My-Projects/Edge-Cloud-Resource-Allocation/marl-edge-cloud/results/resource_coefficient=1.6/PPO_EdgeCloudEnv1_e0be7_00000_0_2022-03-29_22-26-21/checkpoint_000033/checkpoint-33"  # checkpoint dir for RLlib
#python "./src/marl.py" run --config "$CONFIG_FILE" --checkpoint "$CHECKPOINT_DIR" --env_seed 1


## execute the trained agent/policy (RC = 1, n_actions = 20)
#CONFIG_FILE="/Users/fan/OneDrive - University of Southampton/My-Projects/Edge-Cloud-Resource-Allocation/marl-edge-cloud/simulations/number_of_tasks/config_execution_n_tasks_local_S.yaml"  # config file for RLlib
#CHECKPOINT_DIR="/Users/fan/OneDrive - University of Southampton/My-Projects/Edge-Cloud-Resource-Allocation/marl-edge-cloud/results/resource_coefficient=1/PPO_EdgeCloudEnv1_42ae1_00000_0_2022-03-29_17-21-17/checkpoint_000033/checkpoint-33"  # checkpoint dir for RLlib
#python "./src/marl.py" run --config "$CONFIG_FILE" --checkpoint "$CHECKPOINT_DIR" --env_seed 1
#
## execute the trained agent/policy (RC = 1, n_actions = 40)
#CONFIG_FILE="/Users/fan/OneDrive - University of Southampton/My-Projects/Edge-Cloud-Resource-Allocation/marl-edge-cloud/simulations/number_of_tasks/config_execution_n_tasks_local_M.yaml"  # config file for RLlib
#CHECKPOINT_DIR="/Users/fan/OneDrive - University of Southampton/My-Projects/Edge-Cloud-Resource-Allocation/marl-edge-cloud/results/resource_coefficient=1/PPO_EdgeCloudEnv1_42ae1_00000_0_2022-03-29_17-21-17/checkpoint_000033/checkpoint-33"  # checkpoint dir for RLlib
#python "./src/marl.py" run --config "$CONFIG_FILE" --checkpoint "$CHECKPOINT_DIR" --env_seed 1

## execute the trained agent/policy (RC = 1, n_actions = 60)
#CONFIG_FILE="/Users/fan/OneDrive - University of Southampton/My-Projects/Edge-Cloud-Resource-Allocation/marl-edge-cloud/simulations/number_of_tasks/config_execution_n_tasks_local_L.yaml"  # config file for RLlib
#CHECKPOINT_DIR="/Users/fan/OneDrive - University of Southampton/My-Projects/Edge-Cloud-Resource-Allocation/marl-edge-cloud/results/resource_coefficient=1/PPO_EdgeCloudEnv1_42ae1_00000_0_2022-03-29_17-21-17/checkpoint_000033/checkpoint-33"  # checkpoint dir for RLlib
#python "./src/marl.py" run --config "$CONFIG_FILE" --checkpoint "$CHECKPOINT_DIR" --env_seed 1


## execute the trained agent/policy (RC = 1, n_actions = 80)
#CONFIG_FILE="/Users/fan/OneDrive - University of Southampton/My-Projects/Edge-Cloud-Resource-Allocation/marl-edge-cloud/simulations/number_of_tasks/config_execution_n_tasks_local_XL.yaml"  # config file for RLlib
#CHECKPOINT_DIR="/Users/fan/OneDrive - University of Southampton/My-Projects/Edge-Cloud-Resource-Allocation/marl-edge-cloud/results/resource_coefficient=1/PPO_EdgeCloudEnv1_42ae1_00000_0_2022-03-29_17-21-17/checkpoint_000033/checkpoint-33"  # checkpoint dir for RLlib
#python "./src/marl.py" run --config "$CONFIG_FILE" --checkpoint "$CHECKPOINT_DIR" --env_seed 1
