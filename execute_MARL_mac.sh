#!/bin/bash

### execute the trained agent/policy (RC = 0.7)
#CONFIG_FILE="/Users/fan/OneDrive - University of Southampton/My-Projects/Edge-Cloud-Resource-Allocation/marl-edge-cloud/simulations/resource_coefficient/config_HPO_execution_local_S.yaml"  # config file for RLlib
#CHECKPOINT_DIR="/Users/fan/OneDrive - University of Southampton/My-Projects/Edge-Cloud-Resource-Allocation/marl-edge-cloud/results/resource_coefficient=0.7/PPO_EdgeCloudEnv1_baf33_00000_0_2022-03-29_21-06-33/checkpoint_000033/checkpoint-33"  # checkpoint dir for RLlib
## execute the trained agent/policy
#python "./src/marl.py" run --config "$CONFIG_FILE" --checkpoint "$CHECKPOINT_DIR" --env_seed 1


## execute the trained agent/policy (RC = 1)
#CONFIG_FILE="/Users/fan/OneDrive - University of Southampton/My-Projects/Edge-Cloud-Resource-Allocation/marl-edge-cloud/configs/config_HPO_execution_local_M.yaml"  # config file for RLlib
#CHECKPOINT_DIR="/Users/fan/OneDrive - University of Southampton/My-Projects/Edge-Cloud-Resource-Allocation/marl-edge-cloud/results/resource_coefficient=1/PPO_EdgeCloudEnv1_42ae1_00000_0_2022-03-29_17-21-17/checkpoint_000033/checkpoint-33"  # checkpoint dir for RLlib

# execute the trained agent/policy (RC = 1.3)
CONFIG_FILE="/Users/fan/OneDrive - University of Southampton/My-Projects/Edge-Cloud-Resource-Allocation/marl-edge-cloud/simulations/resource_coefficient/config_HPO_execution_local_L.yaml"  # config file for RLlib
CHECKPOINT_DIR="/Users/fan/OneDrive - University of Southampton/My-Projects/Edge-Cloud-Resource-Allocation/marl-edge-cloud/results/resource_coefficient=1.3/PPO_EdgeCloudEnv1_06d60_00000_0_2022-03-29_21-37-18/checkpoint_000033/checkpoint-33"  # checkpoint dir for RLlib
python "./src/marl.py" run --config "$CONFIG_FILE" --checkpoint "$CHECKPOINT_DIR" --env_seed 1


# execute the trained agent/policy (RC = 1.6)
CONFIG_FILE="/Users/fan/OneDrive - University of Southampton/My-Projects/Edge-Cloud-Resource-Allocation/marl-edge-cloud/simulations/resource_coefficient/config_HPO_execution_local_XL.yaml"  # config file for RLlib
CHECKPOINT_DIR="/Users/fan/OneDrive - University of Southampton/My-Projects/Edge-Cloud-Resource-Allocation/marl-edge-cloud/results/resource_coefficient=1.6/PPO_EdgeCloudEnv1_e0be7_00000_0_2022-03-29_22-26-21/checkpoint_000033/checkpoint-33"  # checkpoint dir for RLlib
python "./src/marl.py" run --config "$CONFIG_FILE" --checkpoint "$CHECKPOINT_DIR" --env_seed 1
