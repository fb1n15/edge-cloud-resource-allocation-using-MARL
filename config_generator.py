# read the config template and modify it to generate the config file used in the checkpoint
import random  # for generating random numbers

import yaml  # for reading the config template
from pprint import pprint  # for printing the config file

with open("./configs/config_template.yaml", 'r') as stream:
    config = yaml.load(stream, Loader=yaml.FullLoader)
    # print("config: ")
    # pprint(config)

seed = 1
random.seed(seed)  # set the seed for the random number generator

# modify the config template
# set the seeds
config['env-config']['seed'] = random.randint(0, 10000)
print("environment seed: ", config['env-config']['seed'])
config['trainer-config']['seed'] = random.randint(0, 10000)
print("RL algorithm seed: ", config['trainer-config']['seed'])

# set the synthetic data
# number of fog nodes
config['env-config']['n_nodes'] = 6
config['env-config']['num_agents'] = 6
config['env-config']['duration'] = 7
config['env-config']['avg_resource_capacity'] = {0: [6, 6, 6], 1: [6, 6, 6], 2: [6, 6, 6],
                                         3: [6, 6, 6], 4: [4, 4, 4], 5: [4, 4, 4]}
config['env-config']['avg_unit_cost'] = {0: [2, 2, 2], 1: [2, 2, 2], 2: [2, 2, 2],
                                         3: [2, 2, 2], 4: [4, 4, 4], 5: [4, 4, 4]}

# stop the checkpoint after a certain number of timesteps
config['stop']['timesteps_total'] = 100_000
config['samples'] = 1  # number of samples/trials
# set the auction type
config['env-config']['auction_type']['gridsearch'] = ['first-price']

# hyperparameters setting
# learning rate
config['trainer-config']['lr']['gridsearch'] = [0.0001]
# clipping range
config['trainer-config']['clip_param']['gridsearch'] = [0.1, 0.2, 0.3]
# config['trainer-config']['clip_param']['gridsearch'] = [0.3]
# controls the entropy trade-off in the model.
config['trainer-config']['lambda']['gridsearch'] = [0.9, 0.95, 1.0]
# config['trainer-config']['lambda']['gridsearch'] = [0.9]
# controls the entropy trade-off in the model.
config['trainer-config']['entropy_coeff']['gridsearch'] = [0, 0.01]
# config['trainer-config']['entropy_coeff']['gridsearch'] = [0.01]
# train batch size
config['trainer-config']['train_batch_size']['gridsearch'] = [3000]
# mini-batch size
config['trainer-config']['sgd_minibatch_size']['gridsearch'] = [128]

# set the number of CPU cores and GPUs
config['trainer-config']['num_workers'] = 18  # number of CPU cores
config['trainer-config']['num_gpus'] = 0  # number of GPUs

# set the name of the trial
config['name'] = f'5-actions_no-history_revenue'
# write the config file
with open(f"./configs/config_file.yaml", 'w') as outfile:
    yaml.dump(config, outfile)

# for auction_type in ['first-price', 'second-price']:
#     # set the auction type
#     config['env-config']['auction_type'] = auction_type
#     # set the name of the trial
#     config['name'] = f'5-actions_{auction_type}_no-history_revenue'
#     # write the config file
#     with open(f"./configs/config_{auction_type}.yaml", 'w') as outfile:
#         yaml.dump(config, outfile)
