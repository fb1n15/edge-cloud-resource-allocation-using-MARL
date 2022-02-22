# read the config template and modify it to generate the config file used in the experiment
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
# stop the experiment after a certain number of timesteps
config['stop']['timesteps_total'] = 50000
config['samples'] = 1  # number of samples/trials
# set the auction type
config['env-config']['auction_type']['gridsearch'] = ['first-price', 'second-price']

# hyperparameters setting
# learning rate
config['trainer-config']['lr']['gridsearch'] = [0.0001, 0.001]
# train batch size
config['trainer-config']['train_batch_size']['gridsearch'] = [3000, 4000]
# mini-batch size
config['trainer-config']['sgd_minibatch_size']['gridsearch'] = [64, 128]

# set the number of CPU cores and GPUs
config['trainer-config']['num_workers'] = 5  # number of CPU cores
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
