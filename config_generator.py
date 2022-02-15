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
config['env-config']['seed'] = random.randint(0, 10000)
print("environment seed: ", config['env-config']['seed'])
config['trainer-config']['seed'] = random.randint(0, 10000)
print("RL algorithm seed: ", config['trainer-config']['seed'])
config['trainer-config']['train_batch_size']['gridsearch'] = [1000]
config['stop']['timesteps_total'] = 3000


# write the config file
with open("./configs/config_local.yaml", 'w') as outfile:
    yaml.dump(config, outfile)