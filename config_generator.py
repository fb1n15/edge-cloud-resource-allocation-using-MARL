# read the config template and modify it to generate the config file used in the experiment

import yaml
from pprint import pprint

with open("./configs/config_template.yaml", 'r') as stream:
    config = yaml.load(stream, Loader=yaml.FullLoader)
    print("config: ")
    pprint(config)

# modify the config template
config['env-config']['seed'] = 777
config['stop']['timesteps_total'] = 100

# write the config file
with open("./configs/config_local.yaml", 'w') as outfile:
    yaml.dump(config, outfile)