# Multi-Agent Reinforcement Learning for Disaster Response
A third year individual project.

## Build instructions for MacOS

Install Python 3.6.4

Install CUDA toolkit 10.2 (https://developer.nvidia.com/cuda-10.2-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exenetwork), and cudnn 8.1.1 (https://developer.nvidia.com/rdp/cudnn-download)

For cudnn installation guide, see: https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#install-windows

`python3 -m venv venv`

`source venv/bin/activate`

`pip install --upgrade pip`

`pip install -r requirements.txt`

For Windows

`pip install -U https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-2.0.0.dev0-cp36-cp36m-win_amd64.whl`

For MacOS

`pip install -U https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-2.0.0.dev0-cp36-cp36m-macosx_10_13_intel.whl`

`pip install torch==1.8.1+cu102 torchvision==0.9.1+cu102 torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html`

`pip install torch==1.8.1 torchvision==0.9.1 torchaudio===0.8.1 -f 
https://download.pytorch.org/whl/torch_stable.html`


## Build Instructions on Lycuim 5

### First time setup

`module load python`

`python3 -m venv venv`

`source venv/bin/activate`

`pip install --upgrade pip`

`pip install -r requirements.txt`

`pip install -U https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-2.0.0.dev0-cp36-cp36m-manylinux2014_x86_64.whl`

### Queue Job

`sbatch --array=<config_indices> -p lyceum run_<experiment_name>.sh`

### Set up tensorboard

(Optional) Use tmux: `module load tmux` then `tmux`

`tensorboard --logdir="results/<experiment>"`

or

`tensorboard dev upload --logdir "results/<experiment>" --name "MARL Disaster Response" --description "Training drones in a gridworld disaster simulation environment"`

(`ctrl-b d` to exit tmux window)