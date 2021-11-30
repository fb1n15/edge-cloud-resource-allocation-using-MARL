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

## Instructions for Iridis5

## Run on the login node
```shell
cd MARL-ReverseAuction/marl-edge-cloud
module load conda/py3-latest
source activate jack
bash ./src/run_jobs_locally.sh
```

## Request an interactive node
```shell
cd MARL-ReverseAuction/marl-edge-cloud
module load conda/py3-latest
source activate jack
sinteractive -p gpu --gres=gpu:2 --time=04:00:00 --mem=16GB --job-name=marl-edge-cloud
```

then use it like a local machine, e.g.,
```shell
bash ./run_jobs_locally.sh
```

## Submit batch jobs

```sbatch --array=<config_indices> -p batch run_<experiment_name>.sh```
or
```sbatch --array=<config_indices> -p gpu run_<experiment_name>.sh```


## Local port forwarding

### method 1

First, add the following lines to your `.ssh/config` on your laptop. Look here if you want a slightly more detailed instruction on how to access your computer remotely.
```shell
Host workstation
     HostName <hostname>
     User <username>
     Port <port>
     LocalForward 8888 localhost:8888
     LocalForward 6006 localhost:6006
```

### method 2

```shell
ssh -L 8000:localhost:8888 fb1n15@iridis5_b.soton.ac.uk
```

### notes

- You can use `ssh -L` to forward a port from your local machine to a remote machine.
- to run tensorboard, typing `tensorboard --logdir="results/<experiment>" --port=6006` on the logging node (not on the iteractive job node). (6006 can be changed to any port number) 
- [source1](https://towardsdatascience.com/jupyter-and-tensorboard-in-tmux-5e5d202a4fb6), [source2](https://www.digitalocean.com/community/tutorials/how-to-install-run-connect-to-jupyter-notebook-on-remote-server)


## ToDos

- move iridis5 output file (slurm-***.out) to a folder (maybe called iridis5-output)
- how to put other parameters to the name of the RL results? So that I can distinguish different simulations.