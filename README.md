# Multi-Agent Reinforcement Learning for Disaster Response
A third year individual project.

## Build instructions for Windows 10

Install Python 3.6

Install CUDA toolkit 10.2 (https://developer.nvidia.com/cuda-10.2-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exenetwork), and cudnn 8.1.1 (https://developer.nvidia.com/rdp/cudnn-download)

For cudnn installation guide, see: https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#install-windows

`python3 -m venv venv`

`source venv/bin/activate`

`pip install --upgrade pip`

`pip install -r requirements.txt`

`pip install -U https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-2.0.0.dev0-cp36-cp36m-win_amd64.whl`

`pip install torch==1.8.1+cu102 torchvision==0.9.1+cu102 torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html`

## Build Instructions for Lycuim 4

### Create docker image on docker hub

`docker build --tag marl-diaster-image .`

`docker tag marl-diaster-image jparons74/marl-disaster:v2 `

`docker push jparons74/marl-disaster:v2`

### Run on Iridis 4 with singularity

Create singularity container with 

`module load singularity`

`singularity build image.sif docker://jparons74/marl-disaster:v4`


Set python path to be the sources root of src/

`export PYTHONPATH="${PYTHONPATH}:/lyceum/jp6g18/git/marl_disaster_relief/src"`

Run with 

`singularity exec image.sif python src/marl-disaster.py train`

or

`qsub run.sh -l nodes=2:ppn=16`

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