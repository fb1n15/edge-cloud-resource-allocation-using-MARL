# Multi-Agent Reinforcement Learning for Disaster Response
A third year individual project.

## Build Instructions for Iridis 4

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

`python3 -m venv venv`

`source venv/bin/activate`

`pip install --upgrade pip`

`pip install requirements.txt`

### Queue Job

`sbatch -p lyceum run.sh`

