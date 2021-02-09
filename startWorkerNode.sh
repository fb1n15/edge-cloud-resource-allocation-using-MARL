#!/bin/bash -l
source $HOME/.bashrc
cd $PBS_O_WORKDIR
param1=$1
destnode=`uname -n`
echo "destnode is = [$destnode]"

# Join the ray cluster
module load singularity/3.2.0
singularity exec image.sif ray start --address="${param1}" --redis-password='5241590000000000'
