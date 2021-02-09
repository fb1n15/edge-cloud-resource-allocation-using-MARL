#!/bin/bash
#PBS -l walltime=00:05:00

# Adapted from https://github.com/ray-project/ray/issues/10466


# Load conda environement
module load singularity/3.2.0
echo "starting"
# Navigate to working dir
cd $PBS_O_WORKDIR

ln -s $PWD $PBS_O_WORKDIR/$PBS_JOBID

cd $PBS_O_WORKDIR

jobnodes=`uniq -c ${PBS_NODEFILE} | awk -F. '{print $1 }' | awk '{print $2}' | paste -s -d " "`

thishost=`uname -n | awk -F. '{print $1.}'`
thishostip=`hostname -i`
rayport=6379

thishostNport="${thishostip}:${rayport}"
echo "Allocate Nodes = <$jobnodes>"

echo "set up ray cluster..."
for n in `echo ${jobnodes}`
do
        if [[ ${n} == "${thishost}" ]]
        then
                echo "first allocate node - use as headnode ..."
                module load PyTorch
                ray start --head
                sleep 5
        else
                ssh ${n}  $PBS_O_WORKDIR/startWorkerNode.sh ${thishostNport}
                sleep 10
        fi
done

export PYTHONPATH="${PYTHONPATH}:/lyceum/jp6g18/marl_disaster_relief/src"
singularity exec image.sif python src/marl-disaster.py train

#rm $PBS_O_WORKDIR/$PBS_JOBID


echo "Finishing job"