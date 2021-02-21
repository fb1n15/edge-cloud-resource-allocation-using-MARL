# Use an official ubuntu image
FROM nvidia/cuda:10.2-devel-ubuntu16.04

# Install any needed packages
#RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get -y update
RUN apt-get install -y software-properties-common

RUN add-apt-repository ppa:deadsnakes/ppa \
&& apt-get update \
&& apt-get -y install python3.6 python3-pip
RUN apt-get install -y python3.6-dev

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.6 2
RUN update-alternatives --config python
RUN update-alternatives  --set python /usr/bin/python3.6

RUN python -m pip install --upgrade "pip < 21.0"

COPY requirements.txt .

# Install torch
#RUN python -m pip install torch==1.7.1+cu92 torchvision==0.8.2+cu92 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
RUN python -m pip install torch torchvision

RUN python -m pip install -r requirements.txt

RUN apt -y install libgl1-mesa-glx

#RUN export CUDA_HOME=/local/software/cuda/10.0 \
#&& export PATH=/local/software/cuda/10.0/bin:$PATH \
#&& export LD_LIBRARY_PATH=/local/software/cuda/10.0/lib64:$LD_LIBRARY_PATH \
#&& export LD_LIBRARY_PATH=/local/software/cuda/10.0/lib64/stubs:$LD_LIBRARY_PATH \
#&& export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH \
#&& export LIBRARY_PATH=/local/software/cuda/10.0/lib64:$LIBRARY_PATH \
#&& export LIBRARY_PATH=/local/software/cuda/10.0/lib64/stubs:$LIBRARY_PATH \
#&& export LIBRARY_PATH=/usr/lib64:$LIBRARY_PATH \
#&& export CPATH=/local/software/cuda/10.0/include:$CPATH


## Install CUDA
#RUN apt-get --purge remove -y nvidia*
#RUN apt-get install -y wget
#RUN apt-get install -y apt-transport-https ca-certificates
#RUN apt-get update
#
#RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-ubuntu1604.pin
#RUN mv cuda-ubuntu1604.pin /etc/apt/preferences.d/cuda-repository-pin-600
#RUN wget https://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda-repo-ubuntu1604-10-2-local-10.2.89-440.33.01_1.0-1_amd64.deb
#RUN dpkg -i cuda-repo-ubuntu1604-10-2-local-10.2.89-440.33.01_1.0-1_amd64.deb
#RUN apt-key add /var/cuda-repo-10-2-local-10.2.89-440.33.01/7fa2af80.pub
#RUN apt-get update
#RUN DEBIAN_FRONTEND=noninteractive apt-get -y install cuda-10-2
