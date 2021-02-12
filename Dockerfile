# Use an official ubuntu image
FROM ubuntu:16.04

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
RUN python -m pip install torch==1.7.1+cu92 torchvision==0.8.2+cu92 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
#RUN python -m pip install torch torchvision

#RUN python -m pip install numpy==1.19.5
RUN python -m pip install -r requirements.txt

# DO whacky thing to fix ray
#RUN pip3 -y uninstall tree
#RUN pip3 -y uninstall dm-tree
#RUN pip3 install --upgrade ray
#RUN pip3 install dm-tree

RUN apt -y install libgl1-mesa-glx
RUN apt-get -y install nvidia-modprobe

