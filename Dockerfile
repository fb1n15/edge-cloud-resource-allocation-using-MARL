FROM ubuntu:16.04
RUN apt-get update && apt-get -y install apt-utils
RUN apt -y install wget bzip2

RUN wget http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -p /miniconda -b
RUN rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH=/miniconda/bin:${PATH}
RUN conda update -y conda
RUN conda install -y pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch