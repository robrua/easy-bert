FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04 as build


MAINTAINER Rob Rua <robertrua@gmail.com>


ENV DEBIAN_FRONTEND noninteractive
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8


# Install utilities
RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates curl git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


# Anaconda home setup
ENV CONDA_HOME /opt/conda
ENV PATH $CONDA_HOME/bin:$PATH


# Python version
ARG python_version=3.6


# Install Miniconda python
RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/conda.sh && \
    /bin/bash ~/conda.sh -b -p $CONDA_HOME && \
    rm ~/conda.sh && \
    conda install -y python=${python_version} && \
    $CONDA_HOME/bin/conda clean -tipsy


# Install easy-bert
ADD easybert /opt/easy-bert/easybert
ADD MANIFEST.in /opt/easy-bert/MANIFEST.in
ADD requirements-gpu.txt /opt/easy-bert/requirements.txt
ADD setup.py /opt/easy-bert/setup.py

WORKDIR /opt/easy-bert
RUN python setup.py install


# Use multi-stage build to minimize image size
FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04


MAINTAINER Rob Rua <robertrua@gmail.com>


COPY --from=build $CONDA_HOME $CONDA_HOME


ENTRYPOINT ["bert"]
