ARG CUDA_VERSION="10.0"
ARG DISTRO="ubuntu18.04"
ARG IMAGE_TYPE=base

#-v /raid/work/rapids:/work
#-v /raid/datasets:/datasets:ro

FROM nvidia/cuda:${CUDA_VERSION}-${IMAGE_TYPE}-${DISTRO}
#FROM rapidsai/rapidsai-nightly:cuda${CUDA_VERSION}-${IMAGE_TYPE}-${DISTRO}

ARG GROUP_ID=10000
ARG USER_ID=10000
ARG CONDA_USER

#Setup Conda
ADD https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh miniconda.sh
RUN bash miniconda.sh -f -b -p /conda
ENV PATH /conda/bin:$PATH
RUN conda init bash

ENV CUDA_VERSION=${CUDA_VERSION}
ENV NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES}
ENV hostname=${hostname}_cuda${CUDA_VERSION}-${DISTRO}

ENV DEBIAN_FRONTEND=noninteractive

RUN groupadd --gid ${GROUP_ID} conda && \
    useradd -g ${GROUP_ID} -u ${USER_ID} -ms /bin/bash ${CONDA_USER} && \
    cat /conda/etc/profile.d/conda.sh >> /home/${CONDA_USER}/.bashrc && \
    chgrp -R conda /conda && \
    chmod -R g+rwx /conda

RUN echo "conda activate benchmark" >> /home/${CONDA_USER}/.bashrc
	
USER ${CONDA_USER}

ADD environment.yml environment.yml
RUN /bin/bash -c "source /conda/etc/profile.d/conda.sh && \
                  conda env create -f environment.yml -n benchmark && \
                  conda activate benchmark && \
                  pip install 'git+https://gitlab-master.nvidia.com/kgerman/asv@resultsfile' && \
                  pip install 'git+https://gitlab-master.nvidia.com/kgerman/condarepo-asv-plugin@master'"