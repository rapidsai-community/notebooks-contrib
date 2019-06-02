#!/bin/bash

set -e

wget https://github.com/randerzander/notebooks-extended/raw/master/utils/env-check.py

python env-check.py

if [ ! -f Miniconda3-4.5.4-Linux-x86_64.sh ]; then
    echo "Removing existing dask & xgboost packages"
    # remove existing xgboost and dask installs
    pip uninstall -y xgboost dask distributed dask-xgboost

    # intall miniconda
    echo "Installing conda"
    wget https://repo.continuum.io/miniconda/Miniconda3-4.5.4-Linux-x86_64.sh
    chmod +x Miniconda3-4.5.4-Linux-x86_64.sh
    bash ./Miniconda3-4.5.4-Linux-x86_64.sh -b -f -p /usr/local

    echo "Installing RAPIDS packages"
    # install RAPIDS packages
    conda install -y --prefix /usr/local \
      -c rapidsai-nightly/label/xgboost -c rapidsai-nightly -c nvidia -c conda-forge \
      python=3.6 cudatoolkit=10.0 \
      cudf dask-cudf \
      cuml dask-cuml \
      cugraph \
      xgboost=>0.83 dask-xgboost=>0.2 \
      gcsfs
    
    echo "Copying shared object files to /usr/lib"
    # copy .so files to /usr/lib, where Colab's Python looks for libs
    cp /usr/local/lib/libcudf.so /usr/lib/libcudf.so
    cp /usr/local/lib/librmm.so /usr/lib/librmm.so
    cp /usr/local/lib/libxgboost.so /usr/lib/libxgboost.so
    cp /usr/local/lib/libnccl.so /usr/lib/libnccl.so

    echo "Installing ngrok"
    # expose Dask's status dashboard to a public URL
    wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip
    unzip ngrok-stable-linux-amd64.zip
    mv ngrok /usr/local/bin/ngrok
fi

export PATH=$PATH:/usr/local/lib/python3.6/site-packages
export NUMBAPRO_NVVM=/usr/local/cuda/nvvm/lib64/libnvvm.so
export NUMBAPRO_LIBDEVICE=/usr/local/cuda/nvvm/libdevice/
