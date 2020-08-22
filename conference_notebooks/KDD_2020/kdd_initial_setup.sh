#!/bin/bash
# This script sets up the RAPIDS nightly container for the KDD 2020 hands-on tutorial

echo "****************************************************************"
echo "** Stopping Jupyter Server                                    **"
echo "****************************************************************"
kill $(ps aux | grep '[j]upyter' | awk '{print $2}')

echo ""
echo "****************************************************************"
echo "** Installing cyBERT requirements                             **"
echo "****************************************************************"
pip install torch torchvision
pip install transformers
pip install seqeval
pip install s3fs

echo ""
echo "****************************************************************"
echo "** Installing NVT requirements                                **"
echo "****************************************************************"
pip install git+https://github.com/NVIDIA/NVTabular.git
pip install tensorflow

echo ""
echo "****************************************************************"
echo "** Patching CUDA Version 10.2->10.1                           **"
echo "****************************************************************"
ln -s /usr/local/cuda/lib64/libcudart.so.10.2 /usr/local/cuda/lib64/libcudart.so.10.1

echo ""
echo "****************************************************************"
echo "** Modifying LD_LIBRARY_PATH                                  **"
echo "****************************************************************"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64/"

echo ""
echo "****************************************************************"
echo "** Reloacing Config (ldconfig)                                **"
echo "****************************************************************"
ldconfig

echo ""
echo "****************************************************************"
echo "** Starting Jupyter Server                                    **"
echo "****************************************************************"
/rapids/utils/start_jupyter.sh