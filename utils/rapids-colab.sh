#!/bin/bash

set -eu


RAPIDS_VERSION="${1:-0.11}"
echo "PLEASE READ"
echo "********************************************************************************************************"
echo "Colab v0.11 Migration Bulletin:"
echo " "
echo "There has been a NECESSARY Colab script code change that MAY REQUIRE an update how we install RAPIDS into Colab!  "
echo "Not all Colab notebooks are updated (like personal Colabs) and while the script will install RAPIDS correctly, "
echo "a neccessary script to update pyarrow to v0.15.x to be compatible with RAPIDS v0.11+ may not run, and your RAPIDS instance"
echo "will BREAK"
echo " "
echo "The code in that update is below.  If your code does not look like the snippet below, "
echo "Please:"
echo "1. STOP cell execution" 
echo "2. CUT and PASTE the script below into the cell you just ran "
echo "3. Reset All Runtimes, get a compatible GPU, and rerun your Colab Notebook."
echo " "
echo "SCRIPT TO COPY:"
echo "!wget -nc https://raw.githubusercontent.com/rapidsai/notebooks-contrib/master/utils/rapids-colab.sh"
echo "!bash rapids-colab.sh"
echo "import sys, os"
echo "dist_package_index = sys.path.index("/usr/local/lib/python3.6/dist-packages")"
echo "sys.path = sys.path[:dist_package_index] + ["/usr/local/lib/python3.6/site-packages"] + sys.path[dist_package_index:]"
echo "sys.path"
echo "if os.path.exists('update_pyarrow.py'): ## Only exists if RAPIDS version is 0.11 or higher"
echo "  exec(open("update_pyarrow.py").read(), globals())"
echo "********************************************************************************************************"
echo " "

wget -nc https://github.com/rapidsai/notebooks-contrib/raw/master/utils/env-check.py
echo "Checking for GPU type:"
python env-check.py

if [ ! -f Miniconda3-4.5.4-Linux-x86_64.sh ]; then
    echo "Removing conflicting packages, will replace with RAPIDS compatible versions"
    # remove existing xgboost and dask installs
    pip uninstall -y xgboost dask distributed

    # intall miniconda
    echo "Installing conda"
    wget https://repo.continuum.io/miniconda/Miniconda3-4.5.4-Linux-x86_64.sh
    chmod +x Miniconda3-4.5.4-Linux-x86_64.sh
    bash ./Miniconda3-4.5.4-Linux-x86_64.sh -b -f -p /usr/local
    
    if [ $RAPIDS_VERSION == "0.11" ] ;then
    echo "Installing RAPIDS $RAPIDS_VERSION packages from the nightly release channel"
    echo "Please standby, this will take a few minutes..."
    # install RAPIDS packages
        conda install -y --prefix /usr/local \
                -c rapidsai-nightly/label/xgboost -c rapidsai-nightly -c nvidia -c conda-forge \
                python=3.6 cudatoolkit=10.1 \
                cudf=$RAPIDS_VERSION cuml cugraph gcsfs pynvml cuspatial \
                dask-cudf \
                xgboost
        # check to make sure that pyarrow is running the right version (0.15) for v0.11 or later
        wget -nc https://github.com/rapidsai/notebooks-contrib/raw/master/utils/update_pyarrow.py

    else
        echo "Installing RAPIDS $RAPIDS_VERSION packages from the stable release channel"
        echo "Please standby, this will take a few minutes..."
        # install RAPIDS packages
        conda install -y --prefix /usr/local \
            -c rapidsai/label/xgboost -c rapidsai -c nvidia -c conda-forge \
            python=3.6 cudatoolkit=10.1 \
            cudf=$RAPIDS_VERSION cuml cugraph cuspatial gcsfs pynvml \

            dask-cudf \
            xgboost
    fi
      
    echo "Copying shared object files to /usr/lib"
    # copy .so files to /usr/lib, where Colab's Python looks for libs
    cp /usr/local/lib/libcudf.so /usr/lib/libcudf.so
    cp /usr/local/lib/librmm.so /usr/lib/librmm.so
    cp /usr/local/lib/libnccl.so /usr/lib/libnccl.so
fi

echo ""
echo "*********************************************"
echo "Your Google Colab instance is RAPIDS ready!"
echo "*********************************************"
echo "IF YOUR RAPIDS INSTALL DOESN'T WORK, please read the Migration Notice below.  You may have missed it when the script first ran! "
echo "PLEASE READ"
echo "********************************************************************************************************"
echo "Colab v0.11 Migration Bulletin:"
echo " "
echo "There has been a NECESSARY Colab script code change that MAY REQUIRE an update how we install RAPIDS into Colab!  "
echo "Not all Colab notebooks are updated (like personal Colabs) and while the script will install RAPIDS correctly, "
echo "a neccessary script to update pyarrow to v0.15.x to be compatible with RAPIDS v0.11+ may not run, and your RAPIDS instance"
echo "will BREAK"
echo " "
echo "The code in that update is below.  If your code does not look like the snippet below, "
echo "Please:"
echo "1. STOP cell execution" 
echo "2. CUT and PASTE this script into the cell you just ran "
echo "3. Reset All Runtimes, get a compatible GPU, and rerun your Colab Notebook."
echo " "
echo "SCRIPT:"
echo "!wget -nc https://raw.githubusercontent.com/rapidsai/notebooks-contrib/master/utils/rapids-colab.sh"
echo "!bash rapids-colab.sh"
echo "import sys, os"
echo "dist_package_index = sys.path.index("/usr/local/lib/python3.6/dist-packages")"
echo "sys.path = sys.path[:dist_package_index] + ["/usr/local/lib/python3.6/site-packages"] + sys.path[dist_package_index:]"
echo "sys.path"
echo "if os.path.exists('update_pyarrow.py'): ## Only exists if RAPIDS version is 0.11 or higher"
echo "  exec(open("update_pyarrow.py").read(), globals())"
echo "********************************************************************************************************"
echo " "
