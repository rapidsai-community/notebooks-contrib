# BlazingSQL Demos
Demo Python notebooks using BlazingSQL with the RAPIDS AI ecoystem.

| Notebook Title | Description |Launch in Colab|
|----------------|----------------|----------------|
| Netflow | Query 73M+ rows of network security data (netflow) with BlazingSQL and then pass to Graphistry to visualize and interact with the data. |[![Google Colab Badge](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BlazingDB/bsql-demos/blob/master/colab_notebooks/graphistry_netflow_demo.ipynb)|
| Taxi | Train a linear regression model with cuML on 20 million rows of public NYC Taxi Data loaded with BlazingSQL. |[![Google Colab Badge](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BlazingDB/bsql-demos/blob/master/colab_notebooks/taxi_fare_prediction.ipynb)|
| BlazingSQL vs. Apache Spark | Analyze over 73 million rows of net flow data to compare BlazingSQL and Apache Spark timings for the same workload. |[![Google Colab Badge](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BlazingDB/bsql-demos/blob/master/colab_notebooks/vs_pyspark_netflow.ipynb)|

## Getting Started with BlazingSQL

You can install BlazingSQL simply by running the [shell script](https://github.com/rapidsai/notebooks-contrib/tree/branch-0.12/utils) `now_were_blazing.sh` found in the `notebooks-contrib/utils/` directory.

#### Nightly (Recommended) Version

We are undergoing an architecture transition that has made the engine more stable and performant. For that reason we recommend our *Nightly* release over our stable, *Stable* will be updated with the latest cuDF v0.11 release. Find the latest install script [in our docs here](https://docs.blazingdb.com/docs/install-via-conda).

```bash
# for CUDA 9.2
conda install -c blazingsql-nightly/label/cuda9.2 -c blazingsql-nightly -c rapidsai-nightly -c conda-forge -c defaults blazingsql python=3.7

# for CUDA 10.0
conda install -c blazingsql-nightly/label/cuda10.0 -c blazingsql-nightly -c rapidsai-nightly -c conda-forge -c defaults blazingsql python=3.7
```
Note: BlazingSQL-Nightly is supported only on Linux, and with Python versions 3.6 or 3.7.
