# BlazingSQL Demos
Demo Python notebooks using BlazingSQL with the RAPIDS AI ecoystem.

| Notebook Title | Description |Launch in Colab|
|----------------|----------------|----------------|
| Netflow | Query 73M+ rows of network security data (netflow) with BlazingSQL and then pass to Graphistry to visualize and interact with the data. |[![Google Colab Badge](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BlazingDB/bsql-demos/blob/master/colab_notebooks/graphistry_netflow_demo.ipynb)|
| Taxi | Train a linear regression model with cuML on 20 million rows of public NYC Taxi Data loaded with BlazingSQL. |[![Google Colab Badge](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BlazingDB/bsql-demos/blob/master/colab_notebooks/taxi_fare_prediction.ipynb)|
| BlazingSQL vs. Apache Spark | Analyze over 73 million rows of net flow data to compare BlazingSQL and Apache Spark timings for the same workload. |[![Google Colab Badge](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BlazingDB/bsql-demos/blob/master/colab_notebooks/vs_pyspark_netflow.ipynb)|

## Getting Started with BlazingSQL

You can install BlazingSQL simply by running the [python script](https://github.com/rapidsai/notebooks-contrib/tree/branch-0.12/utils/sql_check.py) `sql_check.py` found in the `notebooks-contrib/utils/` directory.

#### Stable (v0.11)

You can find the latest install scripts [in our docs here](https://docs.blazingdb.com/docs/install-via-conda) or just below.

```bash
# for CUDA 9.2 & Python 3.7
conda install -c blazingsql/label/cuda9.2 -c blazingsql -c rapidsai -c nvidia -c conda-forge -c defaults blazingsql python=3.7 cudatoolkit=9.2

# for CUDA 10.0 & Python 3.7
conda install -c blazingsql/label/cuda10.0 -c blazingsql -c rapidsai -c nvidia -c conda-forge -c defaults blazingsql python=3.7 cudatoolkit=10.0
```

#### Nightly 

```bash
conda install -c blazingsql-nightly/label/cuda10.0 -c blazingsql-nightly -c rapidsai-nightly -c conda-forge -c defaults blazingsql
```

Note: BlazingSQL-Nightly is supported only on Linux, with CUDA 9.2 or 10 and Python 3.6 or 3.7.

## Troubleshooting

### On RAPIDS Docker
**Can't locate "/usr/lib/jvm" error when running `bsql_start()` or `import blazingsql`**

1. Run the comands below in teminal
```
apt-get update
apt-get install default-jre
```
2. Retry