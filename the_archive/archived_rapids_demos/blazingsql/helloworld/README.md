# BlazingSQL Demos
Demo Python notebooks using BlazingSQL with the RAPIDS AI ecoystem.

| Notebook Title | Description |Launch in Colab|
|----------------|----------------|----------------|
| Getting Started | How to set up and get started with BlazingSQL and the RAPIDS AI suite |[![Google Colab Badge](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BlazingDB/bsql-demos/blob/master/colab_notebooks/blazingsql_demo.ipynb)|
| Federated Query | In a single query, join an Apache Parquet file, a CSV file, and a GPU DataFrame (GDF) in GPU memory. |[![Google Colab Badge](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BlazingDB/bsql-demos/blob/master/colab_notebooks/federated_query_demo.ipynb)|

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