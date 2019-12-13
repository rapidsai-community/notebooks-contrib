# BlazingSQL Demos
Demo Python notebooks using BlazingSQL with the RAPIDS AI ecoystem.

| Notebook Title | Description |Launch in Colab|
|----------------|----------------|----------------|
| Getting Started | How to set up and get started with BlazingSQL and the RAPIDS AI suite |[![Google Colab Badge](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BlazingDB/bsql-demos/blob/master/colab_notebooks/blazingsql_demo.ipynb)|
| Federated Query | In a single query, join an Apache Parquet file, a CSV file, and a GPU DataFrame (GDF) in GPU memory. |[![Google Colab Badge](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BlazingDB/bsql-demos/blob/master/colab_notebooks/federated_query_demo.ipynb)|

## Getting Started with BlazingSQL

#### Nightly (Recommended) Version

We are undergoing an architecture transition that has made the engine more stable and performant. For that reason we recommend our *Nightly* release over our stable, *Stable* will be updated with the latest cuDF v0.11 release. Find the latest install script [in our docs here](https://docs.blazingdb.com/docs/install-via-conda).

```bash
# for CUDA 9.2
conda install -c blazingsql-nightly/label/cuda9.2 -c blazingsql-nightly -c rapidsai-nightly -c conda-forge -c defaults blazingsql python=3.7

# for CUDA 10.0
conda install -c blazingsql-nightly/label/cuda10.0 -c blazingsql-nightly -c rapidsai-nightly -c conda-forge -c defaults blazingsql python=3.7
```
Note: BlazingSQL-Nightly is supported only on Linux, and with Python versions 3.6 or 3.7.
