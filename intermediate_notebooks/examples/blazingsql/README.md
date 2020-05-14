# BlazingSQL Demos
Demo Python notebooks using BlazingSQL with the RAPIDS AI ecoystem.

| Notebook Title | Description |Launch in Colab|
|----------------|----------------|----------------|
| Taxi | Train a linear regression model with cuML on 20 million rows of public NYC Taxi Data loaded with BlazingSQL. |[![Google Colab Badge](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BlazingDB/bsql-demos/blob/master/colab_notebooks/taxi_fare_prediction.ipynb)|
| BlazingSQL vs. Apache Spark | Analyze over 73 million rows of net flow data to compare BlazingSQL and Apache Spark timings for the same workload. |[![Google Colab Badge](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BlazingDB/bsql-demos/blob/master/colab_notebooks/vs_pyspark_netflow.ipynb)|

### BlazingSQL Notebooks
Run demos free in BlazingSQL Notebooks, a web-based Jupyter Notebook that lets you quickly run BlazingSQL + RAPIDS AI. We will walk you through each demo, but feel free to modify each demo for your own needs. 

| Notebook Title | Description | Try Now |
| -------------- | ----------- | ------- |
| Welcome Notebook | An introduction to BlazingSQL Notebooks and the GPU Data Science Ecosystem. | <a href='https://app.blazingsql.com/jupyter/user-redirect/lab/workspaces/auto-b/tree/Welcome_to_BlazingSQL_Notebooks/welcome.ipynb'><img src="https://blazingsql.com/launch-notebooks.png" alt="Launch on BlazingSQL Notebooks" width="500"/></a> |
| The DataFrame | Learn how to use BlazingSQL and cuDF to create GPU DataFrames with SQL and Pandas-like APIs. | <a href='https://app.blazingsql.com/jupyter/user-redirect/lab/workspaces/auto-b/tree/Welcome_to_BlazingSQL_Notebooks/intro_notebooks/the_dataframe.ipynb'><img src="https://blazingsql.com/launch-notebooks.png" alt="Launch on BlazingSQL Notebooks" width="500"/></a> |
| Data Visualization | Plug in your favorite Python visualization packages, or use GPU accelerated visualization tools to render millions of rows in a flash. | <a href='https://app.blazingsql.com/jupyter/user-redirect/lab/workspaces/auto-b/tree/Welcome_to_BlazingSQL_Notebooks/intro_notebooks/data_visualization.ipynb'><img src="https://blazingsql.com/launch-notebooks.png" alt="Launch on BlazingSQL Notebooks" width="500"/></a> |
| Machine Learning | Learn about cuML, mirrored after the Scikit-Learn API, it offers GPU accelerated machine learning on GPU DataFrames. | <a href='https://app.blazingsql.com/jupyter/user-redirect/lab/workspaces/auto-b/tree/Welcome_to_BlazingSQL_Notebooks/intro_notebooks/machine_learning.ipynb'><img src="https://blazingsql.com/launch-notebooks.png" alt="Launch on BlazingSQL Notebooks" width="500"/></a> |

## Getting Started with BlazingSQL

You can install BlazingSQL simply by running the [python script](https://github.com/rapidsai/notebooks-contrib/tree/branch-0.12/utils/sql_check.py) `sql_check.py` found in the `notebooks-contrib/utils/` directory.

#### Stable (v0.13)

You can find the latest install scripts [in our docs here](https://docs.blazingdb.com/docs/install-via-conda), [in our main GitHub repo](https://github.com/blazingdb/blazingsql#install-using-conda) or just below.

```bash
# for CUDA 10.0 & Python 3.6
conda install -c blazingsql/label/cuda10.0 -c blazingsql -c rapidsai -c nvidia -c conda-forge -c defaults blazingsql python=3.6

# for CUDA 10.2 & Python 3.7
conda install -c blazingsql/label/cuda10.2 -c blazingsql -c rapidsai -c nvidia -c conda-forge -c defaults blazingsql python=3.7
```

#### Nightly 

```bash
conda install -c blazingsql-nightly/label/cuda10.0 -c blazingsql-nightly -c rapidsai-nightly -c nvidia -c conda-forge -c defaults blazingsql python=3.7
```

## Troubleshooting

### On RAPIDS Docker
**Can't locate "/usr/lib/jvm" error when running `bsql_start()` or `import blazingsql`**

1. Run the comands below in teminal
```
apt-get update
apt-get install default-jre
```
2. Retry