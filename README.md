# RAPIDS Extended Notebooks
## Intro
Welcome to the extended notebooks repo!

The purpose of this collection of notebooks is to help users understand what RAPIDS has to offer, learn why, how, and when including RAPIDS in a data science pipeline makes sense, and contain community contributions of RAPIDS knowledge. 

Many of these notebooks use additional PyData ecosystem packages, and include code for downloading datasets, thus they require network connectivity. If running on a system with no network access, please use the [core notebooks repo](https://github.com/rapidsai/notebooks).

## Installation

Please use the [BUILD.md](https://github.com/rapidsai/notebooks-extended/tree/master) to check the pre-requisite packages and installation steps.

## Contributing

Please see our [guide for contributing to notebooks-extended](https://github.com/rapidsai/notebooks-extended/blob/master/CONTRIBUTING.md).

## Exploring the Repo

Notebooks live under two subfolders:
- `beginner` - contains notebooks showing “how [to master] RAPIDS”:
    - `basics` - to help you quickly learn the basic RAPIDS APIs.  It contains these notebooks
        - Dask Hello World
        - Getting Started with cuDF
    - `tutorial` - which is a basic tutorial on all the libraries present in RAPIDS.
    
- `advanced` - these notebooks demonstrate "why RAPIDS" by directly comparing compute time between single and multi threaded CPU implementations vs GPU (RAPIDS library) implementations. Of note here is the similarity of RAPIDS APIs to common PyData ecosystem packages like Pandas and scikit-learn. This folder includes the following folders: 
    - E2E: The E2E folder contains end to end notebooks which use the RAPIDS libraries
    - benchmarks: The benchmarks folder contains notebooks which are used to benchmark the cuGraph and cuML algorithms
    - blog notebooks: contains the notebooks mentioned and used in blogs written by RAPIDS
    - conference notebooks
    - examples: contains high level examples for users familiar with RAPIDS libraries

`/data` contains small data samples used for purely functional demonstrations. Some notebooks include cells that download larger datasets from external websites.

The `/data` folder is also symlinked into `/rapids/notebooks/extended/data` so you can browse it from JupyterLab's UI.

# RAPIDS Notebooks-extended

## Beginner Notebooks:
| Folder    | Notebook Title         | Description                                                                                                                                                                                                                   |
|-----------|------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| basics   | Dask_Hello_World           | This notebook shows how to quickly setup Dask and run a "Hello World" example.                                                                                                                                       |
| basics   | Getting_Started_with_cuDF  | This notebook shows how to get started with GPU DataFrames using cuDF in RAPIDS.                                                                                                                                      |
| tutorial   | 01_Introduction_to_RAPIDS  | This notebook shows at a high level what each of the packages in RAPIDS are as well as what they do.                                                                                                                                      |
| tutorial   | 02_Introduction_to_cuDF  | This notebook shows how to work with cuDF DataFrames in RAPIDS.                                                                                                                                      |
| tutorial   | 03_Introduction_to_Dask  | This notebook shows how to work with Dask using basic Python primitives like integers and strings.                                                                                                                                      |
| tutorial   | 04_Introduction_to_Dask_using_cuDF_DataFrames  | This notebook shows how to work with cuDF DataFrames using Dask.                                                                                                                                      |
| tutorial   | 05_Introduction_to_Dask_cuDF  | This notebook shows how to work with cuDF DataFrames distributed across multiple GPUs using Dask.                                                                                                                                      |
| tutorial   | 06_Introduction_to_Supervised_Learning  | This notebook shows how to do GPU accelerated Supervised Learning in RAPIDS.                                                                                                                                      |
| tutorial   | 07_Introduction_to_XGBoost  | This notebook shows how to work with GPU accelerated XGBoost in RAPIDS.                                                                                                                                      |
| tutorial   | 08_Introduction_to_Dask_XGBoost  | This notebook shows how to work with Dask XGBoost in RAPIDS.                                                                                                                                      |
| tutorial   | 09_Introduction_to_Dimensionality_Reduction  | This notebook shows how to do GPU accelerated Dimensionality Reduction in RAPIDS.                                                                                                                                      |
| tutorial   | 10_Introduction_to_Clustering  | This notebook shows how to do GPU accelerated Clustering in RAPIDS.                                                                                                                                      |

## Advanced Notebooks:
 
| Folder    | Notebook Title         | Description                                                                                                                                                                                                                   |
|-----------|------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| E2E-> mortgage      | mortgage_e2e            | This is an end to end notebook consisting of `ETL`, `data conversion` and `machine learning for training` operations performed on the mortgage dataset.                                                                               |
| E2E-> mortgage      | mortgage_e2e_deep_learning     | This notebook combines the RAPIDS GPU data processing with a PyTorch deep learning neural network to predict mortgage loan delinquency.                                                                                                                          |
| E2E-> taxi      | NYCTaxi     | Demonstrates multi-node ETL for cleanup of raw data into cleaned train and test dataframes. Shows how to run multi-node XGBoost training with dask-xgboost |
| E2E-> synthetic_3D      | rapids_ml_workflow_demo | A 3D visual showcase of a machine learning workflow with RAPIDS (load data, transform/normalize, train XGBoost model, evaluate accuracy, use model for inference). Along the way we compare the performance gains of RAPIDS [GPU] vs sklearn/pandas methods [CPU].                                                                                                                                             |
| benchmarks      | cuml_benchmarks  | The purpose of this notebook is to benchmark all of the single GPU cuML algorithms against their skLearn counterparts, while also providing the ability to find and verify upper bounds.                                                                                                                                          |
| benchmarks-> cugraph_benchmarks      | louvain_benchmark.ipynb   | This notebook benchmarks performance improvement of running the Louvain clustering algorithm within cuGraph against NetworkX.                                                                                                                                                                |
|  benchmarks-> cugraph_benchmarks    |  pagerank_benchmark.ipynb             | This notebook benchmarks performance improvement of running PageRank within cuGraph against NetworkX.                                                                                                                |

### Blog notebooks:
#### Cyber notebooks:
| Folder    | Notebook Title         | Description                                                                                                                                                                                                                   |
|-----------|------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|flow_classification     | flow_classification_rapids              | This notebook demonstrates how to load netflow data into cuDF and create a multiclass classification model using XGBoost.                                                                                                     |
| network_mapping      | lanl_network_mapping_using_rapids               | This notebook demonstrates how to parse raw windows event logs using cudf and uses cuGraph's pagerank model to build a network graph.                                                                         |
| raw_data_generator      | run_raw_data_generator              | The notebook is used showcase how to generate raw logs from the parsed LANL 2017 json data. The intent is to use the raw data to demonstrate parsing capabilities using cuDF.                       |

#### Databrix Notebooks:
| Folder    | Notebook Title         | Description                                                                                                                                                                                                                   |
|-----------|------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| databrix   | RAPIDS_PCA_demo_avro_read                | This notebooks purpose is to showcase RAPIDS on Databrix use thier sample datasets and show the CPU vs GPU comparison for the PCA algorithm. There is also an accompanying HTML file for easy Databricks import.                                                                                                                         |                                                                       

#### Regression Notebooks:
| Folder    | Notebook Title         | Description                                                                                                                                                                                                                   |
|-----------|------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| regression   | regression_blog_notebook.ipynb       | This notebook showcases an end to end notebook using the try_this dataset and cuML's implementation of ridge regression.                                                                                                                     |

### Example Notebooks:
| Folder    | Notebook Title         | Description                                                                                                                                                                                                                   |
|-----------|------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| examples   | DBSCAN_Demo_FULL               |  This notebook shows how to use DBSCAN algorithm and its GPU accelerated implementation present in RAPIDS.                                                  |
| examples   | Dask_with_cuDF_and_XGBoost                    | In this notebook we show how to quickly setup Dask and train an XGBoost model using cuDF.                                                                                                |
| examples   | Dask_with_cuDF_and_XGBoost_Disk                   | In this notebook we show how to quickly setup Dask and train an XGBoost model using cuDF and read the data from disk using cuIO.                                                                                                      |
| examples   | One_Hot_Encoding    | In this notebook we show how to use dask and cudf to use xgboost on a dataset.                                              |
| examples   | PCA_Demo_Full               | In this notebook we will show how to use PCA and its GPU accelerated implementation present in RAPIDS.                                   |
| examples   | linear_regression_demo.ipynb     |In this notebook we will show how to use linear regression and its GPU accelerated implementation present in RAPIDS.                                                                                                                |
| examples   | ridge_regression_demo.ipynb     | Demonstration of using both NetworkX and cuGraph  to compute the the number of Triangles in our test dataset.                                                                                                                  |
| examples   | umap_demo.ipynb     | In this notebook we will show how to use UMAP and its GPU accelerated implementation present in RAPIDS.                                                                                             

## Additional Information
* The `cuml` folder also includes a small subset of the Mortgage Dataset used in the notebooks and the full image set from the Fashion MNIST dataset.

* `utils`: contains a set of useful scripts for interacting with RAPIDS

* For additional, community driven notebooks, which will include our blogs, tutorials, workflows, and more intricate examples, please see the [Notebooks Extended Repo](https://github.com/rapidsai/notebooks-extended)

