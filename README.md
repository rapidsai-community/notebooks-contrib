# RAPIDS Extended Notebooks
## Intro
Welcome to the extended notebooks repo!

The purpose of this collection of notebooks is to help users understand what RAPIDS has to offer, learn why, how, and when including RAPIDS in a data science pipeline makes sense, and contain community contributions of RAPIDS knowledge. 

Many of these notebooks use additional PyData ecosystem packages, and include code for downloading datasets, thus they require network connectivity. If running on a system with no network access, please use the [core notebooks repo](https://github.com/rapidsai/notebooks).

## Installation

Please use the [BUILD.md](BUILD.md) to check the pre-requisite packages and installation steps.

## Contributing

Please see our [guide for contributing to notebooks-extended](CONTRIBUTING.md).

## Exploring the Repo

- `getting_started_notebooks` - “how to start using RAPIDS”.  Contains notebooks showing "hello worlds", getting started with RAPIDS libraries, and tutorials around RAPIDS concepts.   
- `intermediate` - “how to accomplish your workflows with RAPIDS”.  Contains notebooks showing algorthim and workflow examples, benchmarking tools, and some complete end-to-end (E2E) workflows.
- `advanced` - "how to master RAPIDS".  Contains notebooks showing kernal customization and advanced end-to-end workflows.
- `colab_notebooks` - contains colab versions of popular notebooks to quickly try out in browser
- `blog notebooks` - contains shared notebooks mentioned and used in blogs that showcase RAPIDS workflows and capabilities
- `conference notebooks` - contains notebooks used in conferences, such as GTC
- `competition notebooks` - contains notebooks used in competitions, such as Kaggle

`/data` contains small data samples used for purely functional demonstrations. Some notebooks include cells that download larger datasets from external websites.

The `/data` folder is also symlinked into `/rapids/notebooks/extended/data` so you can browse it from JupyterLab's UI.

# RAPIDS Notebooks-extended
## Industry Topical Notebooks
Please view our [Industry Topics README]() to see which notebooks align with which industries (coming soon!)

## Getting Started Notebooks:
| Folder    | Notebook Title         | Description                                                                                                                                                                                                                   |
|-----------|------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| basics   | [Dask_Hello_World](getting_started_notebooks/basics/Dask_Hello_World.ipynb)           | This notebook shows how to quickly setup Dask and run a "Hello World" example.                                                                                                                                       |
| basics   | [Getting_Started_with_cuDF](getting_started_notebooks/basics/Getting_Started_with_cuDF.ipynb)  | This notebook shows how to get started with GPU DataFrames using cuDF in RAPIDS.                                                                                                                                      |
| basics   | [hello_streamz](getting_started_notebooks/basics/hello_streamz.ipynb)  | This notebook demonstrates use of cuDF to perform streaming word-count using a small portion of the Streamz API.                                                                                                                                      |
| basics   | [streamz_weblogs](getting_started_notebooks/basics/streamz_weblogs.ipynb)  | This notebook provides an example of how to do streaming web-log processing with RAPIDS, Dask, and Streamz.                                                                                                                                      |
| intro_tutorials   | [01_Introduction_to_RAPIDS](getting_started_notebooks/intro_tutorials/01_Introduction_to_RAPIDS.ipynb)  | This notebook shows at a high level what each of the packages in RAPIDS are as well as what they do.                                                                                                                                      |
| intro_tutorials   | [02_Introduction_to_cuDF](getting_started_notebooks/intro_tutorials/02_Introduction_to_cuDF.ipynb)  | This notebook shows how to work with cuDF DataFrames in RAPIDS.                                                                                                                                      |
| intro_tutorials   | [03_Introduction_to_Dask](getting_started_notebooks/intro_tutorials/03_Introduction_to_Dask.ipynb)   | This notebook shows how to work with Dask using basic Python primitives like integers and strings.                                                                                                                                      |
| intro_tutorials   | [04_Introduction_to_Dask_using_cuDF_DataFrames](getting_started_notebooks/intro_tutorials/04_Introduction_to_Dask_using_cuDF_DataFrames.ipynb)   | This notebook shows how to work with cuDF DataFrames using Dask.                                                                                                                                      |
| intro_tutorials   | [05_Introduction_to_Dask_cuDF](getting_started_notebooks/intro_tutorials/05_Introduction_to_Dask_cuDF.ipynb)   | This notebook shows how to work with cuDF DataFrames distributed across multiple GPUs using Dask.                                                                                                                                      |
| intro_tutorials   | [06_Introduction_to_Supervised_Learning](getting_started_notebooks/intro_tutorials/06_Introduction_to_Supervised_Learning.ipynb)   | This notebook shows how to do GPU accelerated Supervised Learning in RAPIDS.                                                                                                                                      |
| intro_tutorials   | [07_Introduction_to_XGBoost](getting_started_notebooks/intro_tutorials/07_Introduction_to_XGBoost.ipynb)   | This notebook shows how to work with GPU accelerated XGBoost in RAPIDS.                                                                                                                                      |
| intro_tutorials   | [08_Introduction_to_Dask_XGBoost](getting_started_notebooks/intro_tutorials/08_Introduction_to_Dask_XGBoost.ipynb)   | This notebook shows how to work with Dask XGBoost in RAPIDS.                                                                                                                                      |
| intro_tutorials   | [09_Introduction_to_Dimensionality_Reduction](getting_started_notebooks/intro_tutorials/09_Introduction_to_Dimensionality_Reduction.ipynb)   | This notebook shows how to do GPU accelerated Dimensionality Reduction in RAPIDS.                                                                                                                                      |
| intro_tutorials   | [10_Introduction_to_Clustering](getting_started_notebooks/intro_tutorials/10_Introduction_to_Clustering.ipynb)  | This notebook shows how to do GPU accelerated Clustering in RAPIDS.                                                                                                                                      |

## Intermediate Notebooks:
| Folder    | Notebook Title         | Description                                                                                                                                                                                                                   |
|-----------|------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| examples   | [DBSCAN_Demo_FULL](intermediate_notebooks/examples/DBSCAN_Demo_FULL.ipynb)               |  This notebook shows how to use DBSCAN algorithm and its GPU accelerated implementation present in RAPIDS.                                                  |
| examples   | [Dask_with_cuDF_and_XGBoost](intermediate_notebooks/examples/Dask_with_cuDF_and_XGBoost.ipynb)                    | In this notebook we show how to quickly setup Dask and train an XGBoost model using cuDF.                                                                                                |
| examples   | [Dask_with_cuDF_and_XGBoost_Disk](intermediate_notebooks/examples/Dask_with_cuDF_and_XGBoost_Disk.ipynb)                   | In this notebook we show how to quickly setup Dask and train an XGBoost model using cuDF and read the data from disk using cuIO.                                                                                                      |
| examples   | [One_Hot_Encoding](intermediate_notebooks/examples/One_Hot_Encoding.ipynb)    | In this notebook we show how to use dask and cudf to use xgboost on a dataset.                                              |
| examples   | [PCA_Demo_Full](intermediate_notebooks/examples/PCA_Demo_Full.ipynb)               | In this notebook we will show how to use PCA and its GPU accelerated implementation present in RAPIDS.                                   |
| examples   | [linear_regression_demo.ipynb](intermediate_notebooks/examples/linear_regression_demo.ipynb)     |In this notebook we will show how to use linear regression and its GPU accelerated implementation present in RAPIDS.                                                                                                                |
| examples   | [ridge_regression_demo](intermediate_notebooks/examples/ridge_regression_demo.ipynb)     | Demonstration of using both NetworkX and cuGraph  to compute the the number of Triangles in our test dataset.                                                                                                                  |
| examples   | [umap_demo](intermediate_notebooks/examples/umap_demo.ipynb)     | In this notebook we will show how to use UMAP and its GPU accelerated implementation present in RAPIDS.   |
| examples   | [rf_demo](intermediate_notebooks/examples/rf_demo.ipynb)     | Demonstration of using both cuml and sklearn to train a RandomForestClassifier on the Higgs dataset.   |
| E2E-> mortgage      | [mortgage_e2e](intermediate_notebooks/E2E/mortgage/mortgage_e2e.ipynb)            | This is an end to end notebook consisting of `ETL`, `data conversion` and `machine learning for training` operations performed on the mortgage dataset.      |
| E2E-> mortgage      | [mortgage_e2e_deep_learning](intermediate_notebooks/E2E/mortgage/mortgage_e2e_deep_learning.ipynb)     | This notebook combines the RAPIDS GPU data processing with a PyTorch deep learning neural network to predict mortgage loan delinquency.                                                                                                                          |
| E2E-> taxi      | [NYCTaxi](intermediate_notebooks/E2E/taxi/NYCTaxi.ipynb)     | Demonstrates multi-node ETL for cleanup of raw data into cleaned train and test dataframes. Shows how to run multi-node XGBoost training with dask-xgboost |
| E2E-> synthetic_3D      | [rapids_ml_workflow_demo](intermediate_notebooks/E2E/synthetic_3D/rapids_ml_workflow_demo.ipynb) | A 3D visual showcase of a machine learning workflow with RAPIDS (load data, transform/normalize, train XGBoost model, evaluate accuracy, use model for inference). Along the way we compare the performance gains of RAPIDS [GPU] vs sklearn/pandas methods [CPU].   |
| E2E-> census      | [census_education2income_demo](intermediate_notebooks/E2E/census/census_education2income_demo.ipynb)     | In this notebook we use 50 years of census data to see how education affects income.  |
| E2E-> gdelt    | [Ridge_regression_with_feature_encoding](intermediate_notebooks/E2E/gdelt/Ridge_regression_with_feature_encoding.ipynb)    | An end to end example using ridge regression on the gdelt dataset. Includes ETL with `cuDF`, feature scaling/encoding, and model training and evaluation with `cuML` |
| benchmarks      | [cuml_benchmarks](intermediate_notebooks/benchmarks/cuml_benchmarks.ipynb)  | The purpose of this notebook is to benchmark all of the single GPU cuML algorithms against their skLearn counterparts, while also providing the ability to find and verify upper bounds.                                                                                                                                          |
| benchmarks-> cugraph_benchmarks      | [louvain_benchmark](intermediate_notebooks/benchmarks/cugraph_benchmarks/louvain_benchmark.ipynb)   | This notebook benchmarks performance improvement of running the Louvain clustering algorithm within cuGraph against NetworkX.                                                                                                                                                                |
|  benchmarks-> cugraph_benchmarks    |  [pagerank_benchmark](intermediate_notebooks/benchmarks/cugraph_benchmarks/pagerank_benchmark.ipynb)             | This notebook benchmarks performance improvement of running PageRank within cuGraph against NetworkX.

## Advanced Notebooks:
| Folder    | Notebook Title         | Description                                                                                                                                                                                                                   |
|-----------|------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| tutorials   | [rapids_customized_kernels](advanced_notebooks/tutorials/rapids_customized_kernels.ipynb)               |  This notebook shows how create customized kernels using CUDA to make your workflow in RAPIDS even faster.    

## Blog Notebooks:
| Folder    | Notebook Title         | Description                                                                                                                                                                                                                   |
|-----------|------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| cyber -> flow_classification     | [flow_classification_rapids](blog_notebooks/cyber/flow_classification/flow_classification_rapids.ipynb)              | This notebook demonstrates how to load netflow data into cuDF and create a multiclass classification model using XGBoost.                                                                                                     |
| cyber ->network_mapping      | [lanl_network_mapping_using_rapids](blog_notebooks/cyber/network_mapping/lanl_network_mapping_using_rapids.ipynb)               | This notebook demonstrates how to parse raw windows event logs using cudf and uses cuGraph's pagerank model to build a network graph.                                                                         |
| cyber ->raw_data_generator      | [run_raw_data_generator](blog_notebooks/cyber/raw_data_generator/run_raw_data_generator.py)              | The notebook is used showcase how to generate raw logs from the parsed LANL 2017 json data. The intent is to use the raw data to demonstrate parsing capabilities using cuDF.                       |
| databricks   | [RAPIDS_PCA_demo_avro_read](blog_notebooks/databricks/RAPIDS_PCA_demo_avro_read.ipynb)              | This notebooks purpose is to showcase RAPIDS on Databricks use thier sample datasets and show the CPU vs GPU comparison for the PCA algorithm. There is also an accompanying HTML file for easy Databricks import.                                                                       
| regression   | [regression_blog_notebook](blog_notebooks/regression/regression_blog_notebook.ipynb)       | This notebook showcases an end to end notebook using the try_this dataset and cuML's implementation of ridge regression.

## Conference Notebooks:
| Folder    | Notebook Title         | Description                                                                                                                                                                                                                   |
|-----------|------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| GTC_SJ_2019   | [GTC_tutorial_instructor](conference_notebooks/GTC_SJ_2019/GTC_tutorial_instructor.ipynb)               |  Description comming soon!   |
| GTC_SJ_2019   | [GTC_tutorial_student](conference_notebooks/GTC_SJ_2019/GTC_tutorial_student.ipynb)               |  Description comming soon!   |

## Competition Notebooks:
| Folder    | Notebook Title         | Description                                                                                                                                                                                                                   |
|-----------|------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| kaggle-> landmark   | [cudf_stratifiedKfold_1000x_speedup](competition_notebooks/kaggle/landmark/cudf_stratifiedKfold_1000x_speedup.ipynb)               |  Description comming soon!    |
| kaggle-> malware   | [malware_time_column_explore](competition_notebooks/kaggle/malware/malware_time_column_explore.ipynb)               |  Description comming soon!   |
| kaggle-> malware   | [rapids_solution_gpu_only](competition_notebooks/kaggle/malware/rapids_solution_gpu_only.ipynb)               |  Description comming soon!   |
| kaggle-> malware   | [rapids_solution_gpu_vs_cpu](competition_notebooks/kaggle/malware/rapids_solution_gpu_vs_cpu.ipynb)               |  Description comming soon!   |
| kaggle-> plasticc-> notebooks   | [rapids_lsst_full_demo](competition_notebooks/kaggle/plasticc/notebooks/rapids_lsst_full_demo)               |  Description comming soon!   |
| kaggle-> plasticc-> notebooks   | [rapids_lsst_gpu_only_demo](competition_notebooks/kaggle/plasticc/notebooks/rapids_lsst_gpu_only_demo.ipynb)               |  Description comming soon!   |
| kaggle-> santander   | [cudf_tf_demo](competition_notebooks/kaggle/santander/cudf_tf_demo.ipynb)               |  Description comming soon!   |
| kaggle-> santander   | [E2E_santander_pandas](competition_notebooks/kaggle/santander/E2E_santander_pandas.ipynb)               |  Description comming soon!   |
| kaggle-> santander   | [E2E_santander](competition_notebooks/kaggle/santander/E2E_santander.ipynb)               |  Description comming soon!   

## Additional Information
* The `cuml` folder also includes a small subset of the Mortgage Dataset used in the notebooks and the full image set from the Fashion MNIST dataset.

* `utils`: contains a set of useful scripts for interacting with RAPIDS

* For additional, community driven notebooks, which will include our blogs, tutorials, workflows, and more intricate examples, please see the [Notebooks Extended Repo](https://github.com/rapidsai/notebooks-extended)

