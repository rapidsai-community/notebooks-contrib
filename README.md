# RAPIDS Notebooks-Contrib
---
## Table of Contents
* [Intro](#intro)
* [Installation](#install)
* [Exploring the Repo](#explore)

Notebooks:
* [Getting Started](#get_started)
* [Intermideate](#middle)
* [Advanced](#advanced)
* [BLOGS](#blogs)
* [Conference](#conference)
  
---

## Introduction <a name="intro"></a>

Welcome to the community contributed notebooks repo! (formerly known as Notebooks-Extended)

The purpose of this collection of notebooks is to help users understand what RAPIDS has to offer, learn why, how, and when including RAPIDS in a data science pipeline makes sense, and contain community contributions of RAPIDS knowledge. The difference between this repo and the [Notebooks Repo](https://github.com/rapidsai/notebooks) are:
1. These are vetted, community-contributed notebooks (includes RAPIDS team member contributions).  
1. These notebooks won't run on air gapped systems, which is one of our container requirements.  Many RAPIDS notebooks use additional PyData ecosystem packages, and include code for downloading datasets, thus they require network connectivity. If running on a system with no network access, please download all the data that you plan to use ahead of time or simply use the [core notebooks repo](https://github.com/rapidsai/notebooks).


## Installation <a name="install"></a>

Please use the [BUILD.md](BUILD.md) to check the pre-requisite packages and installation steps.

## Contributing

Please see our [guide for contributing to notebooks-contrib](CONTRIBUTING.md).

Once you've followed our guide, please don't forget to [test your notebooks!](TESTING.md) before making a PR.

## Exploring the Repo <a name="explore"></a>
### Folders

- `getting_started_notebooks` - “how to start using RAPIDS”.  Contains notebooks showing "hello worlds", getting started with RAPIDS libraries, and tutorials around RAPIDS concepts.   
- `intermediate_notebooks` - “how to accomplish your workflows with RAPIDS”.  Contains notebooks showing algorithm and workflow examples, benchmarking tools, and some complete end-to-end (E2E) workflows.
- `advanced_notebooks` - "how to master RAPIDS".  Contains notebooks showing kernel customization and advanced end-to-end workflows.
- `blog notebooks` - contains shared notebooks mentioned and used in blogs that showcase RAPIDS workflows and capabilities
- `conference notebooks` - contains notebooks used in conferences, such as GTC
- `data` - contains small data samples used for purely functional demonstrations. Some notebooks include cells that download larger datasets from external websites.

### Lists
- `multimedia_links.md` is a [list of videos](multimedia_links.md) by RAPIDS or our community talking about or showing how to use RAPIDS.  Feel free to contribute your videos and RAPIDS themed playlists as well!
- `competition_notebooks.md` - contains archived notebooks that were used in competitions, such as Kaggle.  Some of these notebooks were blogged about and can also be found in our `blog notebooks` folder.

# Our Notebooks
Below is a listing of the notebooks in this repository.  Each row will tell you the notebook's 
- Location in **Folder**
- Notebook Title and Direct Link in **Notebook Title**
- Description in **Description**
- Design is for a `Single GPU`(SG) or `Multiple GPUs`(MG) in **GPU** (don't worry, you can still run the multi-GPU notebooks with a single GPU)
- Data can be found in **Dataset Used**


## Getting Started Notebooks: <a name="get_started"></a>

| Folder    | Notebook Title         | Description                                                                                                                                      | GPU  | Dataset Used |
|-----------|------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------|------|--------------|
| basics   | [Getting_Started_with_cuDF](getting_started_notebooks/basics/Getting_Started_with_cuDF.ipynb)  | This notebook shows how to get started with GPU DataFrames (single GPU only) using cuDF in RAPIDS.  | SG | Self Generated |
| basics   | [Dask_Hello_World](getting_started_notebooks/basics/Dask_Hello_World.ipynb)           | This notebook shows how to quickly setup Dask and run a "Hello World" example.   | MG | Self Generated |
| basics   | [Getting_Started_with_Dask](getting_started_notebooks/basics/Getting_Started_with_Dask.ipynb)  | This notebook shows how to get started with multi-GPU DataFrames using Dask and cuDF in RAPIDS.    | MG | Self Generated |
| basics   | [hello_streamz](getting_started_notebooks/basics/hello_streamz.ipynb)  | This notebook demonstrates use of cuDF to perform streaming word-count using a small portion of the Streamz API. | SG | Self Generated |     
|basics -> blazingsql| [Getting Started with BlazingSQL](getting_started_notebooks/basics/blazingsql/getting_started_with_blazingsql.ipynb) | How to set up and get started with BlazingSQL and the RAPIDS AI suite. | SG | [Music Dataset](https://github.com/BlazingDB/bsql-demos/blob/master/data/Music.csv) |
|basics -> blazingsql| [Federated Query Demo](getting_started_notebooks/basics/blazingsql/federated_query_demo.ipynb) | In a single query, join an Apache Parquet file, a CSV file, and a GPU DataFrame (GDF) in GPU memory. | SG | [Breast Cancer Diagnostic](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29) |
| intro_tutorials   | [01_Introduction_to_RAPIDS](getting_started_notebooks/intro_tutorials/01_Introduction_to_RAPIDS.ipynb)  | This notebook shows at a high level what each of the packages in RAPIDS are as well as what they do.   |  MG | Self Generated |
| intro_tutorials   | [02_Introduction_to_cuDF](getting_started_notebooks/intro_tutorials/02_Introduction_to_cuDF.ipynb)  | This notebook shows how to work with cuDF DataFrames in RAPIDS.  | SG | Self Generated |
| intro_tutorials   | [03_Introduction_to_Dask](getting_started_notebooks/intro_tutorials/03_Introduction_to_Dask.ipynb)   | This notebook shows how to work with Dask using basic Python primitives like integers and strings.   |  MG | Self Generated |
| intro_tutorials   | [04_Introduction_to_Dask_using_cuDF_DataFrames](getting_started_notebooks/intro_tutorials/04_Introduction_to_Dask_using_cuDF_DataFrames.ipynb)   | This notebook shows how to work with cuDF DataFrames using Dask.   | MG | Self Generated | 
| intro_tutorials   | [06_Introduction_to_Supervised_Learning](getting_started_notebooks/intro_tutorials/06_Introduction_to_Supervised_Learning.ipynb)   | This notebook shows how to do GPU accelerated Supervised Learning in RAPIDS.  |  SG | Self Generated |
| intro_tutorials   | [07_Introduction_to_XGBoost](getting_started_notebooks/intro_tutorials/07_Introduction_to_XGBoost.ipynb)   | This notebook shows how to work with GPU accelerated XGBoost in RAPIDS.   |  SG | Self Generated |
| intro_tutorials   | [08_Introduction_to_Dask_XGBoost](getting_started_notebooks/intro_tutorials/08_Introduction_to_Dask_XGBoost.ipynb)   | This notebook shows how to work with Dask XGBoost in RAPIDS.   | MG | Self Generated |
| intro_tutorials   | [09_Introduction_to_Dimensionality_Reduction](getting_started_notebooks/intro_tutorials/09_Introduction_to_Dimensionality_Reduction.ipynb)   | This notebook shows how to do GPU accelerated Dimensionality Reduction in RAPIDS.  | SG | Self Generated |
| intro_tutorials   | [10_Introduction_to_Clustering](getting_started_notebooks/intro_tutorials/10_Introduction_to_Clustering.ipynb)  | This notebook shows how to do GPU accelerated Clustering in RAPIDS.     |  SG | Self Generated |
---



## Intermediate Notebooks: <a name="middle"></a>
| Folder    | Notebook Title         | Description                                                 | GPU  | Dataset Used |
|-----------|------------------------|-------------------------------------------------------------|------|--------------|
| examples  | [linear_regression_demo.ipynb](intermediate_notebooks/examples/linear_regression_demo.ipynb)    | This notebook demos how to implement simple and multiple linear regression with cuML to predict median housing price on sklearn's Boston Housing dataset. With corresponding [Medium Story](http://bit.ly/cuml_lin_reg_friend). |  SG | [SKLearn Boston Housing](https://scikit-learn.org/stable/datasets/index.html#boston-dataset)|
| examples  | [umap_demo_full](intermediate_notebooks/examples/umap_demo_full.ipynb)     | In this notebook we will show how to use UMAP and its GPU accelerated implementation present in RAPIDS.   | SG | [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist)|
| examples  | [rf_demo](intermediate_notebooks/examples/rf_demo.ipynb)     | Demonstration of using both cuml and sklearn to train a RandomForestClassifier on the Higgs dataset.   | SG | [Higgs Boson](https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz)
| examples  | [weather](intermediate_notebooks/examples/weather.ipynb)     | Demonstration of using Dask and cuDF to process and analyze weather history | MG | [NOAA Annual Weather Data](ftp://ftp.ncdc.noaa.gov/pub/data/ghcn/daily/by_year/) |
| examples -> blazingsql| [BlazingSQL vs Spark](intermediate_notebooks/examples/blazingsql/vs_pyspark_netflow.ipynb) | Analyze 73 million rows of net flow data. Compare BlazingSQL and Apache Spark timings for the same workload.  | SG | [University of New South Wales LanL Dataset](https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/) |
| examples -> blazingsql| [Taxi Fare Prediction](intermediate_notebooks/examples/blazingsql/taxi_fare_prediction.ipynb) | Build & test a cuML Linear Regression model to predict the cost of a ride from 20 million rows of NYC Taxi data. | SG | [NYC Taxi Dataset](https://blazingsql-colab.s3.amazonaws.com/taxi_data/taxi_00.csv) |
| examples -> custreamz   | [parsing_haproxy_logs](intermediate_notebooks/examples/custreamz/parsing_haproxy_logs.ipynb) | This notebook builds upon the weblogs streaming notebook and demonstrates more advanced features for parsing HAProxy logs. | SG | Self Generated
| examples -> cugraph   | [MG Pagerank](intermediate_notebooks/examples/cugraph/multi_gpu_pagerank.ipynb)     | Analyze a Twitter dataset (26GB on disk) with 41.7 million users with 1.47 billion social relations (edges) to find out the most influential profiles.  | MG | [Twitter](https://s3.us-east-2.amazonaws.com/rapidsai-data/cugraph/benchmark/twitter-2010.csv.gz) |
| E2E -> taxi      | [NYCTaxi](intermediate_notebooks/E2E/taxi/NYCTaxi-E2E.ipynb)     | Demonstrates multi-node ETL for cleanup of raw data into cleaned train and test dataframes. Shows how to run multi-node XGBoost training with dask-xgboost.  **Please Note: requires Google Dataproc to run!** [Blog](https://medium.com/rapids-ai/scale-out-rapids-on-google-cloud-dataproc-8a873233258f) | MG | [Google Dataproc Hosted NYC Taxi Data](https://console.cloud.google.com/storage/browser/anaconda-public-data/nyc-taxi/csv/?pli=1) |
| E2E -> synthetic_3D      | [rapids_ml_workflow_demo](intermediate_notebooks/E2E/synthetic_3D/rapids_ml_workflow_demo.ipynb) | A 3D visual showcase of a machine learning workflow with RAPIDS (load data, transform/normalize, train XGBoost model, evaluate accuracy, use model for inference). Along the way we compare the performance gains of RAPIDS [GPU] vs sklearn/pandas methods [CPU].   | SG | SciKit-Learn's demo datasets |
| E2E -> census      | [census_education2income_demo](intermediate_notebooks/E2E/census/census_education2income_demo.ipynb)     | In this notebook we use 50 years of census data to see how education affects income.  | SG | [Custom IPUMS Data pull](https://rapidsai-data.s3.us-east-2.amazonaws.com/datasets/ipums_education2income_1970-2010.csv.gz)
| E2E -> mortgage      | [mortgage_e2e](intermediate_notebooks/E2E/mortgage/mortgage_e2e.ipynb)     | This notebook demonstrates multi-GPU ETL and XGBoost for data preprocessing and training on 17 years of [Fannie Mae’s Single-Family Loan Performance Data](http://www.fanniemae.com/portal/funding-the-market/data/loan-performance-data.html).  | MG | [Mortgage Loan Data](https://docs.rapids.ai/datasets/mortgage-data)
| benchmarks      | [cuml_benchmarks](intermediate_notebooks/benchmarks/cuml_benchmarks.ipynb)  | The purpose of this notebook is to extensively benchmark all of the single GPU cuML algorithms against their skLearn counterparts, while also providing the ability to find and verify upper bounds. **Note: Best on large memory GPUs**                                                             | SG | Self Generated |
| benchmarks      | [rapids_decomposition](intermediate_notebooks/benchmarks/rapids_decomposition.ipynb)  | This notebook benchmarks and visualize RAPIDS decomposition methods against each other.  You have the opportunity to self-compare it to CPU speeds and methods                    | SG | SciKit-Learn's demo datasets |
| benchmarks -> cugraph_benchmarks      | [louvain_benchmark](intermediate_notebooks/benchmarks/cugraph_benchmarks/louvain_benchmark.ipynb)   | This notebook benchmarks performance improvement of running the Louvain clustering algorithm within cuGraph against NetworkX.  |  SG |  Sparse collection                                             | SG | SciKit-Learn's demo datasets |
|  benchmarks -> cugraph_benchmarks    |  [pagerank_benchmark](intermediate_notebooks/benchmarks/cugraph_benchmarks/pagerank_benchmark.ipynb)             | This notebook benchmarks performance improvement of running PageRank within cuGraph against NetworkX. |  SG | Sparse collection     |
|  benchmarks -> cugraph_benchmarks    |  [BFS benchmark](intermediate_notebooks/benchmarks/cugraph_benchmarks/bfs_benchmark.ipynb)             | This notebook benchmarks performance improvement of running BFS within cuGraph against NetworkX. |  SG | Sparse collection   |
|  benchmarks -> cugraph_benchmarks    |  [SSSP_benchmark](intermediate_notebooks/benchmarks/cugraph_benchmarks/sssp_benchmark.ipynb)             | This notebook benchmarks performance improvement of running SSSP within cuGraph against NetworkX. |  SG | Sparse collection   |
|  benchmarks -> cugraph_mg_hibench    |  [MG pagerank_benchmark](intermediate_notebooks/benchmarks/cugraph_mg_hibench/multi_gpu_pagerank.ipynb)             | This notebook runs cuGraph's multi-GPU PageRank on a dataset of 300GB. It designed for DGX-2 machines. | MG | [HiBench](https://rapidsai-data.s3.us-east-2.amazonaws.com/cugraph/benchmark/hibench/HiBench_300GB.tar.gz) |
---


## Advanced Notebooks: <a name="advanced"></a>
| Folder    | Notebook Title         | Description                                              | GPU  | Dataset Used |
|-----------|------------------------|----------------------------------------------------------|------|--------------|
| tutorials   | [rapids_customized_kernels](advanced_notebooks/tutorials/rapids_customized_kernels.ipynb)               |  **Archive Only.**  This notebook shows how create customized kernels using CUDA to make your workflow in RAPIDS even faster.    | SG | Self Generated |
---



## Blog Notebooks: <a name="blogs"></a>
| Folder    | Notebook Title         | Description                                                | GPU  | Dataset Used |
|-----------|------------------------|------------------------------------------------------------|------|--------------|
| cyber   | [flow_classification_rapids](blog_notebooks/cyber/flow_classification/flow_classification_rapids.ipynb)    | **Archive Only.**  The `cyber` folder contains the associated companion files for the blog [GPU Accelerated Cyber Log Parsing with RAPIDS](https://medium.com/rapids-ai/gpu-accelerated-cyber-log-parsing-with-rapids-10896f57eee9), by Bianca Rhodes US, Bhargav Suryadevara, and Nick Becker.  This notebook demonstrates how to load netflow data into cuDF and create a multiclass classification model using XGBoost. Uses [run_raw_data_generator](blog_notebooks/cyber/raw_data_generator/run_raw_data_generator.py)  | SG | [University of New South Wales LanL Dataset](https://iotanalytics.unsw.edu.au/) |
| cyber    | [lanl_network_mapping_using_rapids](blog_notebooks/cyber/network_mapping/lanl_network_mapping_using_rapids.ipynb)      | **Archive Only.**  The `cyber` folder contains the associated companion files for the blog [GPU Accelerated Cyber Log Parsing with RAPIDS](https://medium.com/rapids-ai/gpu-accelerated-cyber-log-parsing-with-rapids-10896f57eee9), by Bianca Rhodes US, Bhargav Suryadevara, and Nick Becker.  This notebook demonstrates how to parse raw windows event logs using cudf and uses cuGraph's pagerank model to build a network graph. Uses [run_raw_data_generator](blog_notebooks/cyber/raw_data_generator/run_raw_data_generator.py)   |  SG | [University of New South Wales LanL Dataset](https://iotanalytics.unsw.edu.au/) |
| databricks   | [RAPIDS_PCA_demo_avro_read](blog_notebooks/databricks/RAPIDS_PCA_demo_avro_read.ipynb)       | The `databricks` folder is the companion file repository to the blog [RAPIDS can now be accessed on Databricks Unified Analytics Platform](https://medium.com/rapids-ai/rapids-can-now-be-accessed-on-databricks-unified-analytics-platform-666e42284bd1) by Ikroop Dhillon, Karthikeyan Rajendran, and Taurean Dyer.  This notebooks purpose is to showcase RAPIDS on Databricks use their sample datasets and show the CPU vs GPU comparison for the PCA algorithm. There is also an accompanying HTML file for easy Databricks import.  **This notebook is for illustrative purposes only!  Do not expect this notebook to successfully run on its own- this notebook's code is replicates a workflow meant to run on a specific platform, `Databricks`**  | SG | [RAPIDS Toy Data](https://s3.us-east-2.amazonaws.com/rapidsai-data/datasets/mortgage/mortgage.npy.gz)|                                                     
| plasticc   | [rapids_lsst_full_demo](blog_notebooks/plasticc/notebooks/rapids_lsst_full_demo.ipynb)               |   **Archive Only.** This notebook demos the full CPU and GPU implementation of the RAPIDS.ai team's model that placed 8/1094 in the PLAsTiCC Astronomical Classification competition.  [Blog](https://medium.com/rapids-ai/make-sense-of-the-universe-with-rapids-ai-d105b0e5ec95).  [Updated notebooks found here](conference_notebooks/KDD_2019/plasticc/)   | MG | [Kaggle PLAsTiCC-2018 dataset](https://www.kaggle.com/c/PLAsTiCC-2018/data) |
| plasticc   | [rapids_lsst_gpu_only_demo](blog_notebooks/plasticc/notebooks/rapids_lsst_gpu_only_demo.ipynb)               |  **Archive Only.** This GPU only based notebook shows the RAPIDS speedup of the RAPIDS.ai team's model that placed 8/1094 in the PLAsTiCC Astronomical Classification competition.  [Blog](https://medium.com/rapids-ai/make-sense-of-the-universe-with-rapids-ai-d105b0e5ec95).  [Updated notebooks found here](conference_notebooks/KDD_2019/plasticc/) | MG | [Kaggle PLAsTiCC-2018 dataset](https://www.kaggle.com/c/PLAsTiCC-2018/data) |
| santander   | [cudf_tf_demo](blog_notebooks/santander/cudf_tf_demo.ipynb)               |  **Archive Only.** This financial industry facing notebook is the cudf-tensorflow approach from the RAPIDS.ai team for Santander Customer Transaction Prediction.   Placed 17/8808. [Blog](https://medium.com/rapids-ai/financial-data-modeling-with-rapids-5bca466f348)   | SG | [Kaggle Santander Customer Transaction Prediction Dataset]( https://www.kaggle.com/c/santander-customer-transaction-prediction/data)
| santander   | [E2E_santander_pandas](blog_notebooks/santander/E2E_santander_pandas.ipynb)               |  **Archive Only.** This This financial data modelling notebook is the Pandas based version the RAPIDS.ai team's best single model for Santander Customer Transaction Prediction competition.  Placed 17/8808. [Blog](https://medium.com/rapids-ai/financial-data-modeling-with-rapids-5bca466f348)   | SG | [Kaggle Santander Customer Transaction Prediction Dataset]( https://www.kaggle.com/c/santander-customer-transaction-prediction/data)
| santander   | [E2E_santander](blog_notebooks/santander/E2E_santander.ipynb)               |  **Archive Only.** This financial data modelling notebook is the cuDF based version of the RAPIDS.ai team's best single model for Santander Customer Transaction Prediction competition.  It allows you to compare cuDF performance to the Pandas version.  Placed 17/8808. [Blog](https://medium.com/rapids-ai/financial-data-modeling-with-rapids-5bca466f348).   |  SG | [Kaggle Santander Customer Transaction Prediction Dataset]( https://www.kaggle.com/c/santander-customer-transaction-prediction/data)
| regression   | [regression_blog_notebook](blog_notebooks/regression/regression_blog_notebook.ipynb)       | This is the companion notebook for the blog [Essential Machine Learning with Linear Models in RAPIDS: part 1 of a series](https://medium.com/rapids-ai/essential-machine-learning-with-linear-models-in-rapids-part-1-of-a-series-992fab0240da) by Paul Mahler.  It showcases an end to end notebook using the Bike Share dataset and cuML's implementation of ridge regression. | SG | [Bike Share Dataset]() |
| regression   | [regression_2_blog](blog_notebooks/regression/regression_2_blog.ipynb)       | This is the companion notebook for the blog [Regression Blog 2: We’re Practically Giving These Regressions Away](https://medium.com/rapids-ai/regression-blog-2-were-practically-giving-these-regressions-away-932669f52d3b) by Paul Mahler.  It showcases an end to end notebook using the Black Friday dataset and cuML's implementations of L1 and L2 regularizations using Ridge, Lasso, and ElasticNet regression techniques. | SG | [Analytics Vidhya Black Friday Hackathon Dataset](https://datahack.analyticsvidhya.com/contest/black-friday/) |
| NLP   | [show_me_the_word_count_gutenberg](blog_notebooks/nlp/show_me_the_word_count_gutenberg/show_me_the_word_count_gutenberg.ipynb)       | This is the notebook for blog [Show Me The Word Count](https://medium.com/rapids-ai/show-me-the-word-count-3146e1173801) by Vibhu Jawa, Nick Becker, David Wendt, and Randy Gelhausen. This notebook showcases NLP pre-processing capabilties of nvstrings+cudf on the Gutenberg dataset. | SG | [Gutenburg Dataset](https://web.eecs.umich.edu/~lahiri/gutenberg_dataset.html) |
|cuspatial   | [accelerate_geospatial_processing](blog_notebooks/cuspatial/trajectory_clustering.ipynb)       | This is the notebook for blog [cuSpatial Accelerates Geospatial and Spatiotemporal Processing](https://medium.com/rapids-ai/releasing-cuspatial-to-accelerate-geospatial-and-spatiotemporal-processing-b686d8b32a9) by Milind Naphade, Jianting Zhang, Shuo Wang, Thomson Comer, Josh Paterson, Keith Kraus, Mark Harris, and Sujit Biswas. This notebook showcases cuSpatial benchmarking of directed Hausdorff distance for computing trajectory clustering on a large dataset. | SG | Trajectories Data and target_intersection.png |
| randomforest   | [fruits_rf_notebook](blog_notebooks/randomforest/fruits_rf_notebook.ipynb)       | This is the notebook for blog [GPU-accelerated Random Forest]() by Vishal Mehta, Myrto Papadopoulou, Thejaswi Rao. This notebook showcases how to use GPU accelerated Random Forest Classification in cuML. The fruit dataset used is Self generated and used as an example in the [Blog](https://medium.com/rapids-ai/accelerating-random-forests-up-to-45x-using-cuml-dfb782a31bea) | SG | Self Generated
| mortgage deep learning      | [mortgage_e2e_deep_learning](blog_notebooks/mortgage_deep_learning/mortgage_e2e_deep_learning.ipynb)     | **Archive Only.** This end to end notebook for the blog, [Using RAPIDS with PyTorch](https://medium.com/rapids-ai/using-rapids-with-pytorch-e602da018285), by Even Oldridge, combines the RAPIDS GPU data processing with a PyTorch deep learning neural network to predict mortgage loan delinquency.   | MG | [Fannie Mae Mortgage Dataset](https://rapidsai.github.io/demos/datasets/mortgage-data)
| svm   | [svc_covertype](blog_notebooks/svm/svc_covertype.ipynb)       | This notebook provides supplementary information for the Benchmark section of the [RAPIDS cuML SVC blog](https://nvda.ws/3c3Qy8H) post. | SG | [UCI Forest covertype dataset](https://archive.ics.uci.edu/ml/datasets/covertype)
---


## Conference Notebooks: <a name="conference"></a>

| Folder    | Notebook Title         | Description                                                         | GPU  | Dataset Used |
| ----------- | ------------------------ | --------------------------------------------------------------- | ---- | ------------ |
| GTC_SJ_2019 | [GTC_tutorial_instructor](conference_notebooks/GTC_SJ_2019/GTC_tutorial_instructor.ipynb) | This is the instructor notebook for the hands on RAPIDS tutorial presented at San Jose's GTC 2019.  It contains all the demonstrated solutions. | SG   | [Analytics Vidhya Black Friday Hackathon Dataset](https://datahack.analyticsvidhya.com/contest/black-friday/) |
| GTC_SJ_2019 | [GTC_tutorial_student](conference_notebooks/GTC_SJ_2019/GTC_tutorial_student.ipynb) | This is the exercise-filled student notebook for the hands on RAPIDS tutorial presented at San Jose's GTC 2019 | SG   | [Analytics Vidhya Black Friday Hackathon Dataset](https://datahack.analyticsvidhya.com/contest/black-friday/) |
|  |  |  |  |  |
| KDD_2019  | [Cybersecurity_KDD](conference_notebooks/KDD_2019/cyber/Cybersecurity_KDD.ipynb) | Using RAPIDS on network traffic and metadata, we demonstrate how to: 1. Triage and perform data exploration, 2. Model network data as a graph, 3. Perform graph analytics on the graph representation of the cyber network data, and 4. Prepare the results in a way that is suitable for visualization. | SG   | [IDS 2018 dataset](https://www.unb.ca/cic/datasets/ids-2018.html) |
| KDD_2019  | [MiningFrequentPatternsFromGraphs](conference_notebooks/KDD_2019/graph_pattern_mining/MiningFrequentPatternsFromGraphs.ipynb) | This notebook uses PC failure metadata, turns it into a coordinate list, and uses cugraph to find frequent patterns about the population that has failed | SG   | [Microsoft PC Failure Metadata Graph](https://s3.us-east-2.amazonaws.com/rapidsai-data/datasets/fpm_graph/coo_fpm.csv.lzma) |
| KDD_2019  | [Part 1.1 RNN Feature Engineering](conference_notebooks/KDD_2019/plasticc/Part_1-1_RNN_Feature_Engineering.ipynb) | Part 1.1 of this GPU only based notebook shows the RAPIDS speedup of the the RAPIDS.ai team's model that placed 8/1094 in the PLAsTiCC Astronomical Classification competition.  [Blog](https://medium.com/rapids-ai/make-sense-of-the-universe-with-rapids-ai-d105b0e5ec95).  - [Introduction found here.](conference_notebooks/KDD_2019/plasticc/Introduction.ipynb)  - [Exercise Answers found here](conference_notebooks/KDD_2019/plasticc/Exercise_Answers.ipynb) - [Original submission found here](competition_notebooks/kaggle/plasticc/notebooks/rapids_lsst_gpu_only_demo.ipynb) | MG   | [Kaggle PLAsTiCC-2018 dataset](https://www.kaggle.com/c/PLAsTiCC-2018/data) |
| KDD_2019  | [Part 1.2 RNN Extract Bottleneck](conference_notebooks/KDD_2019/plasticc/Part_1-2_RNN_Extract_Bottleneck.ipynb) | Part 1.2 of this GPU only based notebook shows the RAPIDS speedup of the RAPIDS.ai team's model that placed 8/1094 in the PLAsTiCC Astronomical Classification competition.  [Blog](https://medium.com/rapids-ai/make-sense-of-the-universe-with-rapids-ai-d105b0e5ec95). - [Introduction found here.](conference_notebooks/KDD_2019/plasticc/Introduction.ipynb)  - [Exercise Answers found here](conference_notebooks/KDD_2019/plasticc/Exercise_Answers.ipynb) - [Original submission found here](competition_notebooks/kaggle/plasticc/notebooks/rapids_lsst_gpu_only_demo.ipynb) | MG   | [Kaggle PLAsTiCC-2018 dataset](https://www.kaggle.com/c/PLAsTiCC-2018/data) |
| KDD_2019  | [Part 2.1 Feature Engineering](contrib/conference_notebooks/KDD_2019/plasticc/Part_2-1_Feature_Engineering.ipynb) | Part 2.1 of this GPU only based notebook shows the RAPIDS speedup of the the RAPIDS.ai team's model that placed 8/1094 in the PLAsTiCC Astronomical Classification competition.  [Blog](https://medium.com/rapids-ai/make-sense-of-the-universe-with-rapids-ai-d105b0e5ec95).  - [Introduction found here.](conference_notebooks/KDD_2019/plasticc/Introduction.ipynb)  - [Exercise Answers found here](conference_notebooks/KDD_2019/plasticc/Exercise_Answers.ipynb) - [Original submission found here](competition_notebooks/kaggle/plasticc/notebooks/rapids_lsst_gpu_only_demo.ipynb) | MG   | [Kaggle PLAsTiCC-2018 dataset](https://www.kaggle.com/c/PLAsTiCC-2018/data) |
| KDD_2019  | [Part 2.2 Train XGBoost & MLP](conference_notebooks/KDD_2019/plasticc/Part_2-2_Train_XGBoost_&_MLP.ipynb) | Part 2.2 of this GPU only based notebook shows the RAPIDS speedup of the the RAPIDS.ai team's model that placed 8/1094 in the PLAsTiCC Astronomical Classification competition.  [Blog](https://medium.com/rapids-ai/make-sense-of-the-universe-with-rapids-ai-d105b0e5ec95).  - [Introduction found here.](conference_notebooks/KDD_2019/plasticc/Introduction.ipynb)  - [Exercise Answers found here](conference_notebooks/KDD_2019/plasticc/Exercise_Answers.ipynb) - [Original submission found here](competition_notebooks/kaggle/plasticc/notebooks/rapids_lsst_gpu_only_demo.ipynb) | MG   | [Kaggle PLAsTiCC-2018 dataset](https://www.kaggle.com/c/PLAsTiCC-2018/data) |
|  |  |  |  |  |
| SCIPY_2019 | [SCIPY_2019 Tutorial Index](conference_notebooks/SCIPY_2019/index.ipynb) | This index outlines the "getting started" style tutorials within the folder.  The tutorials cover cudf, cuml, and cugraph.  These tutorials were presented at SCIPY 2019 | SG   | Various Self Generated datasets and Zachary Karate Club Data Set |
|  |  |  |  |  |
| ASONAM 2019 | [Cyber](conference_notebooks/ASONAM_2019/Cyber.ipynb) | Example notebook using RAPIDS to let an organization's security and forensics experts collect vast amounts of network traffic and network metadata and perform fast triage, processing, modeling, and visualization capabilities. | MG   | [IDS 2018 dataset](https://www.unb.ca/cic/datasets/ids-2018.html) from the [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/) |
| ASONAM 2019 | [Spotify Playlist](conference_notebooks/ASONAM_2019/Spotify_Playlist.ipynb) | Shows how you can quickly use RAPIDS to explore the Spotify Million Playlist Dataset, which was created for the RecSys 2018 competition, and build a playlist recommender **Note: this dataset requires an independent user download and cannot be pulled from the notebook** | MG   | RecSys 2018 competition |
| ASONAM 2019 | [Weighted Link Prediction](conference_notebooks/ASONAM_2019/Weighted_Link_Prediction.ipynb) | This notebook uses cuGraph for Weighted Link Prediction to mitigate uncertainty on the Epinions Trust Network Dataset to predict the likelihood of trust or distrust between vertices. **Note: this dataset requires an independent user download and cannot be pulled from the notebook** | SG   | Epinions Trust Network Dataset |
|  |  |  |  |  |
| KDD 2020    | [KDD 2020](conference_notebooks/KDD_2020/README.md) | Conference material for the KDD 2020 hands-on tutorial  | SG  |   |
| KDD 2020 | [Taxi](conference_notebooks/KDD_2020/notebooks/Taxi/NYCTax.ipynb)  |  Analysis of the New York City Taxi dataset​. Introductory notebook showing ETL, Statistical Analysis, Machine Learning, Graph, and Visualization     | SG   |    2016 New York Taxi Data    |
| KDD 2020 | [Tabular](conference_notebooks/KDD_2020/notebooks/nvtabular/rossmann-store-sales-example.ipynb)  |  Perform store sales prediction using tabular deep learning​    | SG   |    [ Kaggle Rossmann Store Sales competition](https://www.kaggle.com/c/rossmann-store-sales)    |
| KDD 2020 | [Cell RNA](conference_notebooks/KDD_2020/notebooks/Lungs/hlca_lung_gpu_analysis.ipynb)  |  Single-Cell RNA Sequencing Analysis​​    | SG   | human lung cells from [Travaglini et al. 2020](https://www.biorxiv.org/content/10.1101/742320v2) |
| KDD 2020 | Parking |  Analyzing Seattle Parking data and determining the best parking spot within a walkable distance from Space Needle​​​    | SG   |  |
| KDD 2020 | CyBERT |      Cyber Log Parsing using Neural Networks and Language Based Model​    | SG   |  |

## Additional Information
* The `data` folder also includes the full image set from the [Fashion MNIST dataset](https://github.com/zalandoresearch/fashion-mnist).

* `utils`: contains a set of useful scripts for interacting with RAPIDS Notebooks-Contrib

* For our notebook examples and tutorials found in our standard containers, please see the [Notebooks Repo](https://github.com/rapidsai/notebooks)
