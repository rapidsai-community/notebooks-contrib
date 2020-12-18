# **Intro to RAPIDS Course for Content Creators**
## Introduction

In this intro course, we cover the basic skills you need to accelerate your data analytics and ML pipeline with RAPIDS.  Get to know the RAPIDS core libraries: cuDF, cuML, cuGraph, and cuXFilter, as well as community libraries, including: XGBoost, Dask, and BlazingSQL, to accelerate how you: 
- Ingest data
- Perform your prepare your data with ETL
- Run modeling, inferencing, and predicting algorithms on the data in a GPU dataframe
- Visualize your data throughout the process.  

Each of the three modules should take less than 2 hours to complete.  When complete, you should be able to:
1. Take an existing workflow in a data science or ML pipeline and use a RAPIDS to accelerate it with your GPU
1. Create your own workflows from scratch

This course was written with the expectation that you know Python and Jupyter Lab.  It is helpful, but not necessary, to have at least some understanding of Pandas, Scikit Learn, NetworkX, and Datashader. 

[You should be able to run these exercises and use these libraries on any machine with these prerequisites](https://rapids.ai/start.html#PREREQUISITES), which namely are 
- OS of Ubuntu 16.04 or 18.04 or CentOS7 with gcc 5.4 & 7.3
- an NVIDIA GPU of Pascal Architecture or better (basically 10xx series or newer)

RAPIDS works on a broad range of GPUs, including NVIDIA GeForce, TITAN, Quadro, Tesla, A100, and DGX systems
## NVIDIA Titan RTX
- [NVIDIA Spot on Titan RTX and RAPIDS](https://www.youtube.com/watch?v=tsWPeZTLpkU)
- [t-SNE 600x Speed up on Titan RTX](https://www.youtube.com/watch?v=_4OehmMYr44)



## Questions?
There are a few channels to ask questions or start a discussion:
- [GoAI Slack](https://join.slack.com/t/rapids-goai/shared_invite/enQtMjE0Njg5NDQ1MDQxLTJiN2FkNTFkYmQ2YjY1OGI4NTc5Y2NlODQ3ZDdiODEwYmRiNTFhMzNlNTU5ZWJhZjA3NTg4NDZkMThkNTkxMGQ) to discuss issues and troubleshoot with the RAPIDS community
- [RAPIDS GitHub](https://github.com/rapidsai) to submit feature requests and report bugs

# **Getting Started**
There are 3 steps to installing RAPIDS
1. Provisioning a GPU enabled workspace
1. Installing RAPIDS Prerequisites
1. Installing RAPIDS libraries

## 1. Provisioning a GPU-Enabled Workspace
When installing RAPIDS, first provision a RAPIDS Compatible GPU.  The GPU must be **NVIDIA Pascal™ or better with compute capability 6.0+**.  Here is a list of compatible GPUs. This GPU can local, like in a workstation or server, or in the Cloud. GPUs can reside in:
- Shared cloud
- Dedicated cloud
- Local workspace

### Using Cloud Instance(s)
There are two option for using Cloud Instances: 
1. Shared, **free** instances like app.blazingsql.com and Google Colab
1. Dedicated, **paid** [usually]  [GPU instances from providers like AWS, Azure, GCP, Paperspace, and more](https://rapids.ai/cloud.html)

### Shared Cloud via Free Instances
Free cloud instances have quick start capabilities or scripts to ease onboarding.  
- **Google Colab**: The installation will take about 8 minutes.  First select a GPU instance from Runtime type.  After, use the provided RAPIDS installation scripts, found here by copying and pasting into a code cell.  Please note, RAPIDS will not run on an unsupported GPU instance like K80 - ONLY the T4, P4, and P100s (Refer to `!nvidia-smi`).  If you are given a K80, please factory reset your instance and the check again.  
- **app.blazingsql.com**: these instances are preloaded with RAPIDS and you can start right away

### Dedicated Cloud via Paid Instances
There are several ways to provision a dedicated cloud GPU workspace, and our instructions are found here.  Your OS will need to be **Ubuntu or RHEL/CentOS 7**.  For installing RAPIDS, These instances follow the same installation process as a local instance.  

## 2. Installing RAPIDS Prerequisites
### Downloads
You can satisfy your prerequisites to install RAPIDS by:
1. Install OS and GPU Drivers and OS
1. Install Packaging Environment (Docker or Conda)

### OS and GPU Drivers
 Please ensure that your workstation has these installed as our prerequisites are as follows:
- GPU: NVIDIA Pascal™ or better with compute capability 6.0+ (completed above)
- OS: Ubuntu 16.04/18.04 or CentOS 7 with gcc/++ 7.5+
  - See RSN 1 for details on our recent update to gcc/++ 7.5
  - RHEL 7 support is provided through CentOS 7 builds/installs
- CUDA & NVIDIA Drivers: One of the following supported versions:
  - 10.0 & v410.48+ (valid option for version 0.14 and earlier only)
  - 10.1.2 & v418.87+
  - 10.2 & v440.33+  
  - 11.0 (valid option for version 0.16 and later)
- Python 
  - 3.6 (valid option for version 0.14 and earlier)
  - 3.7 
  - 3.8 (valid option for version 0.16 and later)


### Install Packaging Environment (Docker or Conda)
Depending on if you prefer to use RAPIDS with Docker or Conda, you will need these also installed:

- If Docker: Docker CE v19.03+ and nvidia-container-toolkit
  - Legacy Support - Docker CE v17-18 and nvidia-docker2

- If Conda, please install 
 - [Miniconda](https://conda.io/miniconda.html) for a minimal conda installation 
 - [Anaconda](https://www.anaconda.com/download) for full conda installation
 - [Mamba inside of conda](https://github.com/TheSnakePit/mamba) for a faster conda solving (untested)

### 3. Install RAPIDS Libraries

- Use the [Interactive RAPIDS release selector](https://rapids.ai/start.html#rapids-release-selector) to install RAPIDS as you want it.  The install script at the bottom will update as you change your install parameters of **method, desired RAPIDS release, desired RAPIDS packages, Linux verison, and CUDA version**.  Here is an image of it below.

# <add image>
Great!  Now that you're done getting up and running, let's move on to the Data Science!

## **1. The Basics of RAPIDS: cuDF and Dask**
### Introduction
cuDF lets you create and manipulate your dataframes on GPUs. All other RAPIDS libraries use cuDF to model, infer, regress, reduce, and predict outcomes. The cuDF API is designed to be similar to Pandas with minimal code changes.  
- [latest RAPIDS cuDF documentation](https://docs.rapids.ai/api)
- [RAPIDS cuDF GitHub repo](https://github.com/rapidsai/cudf)

There are situations where the dataframe is larger than available GPU memory.  Dask is used to help RAPIDS algorithms scale up through distributed computing.  Whether you have a single GPU, multiple GPUs, or clusters of multiple GPUs, Dask is used for distributed computing calculations and orchstrattion of the processing of GPU dataframe, no matter the size, just like a regular CPU cluster.  

Let's get started with a couple videos!

### Videos

| Video Title         | Description |
|------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [Video- Getting Started with RAPIDS](https://www.youtube.com/watch?v=T2AU0iVbY5A).  | Walks through the [01_Introduction_to_RAPIDS](intro_tutorials_and_guides/01_Introduction_to_RAPIDS.ipynb) notebook which shows, at a high level, what each of the packages in RAPIDS are as well as what they do. |
| [Video - RAPIDS: Dask and cuDF NYCTaxi Screencast](https://www.youtube.com/watch?v=gV0cykgsTPM) | Shows you have you can use RAPIDS and Dask to easily ingest and model a large dataset (1 year's worth of NYCTaxi data) and then create a model around the question "when do you get the best tips".  This same workload can be done on any GPU. |

### Learning Notebooks


| Notebook Title         | Description |
|------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [01_Introduction_to_RAPIDS](intro_tutorials_and_guides/01_Introduction_to_RAPIDS.ipynb)  | This notebook shows at a high level what each of the packages in RAPIDS are as well as what they do.  |                                                                                                                                    
| [02_Introduction_to_cuDF](intro_tutorials_and_guides/02_Introduction_to_cuDF.ipynb)  | This notebook shows how to work with cuDF DataFrames in RAPIDS.                                                                                                                                      |
| [03_Introduction_to_Dask](intro_tutorials_and_guides/03_Introduction_to_Dask.ipynb)   | This notebook shows how to work with Dask using basic Python primitives like integers and strings.                                                                                                                                      |
| [04_Introduction_to_Dask_using_cuDF_DataFrames](intro_tutorials_and_guides/04_Introduction_to_Dask_using_cuDF_DataFrames.ipynb)   | This notebook shows how to work with cuDF DataFrames using Dask.                                                                                                                                      |
| [Guide to UDFs](https://github.com/rapidsai/cudf/blob/branch-0.18/docs/cudf/source/guide-to-udfs.ipynb) | This notebook provides and overview of User Defined Functions with cuDF |



### Extra credit and Exercises
- [10 minute review of cuDF](https://github.com/rapidsai/cudf/blob/branch-0.18/docs/cudf/source/10min.ipynb)
- [Extra Credit - 10 minute guide to cuDF and cuPY](https://github.com/rapidsai/cudf/blob/branch-0.18/docs/cudf/source/10min-cudf-cupy.ipynb)
- [Extra Credit - Multi-GPU with Dask-cuDF](https://rapidsai.github.io/projects/cudf/en/0.18.0/dask-cudf.html)
- [Review and Exercises 1- Review of cuDF](../the_archive/archived_rapids_event_notebooks/SCIPY_2019/cudf/01-Intro_to_cuDF.ipynb)
- [Review and Exercises 2- Creating User Defined Functions (UDFs) in cuDF](../the_archive/archived_rapids_event_notebooks/SCIPY_2019/cudf/02-Intro_to_cuDF_UDFs.ipynb)

## **2. Accelerating those Algorithms: cuML and XGBoost**
### Introduction
Congrats learning the basics of cuDF and Dask.  Now let's take a look at cuML

cuML runs many common scikit-learn algorithms and methods on cuDF dataframes to model, infer, regress, reduce, and predict outcomes on the data. [Among the ever growing suite of algorithms, you can perform several GPU accelerated algortihms for each of these methods:]()

- Classification / Regression
- Inference
- Clustering
- Decomposition & Dimensionality Reduction
- Time Series

While we look at cuML , we'll take a look at how further on how to increase your speed up with [XGBoost](https://machinelearningmastery.com/gentle-introduction-xgboost-applied-machine-learning/), scale it out with Dask XGboost, then see how to use cuML for Dimensionality Reduction and Clustering.
- [latest RAPIDS cuML documentation](https://docs.rapids.ai/api)
- [RAPIDS cuML GitHub repo](https://github.com/rapidsai/cuml)

Let's look at a few video walkthroughs of XGBoost, as it may be an unfamiliar concept to some, and then experience how to use the above in your learning notebooks.  

### Videos

| Video Title         | Description |
|------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [Video - Introduction to XGBoost](https://www.youtube.com/watch?v=EQR3bP6XFW0) | Walks through the [07_Introduction_to_XGBoost](getting_started_notebooks/intro_tutorials/07_Introduction_to_XGBoost.ipynb) notebook and shows how to work with GPU accelerated XGBoost in RAPIDS. |
| [Video - Introduction to Dask XGBoost](https://www.youtube.com/watch?v=q8HfEZythjM) |  Walks through the [08_Introduction_to_Dask_XGBoost](getting_started_notebooks/intro_tutorials/08_Introduction_to_Dask_XGBoost.ipynb) notebook and hows how to work with Dask XGBoost in RAPIDS.  This can be run on a single GPU as well and is useful when your dataset is larger than the memory size of your GPU. Will be deprecated in 0.15, and removed in 0.16                 |

### Learning Notebooks

| Notebook Title         | Description |
|------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [06_Introduction_to_Supervised_Learning](intro_tutorials_and_guides/06_Introduction_to_Supervised_Learning.ipynb)   | This notebook shows how to do GPU accelerated Supervised Learning in RAPIDS.                                                                                                                                      |
| [07_Introduction_to_XGBoost](intro_tutorials_and_guides/07_Introduction_to_XGBoost.ipynb)   | This notebook shows how to work with GPU accelerated XGBoost in RAPIDS.                                                                                                                                      |
| [09_Introduction_to_Dimensionality_Reduction](intro_tutorials_and_guides/09_Introduction_to_Dimensionality_Reduction.ipynb)   | This notebook shows how to do GPU accelerated Dimensionality Reduction in RAPIDS.                                                                                                                                      |
| [10_Introduction_to_Clustering](intro_tutorials_and_guides/10_Introduction_to_Clustering.ipynb)  | This notebook shows how to do GPU accelerated Clustering in RAPIDS. |


### Extra credit and Exercises

- [10 Review of cuML Estimators](https://github.com/rapidsai/cuml/blob/branch-0.18/docs/source/estimator_intro.ipynb)

- [Review and Exercises 1 - Linear Regression](../the_archive/archived_rapids_event_notebooks/SCIPY_2019/cuml/01-Introduction-LinearRegression-Hyperparam.ipynb)

- [Review and Exercises 2 -  Logistic Regression](../the_archive/archived_rapids_event_notebooks/SCIPY_2019/cuml/02-LogisticRegression.ipynb)

- [Review and Exercises 3- Intro to UMAP](../the_archive/archived_rapids_event_notebooks/SCIPY_2019/cuml/03-UMAP.ipynb)

### RAPIDS cuML Example Notebooks
- [Index of Notebooks](https://github.com/rapidsai/notebooks#cuml-notebooks)
- [Direct Link to Notebooks](https://github.com/rapidsai/notebooks/tree/branch-0.18/cuml)


### Conclusion to Sections 1 and 2
Here ends the basics of cuDF, cuML, Dask, and XGBoost.  These are libraries that everyone who uses RAPIDS will go to every day.  Our next sections will cover libraries that are more niche in usage, but are powerful to accomplish your analytics.

## **3. Graphs on RAPIDS: Intro to cuGraph**

It is often useful to look at the relationships contained in the data, which we do that thought the use of graph analytics. Representing data as a graph is an extremely powerful techniques that has grown in popularity.  Graph analytics are used to helps Netflix recommend shows, Google rank sites in their search engine, connects bits of discrete knowledge into a comprehensive corpus, schedules NFL games, and can even help you optimize seating for your wedding (and it works too!). [KDNuggests has a great in depth guide to graphs here](https://www.kdnuggets.com/2017/12/graph-analytics-using-big-data.html).  Up until now, running a graph analytics was a painfully slow, particularly as the size of the graph (number of nodes and edges) grew.

[RAPIDS' cuGraph library makes graph analytics effortless, as it boasts some of our best speedups](https://www.zdnet.com/article/nvidia-rapids-cugraph-making-graph-analysis-ubiquitous/), (up to 25,000x).  To put it in persepctive, what can take over 20 hours, cuGraph can lets you do in less than a minute (3 seconds).  In this section, we'll look at some examples of cuGraph methods for your graph analytics and look at a simple use case.
- [latest RAPIDS cuGraph documentation](https://docs.rapids.ai/api)
- [RAPIDS cuGraph GitHub repo](https://github.com/rapidsai/cugraph)

### RAPIDS cuGraph Example Notebooks
- [Index of Notebooks](https://github.com/rapidsai/notebooks/#cugraph-notebooks)
- [Direct Link to Notebooks](https://github.com/rapidsai/notebooks/tree/branch-0.18/cugraph)
"""