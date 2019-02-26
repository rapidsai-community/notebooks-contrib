# RAPIDS Extended Notebooks
## Intro
Welcome to the extended notebooks repo!

The purpose of this collection of notebooks is to help users understand what RAPIDS has to offer, learn how, why, and when including it in data science pipelines makes sense, and help get started using RAPIDS libraries by example. 

Many of these notebooks use additional PyData ecosystem packages, and include code for downloading datasets, thus they require network connectivity. If running on a system with no network access, please use the [core notebooks repo](https://github.com/rapidsai/notebooks).

## Exploring the Repo
Notebooks live under two subfolders:
- `cpu_comparisons` - these notebooks demonstrate “why RAPIDS’ by directly comparing compute time between single and multi threaded CPU implementations vs GPU (RAPIDS library) implementations. Of note here is the similarity of RAPIDS APIs to common PyData ecosystem packages like Pandas and scikit-learn. Notebooks in here include: 
    - DBScan Demo Full
    - XGBoost
    - Linear Regression
    - Ridge Regression
    - PCA
- `tutorials` - contains notebooks showing “how [to master] RAPIDS”:
    - `getting_started` - to help you quickly learn the basic RAPIDS APIs.  It contains these notebooks
        - Dask Hello World
        - Getting Started with cuDF
    - `examples` - which will show you how to set up and implement your RAPIDS accelerated data science pipelines
        - Dask with cuDF and XGBoost
        - Dask with cuDF and XGBoost from Disk
    - `advanced` - these notebooks show you the power of the RAPIDS libraries unleashed to solve real world problems.  
        - PLASTICC 

The `data` folder contains small data samples used in running purely functional examples. Some notebooks include cells that download larger datasets from external websites.

Lastly, a Dockerfile is provided for installing pre-requisite packages & launching JupyterLab.

## Installation

Please ensure you meet the [pre-requisites](https://rapids.ai/start.html#prerequisites)

Our container extends the base notebooks container. To build:
```
git clone https://github.com/rapidsai/notebooks-extended
cd notebooks-extended
docker build -t notebooks-extended .
```

To use previous versions of the notebooks, do a `git checkout` to the relevant commit.

## Contributing
You can contribute to this repo in 5 ways:
1. Finding bugs in existing notebooks:
   Sometimes things unexpectedly break in our notebooks. Please raise an issue so we can fix it!
2. Peer reviewing and benchmarking new and existing notebooks:
   Both new and existing notebooks need to be checked against current and new RAPIDS library releases. Your help is truly appreciated in making sure that those notebooks not just work, but are efficient, effective and, well, run rapidly
3. Suggesting notebook content that would interest you:
   We create notebooks that we think are useful to you, but what do we know? You know what’s useful to you. Please tell us by suggesting a new notebook or upvoting an existing suggestion!
4. Creating and submitting a new notebook:
   A good notebooks takes time and effort so it would be great if you would share the load and enhance the community’s knowledge and capabilities
5. Writing a tutorial blog:
   Show your expertise in RAPIDS while teaching people how to use RAPIDS in their data science pipeline.
