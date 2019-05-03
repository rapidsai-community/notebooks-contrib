# RAPIDS Extended Notebooks
## Intro
Welcome to the extended notebooks repo!

The purpose of this collection of notebooks is to help users understand what RAPIDS has to offer, learn how, why, and when including it in data science pipelines makes sense, and help get started using RAPIDS libraries by example. 

Many of these notebooks use additional PyData ecosystem packages, and include code for downloading datasets, thus they require network connectivity. If running on a system with no network access, and you want to use Notebooks Extended, you may have to manually download and transfer any packages and data sources from a machine that has network access.

## Exploring the Repo
```This is about to change.  A new folder hierarchy is coming with 0.7!```
Notebooks live under two subfolders:
- `cpu_comparisons` - these notebooks demonstrate "why RAPIDS" by directly comparing compute time between single and multi threaded CPU implementations vs GPU (RAPIDS library) implementations. Of note here is the similarity of RAPIDS APIs to common PyData ecosystem packages like Pandas and scikit-learn. Notebooks in here include: 
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

`/data` contains small data samples used for purely functional demonstrations. Some notebooks include cells that download larger datasets from external websites.

The `/data` folder is also symlinked into `/rapids/notebooks/extended/data` so you can browse it from JupyterLab's UI.

Lastly, a Dockerfile is provided for installing pre-requisite packages & launching JupyterLab.

## Installation

Please ensure you meet the [pre-requisites](https://rapids.ai/start.html#prerequisites)

Our container extends the base notebooks container. Unlike our Notebooks repo, which comes with the container, Notebooks Extended is meant to be a constatly updating source of community contributions.  We've made it easy to include Notebooks Extended in your RAPIDS container with 3 easy steps:

Step 1: Download your RAPIDS container
```bash
docker pull rapidsai/rapidsai:latest
```

Step 2: Git Pull Notebooks Extended into the folder of your choice (change "#/folder/of/your/choice" into wherever you desire Notebooks Extended to be). 

```bash
mkdir #/folder/of/your/choice
git clone https://github.com/rapidsai/notebooks-extended
```

Step 3: Run Docker mounting your Notebooks Extended folder as a volume, changing "/folder/of/your/choice/" to where you put Notebooks Extended
```bash
docker run --runtime=nvidia --rm -it -p 8888:8888 -p 8787:8787 -p 8786:8786 -v /folder/of/your/choice/ -it rapidsai/rapidsai:latest
```


To use previous versions of Notebooks Extended, do a `git checkout` to the relevant commit.

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
