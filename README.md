# RAPIDS Extended Notebooks
## Intro
Welcome to the auxiliary notebooks repo!  The purpose of this collection of notebooks is to help prospective users experience the power of RAPIDS, understand why they would want to implement RAPIDS in their data science pipeline, and help them get started and grow their skills to master the RAPIDS libraries.  
This collection started as the community notebooks that require external sources for libraries and data, so it requires a network connected workstation.  If you plan to explore on an air gapped system, please use our core notebooks repo.  
## Exploring the Repo
Our notebooks are split into two major folders, with subfolders
- `cpu_comparison_notebooks` - contains notebooks that will let you experience “why RAPIDS’ by directly comparing the data science compute time between popular single and multi threaded CPU implementations and RAPIDS GPU implementation.  You’ll also see how similar RAPIDS APIs are to your favorite data science packages.  
Notebooks in here include: 
    - DBScan Demo Full
    - XGBoost
    - Linear Regression
    - Ridge Regression
    - PCA
- tutorial_notebooks` - contains notebooks that will help you learn “how [to master] RAPIDS”.  It contains these subfolders
    - `getting_started` - to help you quickly learn the basic RAPIDS APIs.  It contains these notebooks
        - Dask Hello World
        - Getting Started with cuDF
    - `examples` - which will show you how to set up and implement your RAPIDS accelerated data science pipelines
        - Dask with cuDF and XGBoost
        - Dask with cuDF and XGBoost from Disk
    - `advanced` - these notebooks show you the power of the RAPIDS libraries unleashed to solve real world problems.  
        - PLASTICC 

The `data` folder is where we save our data for our examples.  There is some small, initial data set in there, but some notebooks may require you to download larger datasets for external websites.
Finally, there is a DOCKERFILE for building a container from source as well as the README, which you are viewing now!

## Installation
### Prerequisites for Installation
Please see our getting started page on our website for full installation requirements and instructions
-	Workstation or Cloud Instance with a Pascal, Tesla, Volta, or Turning GPU (Maxwell , Kepler, and Fermi GPUs are not supported)
-	Ubuntu 16.04 or 18.04
-	Python 3.6 or 3.7
-	Docker CE
-	Nvidia-Docker

### Getting Containers
We will host a precompiled docker container here.  Please run this docker command to pull our late
If you prefer to get the most cutting edge notebooks, or explore a different branch, we have included a docker file to build this notebook by following the instructions below
## Contributing Back
You can contribute to this repo in 5 ways:
1. Finding bugs in existing notebooks – sometimes things unexpectedly break in our notebooks.  Please raise a bug issue appropriately and let us know so that we can quickly fix it!
1. Peer reviewing and benchmarking new and existing notebooks – Both new and existing notebooks need to be checked against current and new RAPIDS library releases.  Your help is truly appreciated in making sure that those notebooks not just work, but are efficient, effective and, well, run rapidly
1. Suggesting notebook content that would interest you – We create notebooks that we think are useful to you, but what do we know?  You know what’s useful to you.  Please tell us by suggesting a new notebook or upvoting an existing suggestion!
1. Creating and submitting a great, new notebook – A good notebooks takes time and  effort so it would be great if you would share the load and enhance the community’s knowledge and capabilities
1. Writing a tutorial blog - 
