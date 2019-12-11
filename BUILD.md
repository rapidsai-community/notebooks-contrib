
A Dockerfile is provided for installing pre-requisite packages & launching JupyterLab. Please ensure you meet the [pre-requisites](https://rapids.ai/start.html#prerequisites) requirements.

# Installation

## Downloada and Volumize
Our container extends the base notebooks container. Unlike our Notebooks repo, which comes with the container, Notebooks Contrib is meant to be a constatly updating source of community contributions. You can run Notebooks Contrib in a container with 3 steps (the example shows how if your are using the latest stable branch):

Here is
Step 1: Download your preferred RAPIDS container from here: https://rapids.ai/start.html#get-rapids
```bash
docker pull rapidsai/rapidsai:latest
```
Step 2: Pull the Notebooks Contrib git repo
```bash
git clone https://github.com/rapidsai/notebooks-contrib
```
Step 3: Run the jupyter server in the docker container:
```bash
cd notebooks-contrib
docker run --runtime=nvidia --rm -it -p 8888:8888 -p 8787:8787 -p 8786:8786 -v /folder/of/your/choice/:notebooks/contrib -it rapidsai/rapidsai:latest
utils/start-jupyter.sh
```

## Using the Right Branch
Notebooks Contrib has to work with both the stable releases and the nightly releases.  To accomplish this, starting in v0.10, we are now patterning Notebook Contrib's branching to be similar to that of Notebooks.  Please switch branches accordingly.

- If Running RAPIDS **Nightlies** versions- Latest nightlies notebooks and updates will be `branch-latest`.  It will have the latest updates to both notebooks and docs.
- If Running RAPIDS **Stable** versions- Current stable release versions of notebooks will be in `master`.  It will also have any recent permalinked materials (blogs, conferences, etc).
- If Running RAPIDS **Older** versions- Older versions of notebooks compatible with older releases with be in `branch-<version numbers>`.  These branches will not be maintained.
