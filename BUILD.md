
A Dockerfile is provided for installing pre-requisite packages & launching JupyterLab. Please ensure you meet the [pre-requisites](https://rapids.ai/start.html#prerequisites) requirements.

# Installation

## 
Our container extends the base notebooks container. Unlike our Notebooks repo, which comes with the container, Notebooks Contrib is meant to be a constatly updating source of community contributions. You can run Notebooks Contrib in a container with 3 steps:

Step 1: Download your RAPIDS container
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

