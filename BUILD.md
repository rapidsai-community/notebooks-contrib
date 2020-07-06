
A Dockerfile is provided for installing pre-requisite packages & launching JupyterLab. Please ensure you meet the [pre-requisites](https://rapids.ai/start.html#prerequisites) requirements.

# Installation

## 
Our container extends the base notebooks container. Unlike our Notebooks repo, which comes with the container, Notebooks Contrib is meant to be a constantly updating source of community contributions. You can run Notebooks Contrib in a container with 4 steps:

Step 1: Clone the Notebooks Contrib repo
```bash
git clone https://github.com/rapidsai/notebooks-contrib
```

Step 2: Build the Docker container
```bash
make build
```

Step 3: Run the Docker container
```bash
make run
```

Step 4: Access the running Jupyter notebook instance at
```text
http://localhost:8888
```