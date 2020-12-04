
A Dockerfile is provided for installing pre-requisite packages & launching JupyterLab. Please ensure you meet the [pre-requisites](https://rapids.ai/start.html#prerequisites) requirements.

# Installation

## 
The custreamz container has all necessary dependencies for running the examples installed (on a GPU enabled host of course). You can run Notebooks Contrib in a container with 4 simple steps:

Step 1: Clone the Notebooks Contrib repo
```bash
git clone https://github.com/rapidsai/notebooks-contrib && cd notebooks-contrib/intermediate_notebooks/examples/custreamz
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

