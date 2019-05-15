
A Dockerfile is provided for installing pre-requisite packages & launching JupyterLab. Please ensure you meet the [pre-requisites](https://rapids.ai/start.html#prerequisites) requirements.

## Installation

Our container extends the base notebooks container. To build:
```bash
git clone https://github.com/rapidsai/notebooks-extended
cd notebooks-extended
docker build -t notebooks-extended .
```

To run:
```bash
docker run -p 8888:8888 -it notebooks-extended
```

To use previous versions of the notebooks, do a `git checkout` to the relevant commit.
