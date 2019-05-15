## Installation

Please ensure you meet the [pre-requisites](https://rapids.ai/start.html#prerequisites)

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
