Benchmark Templates
===================

Rapidsai Notebooks can be used as benchmark scripts by using nbconvert + these templates to generate an executable python module.

The shortest path to using these templates is to build and run the docker image described by [benchmarks/Dockerfile](Dockerfile).

```
docker build --build-arg GROUP_ID=$(id -g) --build-arg USER_ID=$(id -u) --build-arg CONDA_USER=${USER} -t nbbench:cuda10.0_ubuntu18.04 benchmarks

```

There is a helper script to convert notebooks. It will accept one or more notebooks or paths with glob expressions.

```
./benchmarks/convert.sh intermediate_notebooks/examples/weather.ipynb
```

The benchmarks are executed, measured and reported using a tool called [ASV (Airspeed Velocity)](https://github.com/airspeed-velocity/asv). This configuration uses a plugin, [condarepo-asv-plugin](https://github.com/where/condarepo-asv-plugin) to track benchmarks relative to nightly builds of RapidsAI projects.

```
asv run -v -e NEW

asv publish
```

This will execute benchmarks for the latest build of RapidsAI packages and write the results to `.asv/results` in the notebooks-contrib project.
The publish command will generate a static HTML view of the results which can be displayed.

To view the results 

```
cd .asv/html

python -m 'http.server'
```
