# cuGraph Multi-GPU example

## Dependencies
Multi-GPU notebooks have the following dependencies: 

      cugraph>=0.10      
      cudf>=0.10
      rmm>=0.10
      dask-cudf>=0.10
      dask-cuda>=0.10
      cudatoolkit>=9.2
      dask>=2.1.0 
      distributed>=2.1.0 

The simplest way to get all dependencies is through conda, by following the [instructions](https://github.com/rapidsai/cugraph/blob/master/CONTRIBUTING.md) to get the `cugraph_dev` environment.


## Get the Data Set

This step involves downloading a `gz` file of 6GB. Once decompressed the file is 26GB.

Option 1 : Run the script
```bash
sh ./get_data.sh
```

Option 2 : manually run
```bash
wget https://s3.us-east-2.amazonaws.com/rapidsai-data/cugraph/benchmark/twitter-2010.csv.gz
gunzip twitter-2010.csv.gz  
```


**Graph info**

| File Name    | Num of Vertices | Num of Edges    | File size  |
| -------------| --------------: | -------------:  | ---------: |
| twitter      |      41,652,230 | 1,468,365,182   |       26GB | 


## Step 2:  Open the notebook
```bash
ipython notebook multi_gpu_pagerank.ipynb
```
