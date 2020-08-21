# Accelerating and Expanding End-to-End Data Science Workflows with DL/ML Interoperability Using RAPIDS
## Analyzing the Paid Parking Occupancy dataset from Seattle Department of Transportation

In this part of the hands-on tutorial session we have three notebooks:

1. [Where should I park?](codes/1_rapids_seattleParking.ipynb) Using this notebook we will find the parking spots that maximize your chances of having an empty spot when you arrive there.
2. [Where do I walk?](codes/2_rapids_seattleParking_graph.ipynb) With this notebook we will calculate the walking distance to parking spots instead of *as the crow flies* using haversine distance.
3. [Where really are the parking spots?](codes/3_rapids_seattleParking_parkingNodes.ipynb) Finally, in this notebook, we will *walk* in a right way, following the roads by adding additional nodes to the road graph.
 
We will work with a dataset published by Seattle Department of Transportation called Paid Parking Occupancy that enumerates every single parking transaction in the city of Seattle. The dataset is published daily generating around 3GB of data monthly but in this workshop we will only be using two months worth of data. Namely, we will be looking at the period of May and June of 2019. 

In order to run these notebooks you will need access to a machine or a compute instance in the cloud that has a GPU from NVIDIA. The GPU needs to be at least a Pascal or above family so any GTX 1000-series like 1080 Ti should work fine. 

We do support RAPIDS in AzureML and you can use our `dask_cloudprovider.AzureMLCluster` tool to quickly instantiate a Dask cluster running RAPIDS on Azure ML. You can install the dask-cloudprovider package using pip: `pip install dask-cloudprovider`. For example how to start the Dask Cluster check [an example here](https://github.com/drabastomek/GTC/blob/master/SJ_2020/workshop/1_Setup/Setup.ipynb). **Note that you will need to provide a your own `subscription_id`, `resource_id` and `workspace_name`.**


The datasets we will use will automatically download when you use the notebooks.