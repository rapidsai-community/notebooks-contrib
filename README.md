# RAPIDS Extended Notebooks
## Intro
Welcome to the extended notebooks repo!

The purpose of this collection of notebooks is to help users understand what RAPIDS has to offer, learn how, why, and when including it in data science pipelines makes sense, and help get started using RAPIDS libraries by example. 

Many of these notebooks use additional PyData ecosystem packages, and include code for downloading datasets, thus they require network connectivity. If running on a system with no network access, please use the [core notebooks repo](https://github.com/rapidsai/notebooks).

## Installation

Please use the [BUILD.md](https://github.com/rapidsai/notebooks-extended/tree/master) to check the pre-requisite packages and installation steps.

## Contributing

Please see our [guide for contributing to notebooks-extended](https://github.com/rapidsai/notebooks-extended/blob/master/CONTRIBUTING.md).

## Exploring the Repo

Notebooks live under two subfolders:
- `beginner` - contains notebooks showing “how [to master] RAPIDS”:
    - `basics` - to help you quickly learn the basic RAPIDS APIs.  It contains these notebooks
        - Dask Hello World
        - Getting Started with cuDF
    - `tutorial` - which is a basic tutorial on all the libraries present in RAPIDS.
    
- `advanced` - these notebooks demonstrate "why RAPIDS" by directly comparing compute time between single and multi threaded CPU implementations vs GPU (RAPIDS library) implementations. Of note here is the similarity of RAPIDS APIs to common PyData ecosystem packages like Pandas and scikit-learn. This folder includes the following folders: 
    - E2E: The E2E folder contains end to end notebooks which use the RAPIDS libraries
    - benchmarks: The benchmarks folder contains notebooks which are used to benchmark the cuGraph and cuML algorithms
    - blog notebooks: contains the notebooks mentioned and used in blogs written by RAPIDS
    - conference notebooks
    - examples: contains high level examples for users familiar with RAPIDS libraries

`/data` contains small data samples used for purely functional demonstrations. Some notebooks include cells that download larger datasets from external websites.

The `/data` folder is also symlinked into `/rapids/notebooks/extended/data` so you can browse it from JupyterLab's UI.

# RAPIDS Notebooks-extended

## Beginner Notebook
| Folder    | Notebook Title         | Description                                                                                                                                                                                                                   |
|-----------|------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| basics   | Dask_Hello_World           | This notebook shows how to quickly setup Dask and run a "Hello World" example.                                                                                                                                       |
| basics   | Getting_Started_with_cuDF  | This notebook shows how to get started with GPU DataFrames using cuDF in RAPIDS.                                                                                                                                      |
| tutorial   | 01_Introduction_to_RAPIDS  | This notebook shows at a high level what each of the packages in the RAPIDS are as well as what they do.                                                                                                                                      |
## Advanced Notebooks:
 
| Folder    | Notebook Title         | Description                                                                                                                                                                                                                   |
|-----------|------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| E2E: mortgage      | mortgage_e2e            | This notebook showcases density-based spatial clustering of applications with noise (dbscan) algorithm using the `fit` and `predict` functions                                                                              |
| E2E: mortgage      | mortgage_e2e_deep_learning               | This notebook showcases k-nearest neighbors (knn) algorithm using the `fit` and `kneighbors` functions                                                                                                                          |
| E2E      | rapids_ml_workflow_demo | This notebook includes code example for linear regression algorithm and it showcases the `fit` and `predict` functions.                                                                                                                                             |
| benchmarks      | cuml_benchmarks  | This notebook includes code examples of ridge regression and it showcases the `fit` and `predict` functions.                                                                                                                                          |
| benchmarks: cugraph_benchmarks      | louvain_benchmark.ipynb   | This notebook includes code examples of lasso and elastic net models. These models are placed together so a comparison between the two can also be made in addition to their sklearn equivalent.                                                                                                                                                                |
|  benchmarks: cugraph_benchmarks    |  pagerank_benchmark.ipynb             | This notebook showcases principal component analysis (PCA) algorithm where the model can be used for prediction (using `fit_transform`) as well as converting the transformed data into the original dataset (using `inverse_transform`).                                                                                                                |

### Blog notebooks:
#### Cyber notebooks:
| Folder    | Notebook Title         | Description                                                                                                                                                                                                                   |
|-----------|------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|flow_classification     | flow_classification_rapids              | This notebook showcases truncated singular value decomposition (tsvd) algorithm which like PCA performs both prediction and transformation of the converted dataset into the original data using `fit_transform` and `inverse_transform` functions respectively                                                                                                     |
| network_mapping      | lanl_network_mapping_using_rapids               | The stochastic gradient descent algorithm is demostrated in the notebook using `fit` and `predict` functions                                                                        |
| raw_data_generator      | run_raw_data_generator              | The uniform manifold approximation & projection algorithm is compared with the original author's equivalent non-GPU \Python implementation using `fit` and `transform` functions                       |
| cuML      | umap_demo_graphed      | Demonstration of cuML uniform manifold approximation & projection algorithm's supervised approach against mortgage dataset and comparison of results against the original author's equivalent non-GPU \Python implementation. |
| cuML      | umap_demo_supervised   | Demostration of UMAP supervised training.  Uses a set of labels to perform supervised dimensionality reduction. UMAP can also be trained on datasets with incomplete labels, by using a label of "-1" for unlabeled samples. |

#### Databrix Notebooks
| Folder    | Notebook Title         | Description                                                                                                                                                                                                                   |
|-----------|------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| databrix   | RAPIDS_PCA_demo_avro_read                | Demonstration of using cuGraph to identify clusters in a test graph using the Louvain algorithm                                                                                                                               |
| databrix   | spark_rapids_pca_demo      | Demonstration of using cuGraph to compute vertex similarity using both the Jaccard Similarity and the Overlap Coefficient.                                                                          

#### Regression Notebooks
| Folder    | Notebook Title         | Description                                                                                                                                                                                                                   |
|-----------|------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| regression   | regression_blog_notebook.ipynb       | Demonstration of using cuGraph to compute the Weighted Jaccard Similarity metric on our training dataset.                                                                                                                     |

### Example Notebooks
| Folder    | Notebook Title         | Description                                                                                                                                                                                                                   |
|-----------|------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| examples   | DBSCAN_Demo_FULL               | Demonstrate of using the renumbering features to assigned new vertex IDs to the test graph.  This is useful for when the data sets is  non-contiguous or not integer values                                                   |
| examples   | Dask_with_cuDF_and_XGBoost                    | Demonstration of using cuGraph to computer the Bredth First Search space from a given vertex to all other in our training graph                                                                                               |
| examples   | Dask_with_cuDF_and_XGBoost_Disk                   | Demonstration of using cuGraph to computer the The Shortest Path from a given vertex to all other in our training graph                                                                                                       |
| examples   | One_Hot_Encoding    | Demonstration of using cuGraph to identify clusters in a test graph using Spectral Clustering using both the (A) Balance Cut and (B) the Modularity Maximization quality metrics                                              |
| examples   | PCA_Demo_Full               | Demonstration of using both NetworkX and cuGraph to compute the PageRank of each vertex in our test dataset                                                                                                                   |
| examples   | linear_regression_demo.ipynb     | Demonstration of using both NetworkX and cuGraph  to compute the the number of Triangles in our test dataset                                                                                                                  |
| examples   | ridge_regression_demo.ipynb     | Demonstration of using both NetworkX and cuGraph  to compute the the number of Triangles in our test dataset                                                                                                                  |
| examples   | umap_demo.ipynb     | Demonstration of using both NetworkX and cuGraph  to compute the the number of Triangles in our test dataset                                                                                             

## Additional Information
* The `cuml` folder also includes a small subset of the Mortgage Dataset used in the notebooks and the full image set from the Fashion MNIST dataset.

* `utils`: contains a set of useful scripts for interacting with RAPIDS

* For additional, community driven notebooks, which will include our blogs, tutorials, workflows, and more intricate examples, please see the [Notebooks Extended Repo](https://github.com/rapidsai/notebooks-extended)
