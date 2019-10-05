# Colab Notebooks
Ready to experience the speed of RAPIDS (for free) through Google Colab? Each notebook in this folder is ready to launch, modifiable for your experiments, and up to date with the latest RAPIDS AI suite. 
#### Using Colab Notebooks
For easiest use of `colab_notebooks`, use `open in colab` (Google Chrome extension) to automatically open any RAPIDS Jupyter notebook in Google Colaboratory.
- The extension can be found in the Chrome store [here](https://chrome.google.com/webstore/detail/open-in-colab/iogfkhleblhcpcekbiedikdehleodpjo) or on GitHub [here](https://github.com/googlecolab/open_in_colab) 

## Layout
| Notebook Title | Description |
|----------------|----------------|
| cuML (Directory) | folder of `cuML` demos (pca, ridge, sgd, umap, umap_supervised - further desc in folder) like those in `intermediate_notebooks/examples` |
| DBScan_colab_nightlies | Demonstration of `cuML` DBSCAN (clustering) algorithm on mortgage (or random) data in compairson to `sklearn` DBSCAN|
| Louvain_Colab | Implement Louvain method of community detection (greedy hierarchical clustering algorithm) with cuDF; w/ Dask pending |
| Vertex_Similarity_Colab | Use `cuGraph` to compute vertex similarity with both the Jaccard Similarity and the Overlap Coefficient |
| Weighted_Jaccard_Colab | Use `cuGraph` to compute the Weighted Jaccard Similarity metric on `karate-data.csv` |
| black_friday- GTC presentation remix for meetups | black_friday- GTC demo tailored for RAPIDS Meetups @ Galvanize |
| census | ETL & data prep, training Ridge Regression model, and testing predictions - ipums census data |
| coordinate_descent_demo_colab-0.8 | `cuML` vs `sklearn` for Lasso and Elastic Net models |
| cuDatashader | Similarities between `Datashader` and `cuDatashader` for visualizing `nyc_taxi` data |
| knn_demo_colab-0.8 | `cuML` vs `sklearn` K NearestNeighbors (unsupervised) algorithm models on mortgage (or random) data |

## Contributing 
- WE NEED HELP!
  - Find out more [here](https://github.com/rapidsai/notebooks-contrib/blob/master/CONTRIBUTING.md)
