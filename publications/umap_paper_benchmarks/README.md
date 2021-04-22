Datasets are not included in this repository and need to be downloaded separately. 

The necessary dependencies for reproducing the benchmarks have been captured in `conda` environment yaml files. 

To install dependencies for cuml and UMAP-learn benchmarks: 
```
conda env create --name cuml_umap_benchmarks -f conda/umap_paper_cuml_cuda10.2.yml
```

To install dependencies for GPUMAP benchmarks: 
```
conda env create --name gpumap_benchmarks -f conda/umap_paper_gpumap_cuda10.0.yml
```

You can run the notebooks using jupyter lab:
```
conda activate <environment_name>
python -m ipykernel install --user
jupyter lab
```

# Datasets

- PEN Digits - uses sklearn.datasets.load_digits
- GoogleNews Word2Vec - Downloaded from https://code.google.com/archive/p/word2vec/ and loaded using Gensim library
- Fashion MNIST - Downloaded from https://github.com/zalandoresearch/fashion-mnist
- CIFAR-100 - Downloaded from https://www.cs.toronto.edu/~kriz/cifar.html
- Shuttle - Downloaded from https://archive.ics.uci.edu/ml/datasets/Statlog+(Shuttle)
- MNIST - Uses datasets submodule to download and load
- TASIC2018 - Data from : https://portal.brain-map.org/atlases-and-data/rnaseq (see dedicated notebook)
- scRNA - Dataset downloaded from https://cells.ucsc.edu/
- COIL-20 - Uses datasets submodule to download and load


