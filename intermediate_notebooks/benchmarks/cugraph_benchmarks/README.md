# cuGraph Benchmarking

This folder contains a collection of graph algorithm benchmarking notebooks.  Each notebook will compare one cuGraph algorithm against the equivalent NetworkX version.  In some cases, additional popular implementations are also tested.

Before any benchmarking can be done, it is important to fir download the test data sets.


## Getting the Data Sets

Run the data prep script.

```bash
sh ./dataPrep.sh
```

## Benchmarks

1. Louvain
2. PageRank
3. BSF
4. SSSP




#### The data prep script  
By default, each files would be created in its own directory.  The goal here is to have all the MTX files in a single directory.


```bash
#!/bin/bash

mkdir data
cd data
mkdir tmp
cd tmp

wget https://sparse.tamu.edu/MM/DIMACS10/preferentialAttachment.tar.gz
wget https://sparse.tamu.edu/MM/DIMACS10/caidaRouterLevel.tar.gz
wget https://sparse.tamu.edu/MM/DIMACS10/coAuthorsDBLP.tar.gz
wget https://sparse.tamu.edu/MM/LAW/dblp-2010.tar.gz
wget https://sparse.tamu.edu/MM/DIMACS10/citationCiteseer.tar.gz
wget https://sparse.tamu.edu/MM/DIMACS10/coPapersDBLP.tar.gz
wget https://sparse.tamu.edu/MM/DIMACS10/coPapersCiteseer.tar.gz
wget https://sparse.tamu.edu/MM/SNAP/as-Skitter.tar.gz

tar xvzf preferentialAttachment.tar.gz
tar xvzf caidaRouterLevel.tar.gz
tar xvzf coAuthorsDBLP.tar.gz
tar xvzf dblp-2010.tar.gz
tar xvzf citationCiteseer.tar.gz
tar xvzf coPapersDBLP.tar.gz
tar xvzf coPapersCiteseer.tar.gz
tar xvzf as-Skitter.tar.gz

cd ..

find ./tmp -name *.mtx -exec mv {} . \;

rm -rf tmp
```




** About the Test files**

| File Name              | Num of Vertices | Num of Edges | Format |  Graph Type               | Symmetric   |
| ---------------------- | --------------: | -----------: |--------|---------------------------|-------------|
| preferentialAttachment |         100,000 |      999,970 | MTX    | Random Undirected Graph   | Yes         | 
| caidaRouterLevel       |         192,244 |    1,218,132 | MTX    | Undirected Graph          | Yes         | 
| coAuthorsDBLP          |         299,067 |    1,955,352  |MTX    | Undirected Graph          | Yes         | 
| dblp-2010              |         326,186 |    1,615,400 | MTX    | Undirected Graph          | Yes         | 
| citationCiteseer       |         268,495 |    2,313,294 | MTX    | Undirected Graph          | Yes         | 
| coPapersDBLP           |         540,486 |   30,491,458 | MTX    | Undirected Graph          | Yes         | 
| coPapersCiteseer       |         434,102 |   32,073,440 | MTX    | Undirected Graph          | Yes         | 
| as-Skitter             |       1,696,415 |   22,190,596 | MTX    | Undirected Graph          | Yes         | 



### Dataset Acknowlegments

The dataset are downloaded from the Texas A&M SuiteSparse Matrix Collection

```
The SuiteSparse Matrix Collection (formerly known as the University of Florida Sparse Matrix Collection), is a large and actively growing set of sparse matrices that arise in real applications. 
...
The Collection is hosted here, and also mirrored at the University of Florida at www.cise.ufl.edu/research/sparse/matrices. The Collection is maintained by Tim Davis, Texas A&M University (email: davis@tamu.edu), Yifan Hu, Yahoo! Labs, and Scott Kolodziej, Texas A&M University. 
```

| File Name              |  Author        |
| ---------------------- |----------------|
| preferentialAttachment | H. Meyerhenke  |
| caidaRouterLevel       | Unknown        |
| coAuthorsDBLP          | R. Geisberger, P. Sanders, and D. Schultes |
| dblp-2010              | Laboratory for Web Algorithmics (LAW), |
| citationCiteseer       | R. Geisberger, P. Sanders, and D. Schultes  |
| coPapersDBLP           | R. Geisberger, P. Sanders, and D. Schultes  |
| coPapersCiteseer       | R. Geisberger, P. Sanders, and D. Schultes |
| as-Skitter             | J. Leskovec, J. Kleinberg and C. Faloutsos |




#### preferentialAttachment
DIMACS10 set: clustering/preferentialAttachment                    
source: http://www.cc.gatech.edu/dimacs10/archive/clustering.shtml 
                                                                   
This graph has been generated following a preferential attachment  
process (see Barabási and Albert, "Emergence of scaling in random  
networks", Science, 1999). Starting with a clique of five vertices,
the vertices are successively added to the graph. Each new vertex  
chooses exactly five neighbors among the existing vertices, such   
that the probability of choosing a particular vertex is            
proportional to its degree. In our implementation, a vertex can    
choose a neighbour only once, such that the resulting random graph 
is guaranteed to be simple.



#### aidaRouterLevel
coAuthorsDBLP.tar.gz


wget https://sparse.tamu.edu/MM/LAW/dblp-2010.tar.gz
citationCiteseer.tar.gz
coPapersDBLP.tar.gz
coPapersCiteseer.tar.gz
wget https://sparse.tamu.edu/MM/SNAP/as-Skitter.tar.gz