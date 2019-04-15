# cuGraph Benchmarking



## Step 1:  Get the Data Sets

Run the data prep script.

```bash
sh ./dataPrep.sh
```



The script is show below

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

find . -name *.mtx -exec mv {} . \;

rm -rf tmp
```



**Test files**

| File Name              | Num of Vertices | Num of Edges |
| ---------------------- | --------------: | -----------: |
| preferentialAttachment |         100,000 |      999,970 |
| caidaRouterLevel       |         192,244 |    1,218,132 |
| coAuthorsDBLP          |         299,067 |    1,955,352 |
| dblp-2010              |         326,186 |    1,615,400 |
| citationCiteseer       |         268,495 |    2,313,294 |
| coPapersDBLP           |         540,486 |   30,491,458 |
| coPapersCiteseer       |         434,102 |   32,073,440 |
| as-Skitter             |       1,696,415 |   22,190,596 |



## Benchmarks

1. Louvain - louvain benchmark notebook
2. PageRank