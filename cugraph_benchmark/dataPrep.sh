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

