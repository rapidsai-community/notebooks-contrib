#!/bin/bash

echo Downloading ...
wget https://data.rapids.ai/cugraph/benchmark/twitter-2010.csv.gz

echo Decompressing ...
gunzip twitter-2010.csv.gz
