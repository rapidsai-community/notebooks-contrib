#!/bin/bash

echo Downloading ...
wget https://s3.us-east-2.amazonaws.com/rapidsai-data/cugraph/benchmark/twitter-2010.csv.gz

echo Decompressing ...
gunzip twitter-2010.csv.gz 

