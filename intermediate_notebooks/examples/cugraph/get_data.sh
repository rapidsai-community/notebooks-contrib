#!/bin/bash

echo Downloading ...
wget https://s3.us-east-2.amazonaws.com/rapidsai-data/cugraph/benchmark/twitter.csv.gz

echo Decompressing ...
gunzip twitter.csv.gz  

