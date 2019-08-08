# cuGraph intermediate example

## Step 1:  Get the Data Set

This step involves downloading a `gz` file of 6GB. Once decompressed the file is 26GB.

Option 1 : Run the script
```bash
sh ./get_data.sh
```

Option 2 : manually run
```bash
wget https://s3.us-east-2.amazonaws.com/rapidsai-data/cugraph/benchmark/twitter.csv.gz
gunzip twitter.csv.gz  
```


**Graph info**

| File Name    | Num of Vertices | Num of Edges    | File size  |
| -------------| --------------: | -------------:  | ---------: |
| twitter      |      41,652,230 | 1,468,365,182   |       26GB | 


## Step 2:  Open the notebook
```bash
ipython notebook multi_gpu_pagerank.ipynb
```
