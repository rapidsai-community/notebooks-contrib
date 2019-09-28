# RAPIDS and Cybersecurity: A Network Use Case
## Purpose
Computer networks generate massive amounts of heterogeneous data as users interact with other users, computers, and services. These interactions can be modeled as large, heterogeneous property graphs, with multidimensional characteristics of the communication embedded on an edge connecting nodes. Current techniques to identify subgraph evolution over time and extract anomalies require lengthy compute times or necessitate significant pre-filtering of the graph. In this tutorial, we showcase an approach to flagging anomalous network communications in a large graph using a combination of structural graph features and graph analytics, running end-to-end in RAPIDS.

## Authors
 - Rachel Allen, PhD (NVIDIA)
 - Haekyu Park (Georgia Tech, NVIDIA)
 - Bartley Richardson, PhD (NVIDIA)

## Datasets
#### [CDS IDS 2018 dataset](https://www.unb.ca/cic/datasets/ids-2018.html)
* From the [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/)
* Large collection of source files, including flow data
* Multiple types of attacks, with ground truth (labeled) data

##### Using the Pre-Processed Dataset
We have pre-processed the PCAP dataset to create biflows using the process described below. You can [download it here](https://rapidsai-data.s3.us-east-2.amazonaws.com/cyber/kdd2019/Friday-02-03-2018-biflows.tar.gz). In addition, the Jupyter notebook will also download and extract the pre-processed data.

##### Downloading the Raw Data and Proecssing Yourself
If you would prefer to process the raw data yourself, follow the directions below. These directions are also available at the bottom of [this page](https://www.unb.ca/cic/datasets/ids-2018.html).

1. Install the [AWS CLI](https://aws.amazon.com/cli/), available on Mac, Windows and Linux.
2. Run: `aws s3 sync --no-sign-request --region <your-region> "s3://cse-cic-ids2018/" dest-dir`
    
Where `your-region` is your region from the [AWS regions list](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/using-regions-availability-zones.html) and `dest-dir` is the name of the desired destination folder in your machine. For the purposes of this tutorial, we asume the data is in the same directory as the notebook under the `data/` directory. You may change this by pointing the `DATA_LOCATION` directory in the notebook to wherever the data resides.

## Files
### Cybersecurity_KDD.ipynb
* Complete KDD hands-on tutorial notebook for the cybersecurity application
* Using the [RAPIDS](https://rapids.ai) suite of open-source software, we demonstrate how to:
  1. Triage and perform data exploration,
  2. Model network data as a graph,
  3. Perform graph analytics on the graph representation of the cyber network data, and
  4. Prepare the results in a way that is suitable for visualization.

## Acknowledgments
We would like to thank the [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/) for the data used in this tutorial. A complete description of the dataset used is [available online](https://registry.opendata.aws/cse-cic-ids2018/). In addition, the paper associated with this dataset is:

> Iman Sharafaldin, Arash Habibi Lashkari, and Ali A. Ghorbani, “Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization”, 4th International Conference on Information Systems Security and Privacy (ICISSP), Portugal, January 2018

We would also like to acknowledge the contributions of Eli Fajardo (NVIDIA), Brad Rees, PhD (NVIDIA), and the [RAPIDS](https://rapids.ai) engineering team.