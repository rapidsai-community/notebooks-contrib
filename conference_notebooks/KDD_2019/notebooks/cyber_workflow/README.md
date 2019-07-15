# Cyber Workflow 3


## Purpose (from proposal)
Computer networks generate massive amounts of heterogeneous data as users interact with other users, computers, and services. These interactions can be modeled as large, heterogeneous property graphs, with multidimensional characteristics of the communication embedded on an edge connecting nodes. Current techniques to identify subgraph evolution over time and extract anomalies require lengthy compute times or necessitate significant pre-filtering of the graph. In this tutorial, we showcase an approach to flagging anomalous network communications in a large graph using a combination of structural graph features and graph analytics, running end-to-end in RAPIDS.

## Datasets
#### [UNSW-NB15](https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/)
* Large collection of source files (including PCAP and Bro/Zeek)
* Multiple types of attacks, malware, etc. represented in the data
* Ground truth available
* Pre-made training/testing sets 
* Temporal data (spread over two days, multiple hours)

## Files
### 01_DataTriage.ipynb
* Investigate the data
* Enrich Bro `conn` logs with ground truth data
* Visualize

### 02_Viz.ipynb
* Visualization of `src_ip`-->`src_port`-->`dst_port`-->`dst_ip` using parallel coordinates
* Feature engineering to support the above

### Cybersecurity_KDD_v1.ipynb
* v1 of the eventual workshop notebook for cybersecurity applications
* Includes viz components as well as feature engineering
* Items marked as **[TODO]** will be created in future versions