# Classification of IoT Flow Data using RAPIDS and XGBoost

## Goals
In this notebook, we demonstrate how to load netflow data into cuDF and create a multiclass classification model using XGBoost. The goals are to:

* Learn the basics of cyber network data with respect to consumer IoT devices,
* Load network data into a cuDF,
* Explore network data and features using cuDF,
* Use XGBoost to build a classification model, and
* Evaluate the model's performance.

## The Data
The data for this notebook can be found in `/data/unswiot/`. Unpack the files and place them in `tutorials/cyber/flow_classification/data/input/unswiot/`. Data used in this notebook is provided by the [University of New South Wales IoT project](https://iotanalytics.unsw.edu.au/traffic_analysis.html). In addition, the raw PCAP data was processed into flow data using [Zeek](https://www.zeek.org).

## Acknowledgments
We would like to thank the [University of New South Wales](https://www.unsw.edu.au) for the data used in this notebook.