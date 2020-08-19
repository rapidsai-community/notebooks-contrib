![RAPIDS](img/rapids_logo.png)

<p float="left">
  <img src="./img/nvidia_logo.jpg" width="250" /><br/><br/>
  <img src="./img/microsoft_logo.png" width="500" /><br/><br/>
  <img src="./img/njit_logo.png" width=400" />
</p>

# Accelerating and Expanding End-to-End Data Science Workflows with DL/ML Interoperability Using RAPIDS

## KDD 2020 Tutorial
The lines between data science (DS), machine learning (ML), deep learning (DL), and data mining continue to be blurred and removed. This is great as it ushers in vast amounts of capabilities, but it brings increased complexity and a vast number of tools/techniques. It’s not uncommon for DL engineers to use one set of tools for data extraction/cleaning and then pivot to another library for training their models. After training and inference, it’s common to then move data yet again by another set of tools for post-processing. The ​RAPIDS​ suite of open source libraries not only provides a method to execute and accelerate these tasks using GPUs with familiar APIs, but it also provides interoperability with the broader open source community and DL tools while removing unnecessary serializations that slow down workflows. GPUs provide massive parallelization that DL has leveraged for some time, and RAPIDS provides the missing pieces that extend this computing power to more traditional yet important DS and ML tasks (e.g., ETL, modeling). Complete pipelines can be built that encompass everything, including ETL, feature engineering, ML/DL modeling, inference, and visualization, all while removing typical serialization costs and affording seamless interoperability between libraries. All

experiments using RAPIDS can effortlessly be scheduled, logged and reviewed using existing public cloud options.
Join our engineers and data scientists as they walk through a collection of DS and ML/DL engineering problems that show how RAPIDS running on Azure ML can be used for end-to-end, entirely GPU pipelines. This tutorial includes specifics on how to use RAPIDS for feature engineering, interoperability with common ML/DL packages, and creating GPU native visualizations using ​cuxfilter​. The use cases presented here give attendees a hands-on approach to using RAPIDS components as part of a larger workflow, seamlessly integrating with other libraries (e.g., PyTorch) and visualization packages.

## Agenda:
1. Introduction (not hands-on) [20 min]
	1. Speaker Introductions
	2. Getting Connected to the VM Instances
	3. Why RAPIDS, and How RAPIDS Connects to the Larger Ecosystem
2. Tutorial (hands-on) [2 hours 20 min]
	1. [New York Taxi Data](./notebooks/Taxi/NYCTax.ipynb)
      	1. In troduction to RAPIDS via an analysis of the New York City taxi data set
      	2. Key Libraries: cuDF, cuML, cuGraph, cuXfilter​
	2. [Deep Learning for Tabular Data](nvtabular/rossmann-store-sales-example.ipynb)
		1. Perform store sales prediction using tabular deep learning​
		2. Key Libraries: NVTabular, cuDF, TensorFlow​
	3. [Single-Cell RNA Sequencing Analysis](notebooks/Lungs/hlca_lung_gpu_analysis.ipynb)
		1. Analyzing gene expression from a population of cells from a human lung​
		2. Key Libraries: scanpy, cuDF, cuML, cuGraph​
	4. Where to Park​
		1. Analyzing Seattle Parking data and determining the best parking spot within a walkable distance from Space Needle​
		2. Key Libraries: cuSpatial, cuDF, cuGraph​
    5. CyBERT
       1. Cyber Log Parsing using Neural Networks and Language Based Model​
       2. Key Libraries: CLX, cuDF, PyTorch​
3. Conclusions (not hands-on) [15 min]
	5. Future Improvements / Roadmap
	6. Any Additional Questions



## Presenters

[Presenters](Presenters.md)