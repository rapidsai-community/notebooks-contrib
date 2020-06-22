![RAPIDS](img/rapids_logo.png)

<p float="left">
  <img src="./img/nvidia_logo.jpg" width="250" /><br/><br/>
  <img src="./img/microsoft_logo.png" width="500" /><br/><br/>
  <img src="./img/njit_logo.png" width=400" />
</p>

# Accelerating and Expanding End-to-End
Data Science Workflows with DL/ML
Interoperability Using RAPIDS

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
	1. Deep Learning for Tabular Data (use case)
		1. Introduction to data ingest with RAPIDS cuDF
		2. Using the GPU data loader for DL libraries
		3. PyTorch model on tabular data
		4. Post-processing with cuDF
		5. Visualization with cuxfilter
	2. Log Parsing using Neural Networks and a Language Based Model (use case)
		1. Introduction to log parsing and why it matters
		2. Building a BERT model for log parsing (training / fine-tuning)
		3. Inferencing with the BERT model
		4. Wrapping the inference with RAPIDS pre-processing and post-processing
	3. Graph Analysis and Visualization (use case)
		1. Introduction to cuGraph
		2. Presentation of a large dataset suitable for graph analytics
		3. Preprocessing with cuDF and cuGraph
		4. Graph analytics, enriching the dataset/graph
		5. Post-processing with cuDF
		6. Interactive visualization with cuxfilter
		7. Iterating data filtering, graph processing, and visualization in real-time
3. Conclusions (not hands-on) [15 min]
	1. Future Improvements / Roadmap
	2. Any Additional Questions