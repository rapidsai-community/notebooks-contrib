# RAPIDS Community Contrib
---
## Table of Contents
* [Intro](#intro)
* [Exploring the Repo](#exploring)
* [Great places to get started](#get_started)
  
---

## Introduction <a name="intro"></a>

Welcome to the community contributed notebooks repo! (formerly known as Notebooks-Extended)

The purpose of this collection is to introduce RAPIDS to new users by providing useful jupyter notebooks as learning aides.  This collection of notebooks are direct community contributions by the RAPIDS team, our Ecosystem Partners, and RAPIDS users like you!

### What do you mean "Community Notebooks" 

These notebooks are for the community.  It means:
1. YOU can contribute workflow examples, tips and tricks, or tutorials for others to use and share!  [We ask that you follow our Testing and PR process.](#contributing)
2. If your notebook is awesome, your notebook can be featured

There are some additional Community Responsibilities, as the RAPIDS team isn't maintaining these notebooks 
- If you write an awesome notebook, please try to keep it maintained.  You'll be mentioned on the issue.
- If you find an issue, don't just file an issue - please attempt to fix it!  
- If a notebook has a problem and/or its last tested RAPIDS release version is in legacy, it may be removed to archives. 

### RAPIDS Showcase Notebooks
These notebooks are built by the RAPIDS team and will be maintained by them.  When we remove the notebooks, it will become community maintained until it hits `the_archive`

### How to Contribute <a name="contributing"></a>

Please see our [guide for contributing to notebooks-contrib](CONTRIBUTING.md).

Once you've followed our guide, please don't forget to [test your notebooks!](TESTING.md) before making a PR.

## Exploring the Repo <a name="exploring"></a>
### Folders

- `getting_started_notebooks` - “how to start using RAPIDS”.  Contains notebooks showing "hello worlds", getting started with RAPIDS libraries, and tutorials around RAPIDS concepts.   
- `community_tutorials_and_guides` - community contributed “how to accomplish your workflows with RAPIDS”.  Contains notebooks showing algorithm and workflow examples, benchmarking tools, and some complete end-to-end (E2E) workflows.
- `community_archive` - This contains notebooks with known issues that have not have not been fixed in 45 days or more.  contains shared notebooks mentioned and used in blogs that showcase RAPIDS workflows and capabilities
- `the_archive` - contains older notebooks from community members as well as notebooks that the RAPIDS team no longer updates, but are useful to the community, such as [`archived_rapids_blog_notebooks`](community_relaunch/the_archive/archived_rapids_blog_notebooks),  [`archived_rapids_event_notebooks`](the_archive/archived_rapids_event_notebooks), and [`competition_notebooks`](the_archive/archived_rapids_competition_notebooks)
- `data` - contains small data samples used for purely functional demonstrations. Some notebooks include cells that download larger datasets from external websites.

### Additional Resources
- [Visit out Youtube Channel](https://www.youtube.com/channel/UCsoi4wfweA3I5FsPgyQnnqw/featured?view_as=subscriber) or see [list of videos](multimedia_links.md) by RAPIDS or our community.  Feel free to contribute your videos and RAPIDS themed playlists as well!
- [Visit our Blogs on Medium](https://medium.com/rapids-ai/) 

## Great places to get started <a name="get_started"></a>

### Topics
Click each topic to expand
<details>
  <summary>RAPIDS Libraries Basics</summary>

##### Getting Started Document
* [Intro to RAPIDS](getting_started_materials/README.md)

##### Teaching Notebooks
* [Intro Notebooks to RAPIDS](getting_started_materials/intro_tutorials_and_guides)- covers cuDF, Dask, cuML and XGBoost.
* [Learn RAPIDS Getting Started Tour (External)](https://github.com/RAPIDSAcademy/rapidsacademy/tree/master/tutorials/datasci/tour)
* [Hello Worlds](getting_started_materials/hello_worlds)
</details>

<details>
  <summary>Cloud Service Providers</summary>

  * [AWS](https://rapids.ai/cloud#aws) 
    * [Single Instance](https://rapids.ai/cloud#AWS-EC2)
    * [Multi GPU Dask](https://rapids.ai/cloud#AWS-Dask)
    * [Kubernetes](https://rapids.ai/cloud#AWS-Kubernetes)
    * [Sagemaker](https://rapids.ai/cloud#AWS-Sagemaker)
      * [Video- Tutorial of RAPIDS on AWS Sagemaker](https://www.youtube.com/watch?v=BtE4d0v6Css)
  * [Azure](https://rapids.ai/cloud#azure)
    * [Single Instance](https://rapids.ai/cloud#AZ-single)
    * [Multi GPU Dask](https://rapids.ai/cloud#AZ-Dask)
    * [Kubernetes](https://rapids.ai/cloud#AZ-Kubernetes)
    * [AzureML Service](https://rapids.ai/cloud#AZ-ML)
      * [Video- Tutorial of RAPIDS on AzureML](https://www.youtube.com/watch?v=aqTmVVFnEwI)
  * [GCP](https://rapids.ai/cloud#googlecloud)
    * [Single Instance](https://rapids.ai/cloud#GC-single)
    * [Multi GPU Dask (Dataproc)](https://rapids.ai/cloud#GC-Dask)
    * [Kubernetes](https://rapids.ai/cloud#GC-Kubernetes)
    * [CloudAI](https://rapids.ai/cloud#GC-AI)

</details>
<details>
  <summary>Multi GPU </summary>

* [Hello Word to Dask](getting_started_materials/hello_worlds/Dask_Hello_World.ipynb)
* [Intro to Dask](getting_started_materials/intro_tutorials_and_guides/03_Introduction_to_Dask.ipynb)
* [Dask using cuDF](getting_started_materials/intro_tutorials_and_guides/04_Introduction_to_Dask_using_cuDF_DataFrames.ipynb)
* [Learn RAPIDS Multi GPU Mini Tour (External)](https://github.com/RAPIDSAcademy/rapidsacademy/tree/master/tutorials/multigpu/minitour)
* NYC taxi on Dataproc
* [Weather Analysis](community_tutorials_and_guides/intermediate_notebooks/examples/weather.ipynb)

</details>
<details>
  <summary>Streaming Data </summary>
  
* [Chinmay Chandak's cuStreamz Gists (External)](https://gist.github.com/chinmaychandak)
* [Using cuStreamz to Accelerate your Kafka Datasource (Blog)](https://medium.com/rapids-ai/the-custreamz-series-the-accelerated-kafka-datasource-4faf0baeb3f6)
* [GPU accelerated Stream processing with RAPIDS (Blog)](https://medium.com/rapids-ai/gpu-accelerated-stream-processing-with-rapids-f2b725696a61)
* [Hello World Streaming Data](getting_started_materials/hello_worlds/hello_streamz.ipynb)

</details>
<details>
  <summary>NLP</summary>
  
* [NLP with Hashing Vectorizer (Blog)](https://medium.com/rapids-ai/gpu-text-processing-now-even-simpler-and-faster-bde7e42c8c8a)
* [Show me the Word Count (Archives)](the_archive/archived_rapids_blog_notebooks/nlp/show_me_the_word_count_gutenberg)

</details>
<details>
  <summary>Graph Analytics </summary>

</details>
<details>
  <summary>GIS/Spatial Analytics </summary>

* [Seismic Facies Analysis (External)](https://github.com/NVIDIA/energy-sdk/tree/master/rapids_seismic_facies)

</details>
<details>
  <summary>Genomics </summary>

  * [Video- GPU accelerated Single Cell Analytics](https://www.youtube.com/watch?v=nYneL_uif3Q) 

</details>
<details>
  <summary>Cybersecurity </summary>

* [RAPIDS CLX](https://docs.rapids.ai/api/clx/stable/)
  * [CLX API Docs](https://docs.rapids.ai/api/clx/stable/api.html)
  * [10 Minutes to CLX](https://docs.rapids.ai/api/clx/stable/10min-clx.html)
  * [Getting Started with CLX and Streamz](https://docs.rapids.ai/api/clx/stable/intro-clx-streamz.html)
* [Learn RAPIDS Cyber Security mini Tour (External)](https://github.com/RAPIDSAcademy/rapidsacademy/tree/master/tutorials/security/tour)
* [Cyber Blog Notebooks (Archives)](the_archive/archived_rapids_blog_notebooks/cyber)

</details>
<details>
  <summary>Past Competitions </summary>

- [RAPIDS.AI KGMON Competition Notebooks](the_archive/archived_competition_notebooks/kaggle)- contains a selection of notebooks that were used in Kaggle competitions.

</details> 

<details>
  <summary>Benchmarks </summary>

* [MultiGPU PageRank Benchmark (Archived)](the_archive/archived_rapids_benchmarks/cugraph)
* [RAPIDS Decomposition (Archived)](the_archive/archived_rapids_benchmarks/rapids_decomposition.ipynb)

</details>
<details>
  <summary>Random Tips and Tricks </summary>

*  [Synthetic 3D End-to-End ML Workflow](community_tutorials_and_guides/synthetic)

</details>

### How-Tos with our Ecosystem Partners 

- [BlazingSQL](#) - these notebooks supplement app.blazingsql.com and provide tutorials for local BlazingSQL workflows.  Make List.   
- cuStreamz
- [LearnRAPIDS](https://www.learnrapids.com/)
- Graphistry

## Additional Information
* The `data` folder also includes the full image set from the [Fashion MNIST dataset](https://github.com/zalandoresearch/fashion-mnist).

* `utils`: contains a set of useful scripts for interacting with RAPIDS Community Notebooks

* For our notebook examples and tutorials found on [github](https://github.com/rapidsai), in each respective repo.

