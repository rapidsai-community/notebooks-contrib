# RAPIDS Community Contrib
---
## Table of Contents
* [Intro](#intro)
* [Exploring the Repo](#exploring)
* [Great places to get started](#get_started)
* [Additional Resources](#more)
  
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

## Great places to get started <a name="get_started"></a>

### Topics
Click each topic to expand
<details>
  <summary>RAPIDS Libraries Basics</summary>

#### Getting Started Readings
* [RAPIDS Release Deck](https://docs.rapids.ai/overview/latest.pdf)
* [Intro to RAPIDS](getting_started_materials/README.md)
  
#### Teaching Notebooks
* [Intro Notebooks to RAPIDS](getting_started_materials/intro_tutorials_and_guides)- covers cuDF, Dask, cuML and XGBoost.
* [Learn RAPIDS Getting Started Tour (External)](https://github.com/RAPIDSAcademy/rapidsacademy/tree/master/tutorials/datasci/tour)
* [Hello Worlds](getting_started_materials/hello_worlds)
  
#### Official Cheat Sheets
* [cuDF Cheat Sheet (PDF Download)](https://forums.developer.nvidia.com/uploads/short-url/mIndAvHNud3UXeWwC7Ore3d021D.pdf)
* [BlazingSQL Cheat Sheet (PDF Download)](https://forums.developer.nvidia.com/uploads/short-url/v0Wt2kUisxHUwr9fJSD6yA1J2bP.pdf)
* [cuGraph Cheat Sheet (PDF Download)](https://forums.developer.nvidia.com/uploads/short-url/kIbMG6LZjFfLFibbyqvVl2XcSbB.pdf)
* [RAPIDS-Dask Cheat Sheet (PDF Download)](https://forums.developer.nvidia.com/uploads/short-url/xiN07MC8FSHsXS6lekxSaY1CWs4.pdf)
* [CLX and cyBert Cheat Sheet (PDF Download)](https://forums.developer.nvidia.com/uploads/short-url/edzS5WizVTYZMWRtTl3AqHI5AL4.pdf)
* [cuSignal Cheat Sheet (PDF Download)](https://forums.developer.nvidia.com/uploads/short-url/hkh6vQ2rzl6mAHL8Vt0CYhctark.pdf)
</details>

<details>
  <summary>Cloud Service Providers</summary>

  #### [AWS](https://rapids.ai/cloud#aws) 
  * [Single Instance](https://rapids.ai/cloud#AWS-EC2)
  * [Multi GPU Dask](https://rapids.ai/cloud#AWS-Dask)
    * [Getting started with RAPIDS on AWS ECS using Dask Cloud Provider](https://medium.com/rapids-ai/getting-started-with-rapids-on-aws-ecs-using-dask-cloud-provider-b1adfdbc9c6e)
  * [Kubernetes](https://rapids.ai/cloud#AWS-Kubernetes)
  * [Sagemaker](https://rapids.ai/cloud#AWS-Sagemaker)
    * [Video- Tutorial of RAPIDS on AWS Sagemaker](https://www.youtube.com/watch?v=BtE4d0v6Css)
  #### [Azure](https://rapids.ai/cloud#azure)
  * [Single Instance](https://rapids.ai/cloud#AZ-single)
  * [Multi GPU Dask](https://rapids.ai/cloud#AZ-Dask)
  * [Kubernetes](https://rapids.ai/cloud#AZ-Kubernetes)
  * [AzureML Service](https://rapids.ai/cloud#AZ-ML)
    * [Video- Tutorial of RAPIDS on AzureML](https://www.youtube.com/watch?v=aqTmVVFnEwI)
  #### [GCP](https://rapids.ai/cloud#googlecloud)
  * [Single Instance](https://rapids.ai/cloud#GC-single)
  * [Multi GPU Dask (Dataproc)](https://rapids.ai/cloud#GC-Dask)
    * [Bursting Data Science Workloads to GPUs on Google Cloud Platform with Dask Cloud Provider (Blog with Code snippets)](https://medium.com/rapids-ai/bursting-data-science-workloads-to-gpus-on-google-cloud-platform-with-dask-cloud-provider-685be1eff204)
  * [Kubernetes](https://rapids.ai/cloud#GC-Kubernetes)
  * [CloudAI](https://rapids.ai/cloud#GC-AI)
  #### [IBM]()
  * Single Instance
    * [Step by Step - Tutorial of RAPIDS on Virtual Server Instance](https://medium.com/@ahmed_82744/deploy-rapids-on-ibm-cloud-virtual-server-for-vpc-ce3e4b3ede1c)- by  [Muhammad Arif](https://www.linkedin.com/in/arifnafees/) in collabaration with [Syed Afzal Ali]()
  * Kubernetes
    * [Step by Step - Tutorial of RAPIDS on Kubernetes Service](https://medium.com/@ahmed_82744/deploy-rapids-on-ibm-cloud-kubernetes-service-920de68dc6c4)- by  [Muhammad Arif](https://www.linkedin.com/in/arifnafees/) in collabaration with [Syed Afzal Ali]()
  


</details>
<details>
  <summary>Multi GPU </summary>

  #### Getting Started 
* [Hello Word to Dask](getting_started_materials/hello_worlds/Dask_Hello_World.ipynb)
* [Intro to Dask](getting_started_materials/intro_tutorials_and_guides/03_Introduction_to_Dask.ipynb)
* [Dask using cuDF](getting_started_materials/intro_tutorials_and_guides/04_Introduction_to_Dask_using_cuDF_DataFrames.ipynb)
* [10 Minutes to Dask cuDF]()
* [Learn RAPIDS Multi GPU Mini Tour (External)](https://github.com/RAPIDSAcademy/rapidsacademy/tree/master/tutorials/multigpu/minitour)
#### Example Workflows
  
* [NYC Taxi on Dataproc (or Local)](https://github.com/rapidsai-community/notebooks-contrib/blob/main/community_tutorials_and_guides/taxi/NYCTaxi-E2E.ipynb)
* [Weather Analysis](community_tutorials_and_guides/intermediate_notebooks/examples/weather.ipynb)
* Dask Mortgage Analysis
* Performance Mortgage Analysis
* [State of the art NLP at scale with RAPIDS, HuggingFace and Dask (Blog and Code)](https://medium.com/rapids-ai/state-of-the-art-nlp-at-scale-with-rapids-huggingface-and-dask-a885c19ce87b)
* [LearnRAPIDS Multi-GPU Mini Tour (External)](https://github.com/RAPIDSAcademy/rapidsacademy/tree/master/tutorials/multigpu/minitour)
#### Dask Tricks
  
* [Monitoring Dask RAPIDS with Prometheus and Grafana (Blog with Code)](https://medium.com/rapids-ai/monitoring-dask-rapids-with-prometheus-grafana-96eaf6b8f3a0)
* [Scheduling & Optimizing RAPIDS Workflows with Dask and Prefect (Blog and Code)](https://medium.com/rapids-ai/scheduling-optimizing-rapids-workflows-with-dask-and-prefect-6fc26d011bf)
* [Filtered Reading with RAPIDS & Dask to Optimize ETL (Blog and Code)](https://medium.com/rapids-ai/filtered-reading-with-rapids-dask-to-optimize-etl-5f1624f4be55)

</details>
<details>
  <summary>RAPIDS and Deep Learning </summary>
  
* [Official RAPIDSAI Deep Learning Repo](https://github.com/rapidsai/deeplearning)
* [GPU Hackthons RAPIDS + Deep Learning Crash Course](https://github.com/gpuhackathons-org/gpubootcamp/blob/master/ai/RAPIDS/)
* [deeplearningwizard.com's Wizard Tutorial](https://github.com/ritchieng/deep-learning-wizard/) (External, uses Google Colab)
  
</details>

<details>
  <summary>Data Visualizations with RAPIDS </summary>
  
#### Offical RAPIDS Demos
* [Intro to cuXFilter](https://github.com/rapidsai-community/showcase/blob/main/team_contributions/cuxfilter-tutorial/cuxfilter_tutorial.ipynb)
* [Spatial Analytics Viz](https://github.com/exactlyallan/Spatial-Analytics-Viz/tree/main)
 
#### Tutorials
* [Visual EDA on NYC Taxi Spatial Analytics (As Shown in PyDataDC Meetup 11/2020)](https://github.com/taureandyernv/rapidsai_visual_eda)
* [RAPIDS + Plot.ly Dask Tutorial (As shown in PyDataTT on 05/2021)](https://github.com/taureandyernv/rapids-plotly-webapps/tree/main).
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

  * [Clara Parabricks Single Cell Analytics Repo](https://github.com/clara-parabricks/rapids-single-cell-examples) - [Notebooks](https://github.com/clara-parabricks/rapids-single-cell-examples/tree/master/notebooks)
  * [RAPIDS Single Cell Analytics with updated scanpy wrappers](https://github.com/Intron7/rapids_singlecell) - by [Severin Dicks](https://github.com/Intron7) ([Institute of Medical Bioinformatics and Systems Medicine](https://www.uniklinik-freiburg.de/institut-fuer-medizinische-bioinformatik-und-systemmedizin/englisch/en.html), Freiburg)
  * [Video - GPU accelerated Single Cell Analytics](https://www.youtube.com/watch?v=nYneL_uif3Q) 
  * [Video - Accelerate and scale genomic analysis with open source analytics](https://cloudonair.withgoogle.com/events/genomic-analysis) (Free Google registration required)

</details>
<details>
  <summary>Cybersecurity </summary>

* [RAPIDS CLX](https://docs.rapids.ai/api/clx/stable/)
  * [CLX API Docs](https://docs.rapids.ai/api/clx/stable/api.html)
  * [10 Minutes to CLX](https://docs.rapids.ai/api/clx/stable/10min-clx.html)
  * [Getting Started with CLX and Streamz](https://docs.rapids.ai/api/clx/stable/intro-clx-streamz.html)
* [Learn RAPIDS Cyber Security Mini Tour (External)](https://github.com/RAPIDSAcademy/rapidsacademy/tree/master/tutorials/security/tour)
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

* [Synthetic 3D End-to-End ML Workflow](community_tutorials_and_guides/synthetic)
* [Reading Larger than Memory CSVs with RAPIDS and Dask (Blog)](https://medium.com/rapids-ai/reading-larger-than-memory-csvs-with-rapids-and-dask-e6e27dfa6c0f)

</details>

### How-Tos with our Ecosystem Partners 
<details>  
  <summary>BlazingSQL</summary>

* [Main Website](https://blazingsql.com/)
* [Docs](https://docs.blazingsql.com/)
* [Intro Notebooks](https://github.com/BlazingDB/Welcome_to_BlazingSQL_Notebooks/tree/master/intro_notebooks)
* [Welcome to Blazing's RAPIDS Cheatsheets](https://github.com/BlazingDB/Welcome_to_BlazingSQL_Notebooks/tree/master/cheatsheets)
* [Webinar Notebooks](https://github.com/BlazingDB/Welcome_to_BlazingSQL_Notebooks/tree/master/webinars)
  
</details> 

<details>
  <summary>cuStreamz</summary>
</details> 
<details>
  <summary>LearnRAPIDS</summary>

* [Main Website](https://www.learnrapids.com/)
* [Tutorial Github Repo](https://github.com/RAPIDSAcademy/rapidsacademy/tree/master/tutorials)
  
</details>
<details>
  <summary>Graphistry</summary>

* [Graph viz/connectors/transforms for cuGraph/cuDF with Demos](https://github.com/graphistry/pygraphistry) - Demos in /demos
* [RAPIDS dashboarding with Graphistry with Demos](https://github.com/graphistry/graph-app-kit) - Various demos in /python/views
* [Graphistry Hub](https://hub.graphistry.com/) - Includes no-code file uploader + free API keys
  
</details>

## Additional Resources <a name="more"></a>
Beyond our [Official RAPIDS Docs](https://docs.rapids.ai/api), please:
- Visit the [NVIDIA Developer Forums](https://forums.developer.nvidia.com/c/ai-data-science/86)
- [Visit our Youtube Channel](https://www.youtube.com/channel/UCsoi4wfweA3I5FsPgyQnnqw/featured?view_as=subscriber) or see [list of videos](multimedia_links.md) by RAPIDS or our community.  Feel free to contribute your videos and RAPIDS themed playlists as well!
- [Visit our Blogs on Medium](https://medium.com/rapids-ai/) 

### Additional Information
* The `data` folder also includes the full image set from the [Fashion MNIST dataset](https://github.com/zalandoresearch/fashion-mnist).

* `utils`: contains a set of useful scripts for interacting with RAPIDS Community Notebooks

* For our notebook examples and tutorials found on [github](https://github.com/rapidsai), in each respective repo.

