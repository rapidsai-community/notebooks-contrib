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

### RAPIDS Event Notebooks
[These notebooks that we presented at conferences or meetups](https://github.com/rapidsai-community/event-notebooks).  While we strive to use open source or easily accessible data, some notebooks may require datasets that have restricted access.  They also will be frozen in time and not maintained as RAPIDS progresses.  Please download the appropriate RAPIDS version that these workflows were build on or expect to update them to the newer verisons.  Your favorite notebooks from our previous events can now be found there as well!

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

### Notebooks

| Notebook                                                                                                                                                                                                                                        | Topic(s)     | Tool(s)                                                                                                                                                                                                           |
|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [multi_gpu_pagerank](https://github.com/rapidsai-community/notebooks-contrib/blob/main/community_tutorials_and_guides/cugraph/multi_gpu_pagerank.ipynb)                                                                                         | pagerank     | [cugraph](https://docs.rapids.ai/api/cugraph/stable/)                                                                                                                                                             |
| [dsql_vs_pyspark_netflow](https://github.com/rapidsai-community/notebooks-contrib/blob/main/community_tutorials_and_guides/dask-sql/dsql_vs_pyspark_netflow.ipynb)                                                                              | ETL          | [dask-sql](https://dask-sql.readthedocs.io/en/latest/), [spark](https://spark.apache.org/)                                                                                                                        |\n| [smsl-dask-sql](https://github.com/rapidsai-community/notebooks-contrib/blob/main/community_tutorials_and_guides/dask-sql/smsl-dask-sql.ipynb)                                                                                                  | ETL, ML      | [dask-sql](https://dask-sql.readthedocs.io/en/latest/), [cuml](https://docs.rapids.ai/api/cuml/stable/)                                                                                                           |
| [rapids_ml_workflow_demo](https://github.com/rapidsai-community/notebooks-contrib/blob/main/community_tutorials_and_guides/synthetic_3D/rapids_ml_workflow_demo.ipynb)                                                                          | ETL, ML      | [cuml](https://docs.rapids.ai/api/cuml/stable/), [numba](https://numba.pydata.org/), [XGBoost](https://xgboost.readthedocs.io/en/stable/)                                                                         |
| [NYCTaxi-E2E](https://github.com/rapidsai-community/notebooks-contrib/blob/main/community_tutorials_and_guides/taxi/NYCTaxi-E2E.ipynb)                                                                                                          | ETL, EDA, ML | [dask-cudf](https://docs.rapids.ai/api/dask-cudf/stable/), [hvplot](https://hvplot.holoviz.org/), [cuspatial](https://docs.rapids.ai/api/cuspatial/stable/), [XGBoost](https://xgboost.readthedocs.io/en/stable/) |\n| [census_education2income_demo](https://github.com/rapidsai-community/notebooks-contrib/blob/main/community_tutorials_and_guides/census_education2income_demo.ipynb)                                                                             | ETL, ML      | [dask-cuda](https://docs.rapids.ai/api/dask-cuda/stable/), [cuml](https://docs.rapids.ai/api/cuml/stable/)                                                                                                        |
| [rf_demo](https://github.com/rapidsai-community/notebooks-contrib/blob/main/community_tutorials_and_guides/rf_demo.ipynb)                                                                                                                       | ML           | [cudf](https://docs.rapids.ai/api/cudf/stable/), [cuml](https://docs.rapids.ai/api/cuml/stable/)                                                                                                                  |\n| [umap_demo_full](https://github.com/rapidsai-community/notebooks-contrib/blob/main/community_tutorials_and_guides/umap_demo_full.ipynb)                                                                                                         | ML           | [cuml](https://docs.rapids.ai/api/cuml/stable/), [umap](https://umap-learn.readthedocs.io/en/latest/)                                                                                                             |
| [weather](https://github.com/rapidsai-community/notebooks-contrib/blob/main/community_tutorials_and_guides/weather.ipynb)                                                                                                                       | ETL, EDA     | [dask-cudf](https://docs.rapids.ai/api/dask-cudf/stable/), [cuspatial](https://docs.rapids.ai/api/cuspatial/stable/)                                                                                              |
| [Overview-Taxi](https://github.com/rapidsai-community/notebooks-contrib/blob/main/conference_notebooks/TMLS_2020/notebooks/Taxi/Overview-Taxi.ipynb)                                                                                            | ETL          | [rmm](https://docs.rapids.ai/api/rmm/stable/), [cudf](https://docs.rapids.ai/api/cudf/stable/), [cugraph](https://docs.rapids.ai/api/cugraph/stable/)                                                             |
| [federated_query_demo_dasksql](https://github.com/rapidsai-community/notebooks-contrib/blob/main/getting_started_materials/hello_worlds/dask-sql/federated_query_demo_dasksql.ipynb)                                                            | ETL          | [dask-sql](https://dask-sql.readthedocs.io/en/latest/)                                                                                                                                                            |
| [01_Introduction_to_RAPIDS](https://github.com/rapidsai-community/notebooks-contrib/blob/main/getting_started_materials/intro_tutorials_and_guides/01_Introduction_to_RAPIDS.ipynb)                                                             | ETL          | [cudf](https://docs.rapids.ai/api/cudf/stable/), [cuml](https://docs.rapids.ai/api/cuml/stable/)                                                                                                                  |\n| [02_Introduction_to_cuDF](https://github.com/rapidsai-community/notebooks-contrib/blob/main/getting_started_materials/intro_tutorials_and_guides/02_Introduction_to_cuDF.ipynb)                                                                 | ETL          | [cudf](https://docs.rapids.ai/api/cudf/stable/)                                                                                                                                                                   |
| [03_Introduction_to_Dask](https://github.com/rapidsai-community/notebooks-contrib/blob/main/getting_started_materials/intro_tutorials_and_guides/03_Introduction_to_Dask.ipynb)                                                                 | Tutorial     | [dask](https://www.dask.org/)                                                                                                                                                                                     |\n| [04_Introduction_to_Dask_using_cuDF_DataFrames](https://github.com/rapidsai-community/notebooks-contrib/blob/main/getting_started_materials/intro_tutorials_and_guides/04_Introduction_to_Dask_using_cuDF_DataFrames.ipynb)                     | ETL          | [dask-cudf](https://docs.rapids.ai/api/dask-cudf/stable/)                                                                                                                                                         |\n| [06_Introduction_to_Supervised_Learning](https://github.com/rapidsai-community/notebooks-contrib/blob/main/getting_started_materials/intro_tutorials_and_guides/06_Introduction_to_Supervised_Learning.ipynb)                                   | ML           | [cuml](https://docs.rapids.ai/api/cuml/stable/)                                                                                                                                                                   |
| [07_Introduction_to_XGBoost](https://github.com/rapidsai-community/notebooks-contrib/blob/main/getting_started_materials/intro_tutorials_and_guides/07_Introduction_to_XGBoost.ipynb)                                                           | ML           | [XGBoost](https://xgboost.readthedocs.io/en/stable/)                                                                                                                                                              |\n| [09_Introduction_to_Dimensionality_Reduction](https://github.com/rapidsai-community/notebooks-contrib/blob/main/getting_started_materials/intro_tutorials_and_guides/09_Introduction_to_Dimensionality_Reduction.ipynb)                         | ML           | [cuml](https://docs.rapids.ai/api/cuml/stable/)                                                                                                                                                                   |
| [10_Introduction_to_Clustering](https://github.com/rapidsai-community/notebooks-contrib/blob/main/getting_started_materials/intro_tutorials_and_guides/10_Introduction_to_Clustering.ipynb)                                                     | ML           | [cuml](https://docs.rapids.ai/api/cuml/stable/)                                                                                                                                                                   |
| [11_Introduction_to_Strings](https://github.com/rapidsai-community/notebooks-contrib/blob/main/getting_started_materials/intro_tutorials_and_guides/11_Introduction_to_Strings.ipynb)                                                           | EDA          | [cudf](https://docs.rapids.ai/api/cudf/stable/)                                                                                                                                                                   |
| [12_Introduction_to_Exploratory_Data_Analysis_using_cuDF](https://github.com/rapidsai-community/notebooks-contrib/blob/main/getting_started_materials/intro_tutorials_and_guides/12_Introduction_to_Exploratory_Data_Analysis_using_cuDF.ipynb) | ETL          | [cudf](https://docs.rapids.ai/api/cudf/stable/)                                                                                                                                                                   |
| [13_Introduction_to_Time_Series_Data_Analysis_using_cuDF](https://github.com/rapidsai-community/notebooks-contrib/blob/main/getting_started_materials/intro_tutorials_and_guides/13_Introduction_to_Time_Series_Data_Analysis_using_cuDF.ipynb) | ETL          | [cudf](https://docs.rapids.ai/api/cudf/stable/), [cuxfilter](https://docs.rapids.ai/api/cuxfilter/stable/)                                                                                                        |
| [14_Introduction_to_Machine_Learning_using_cuML](https://github.com/rapidsai-community/notebooks-contrib/blob/main/getting_started_materials/intro_tutorials_and_guides/14_Introduction_to_Machine_Learning_using_cuML.ipynb)                   | ETL, ML      | [cudf](https://docs.rapids.ai/api/cudf/stable/), [cuml](https://docs.rapids.ai/api/cuml/stable/)                                                                                                                  |

## Great places to get started <a name="get_started"></a>

### Topics
Click each topic to expand
<details>
  <summary>RAPIDS Libraries Basics</summary>
  
#### Teaching Notebooks and User Guides
* [Intro to RAPIDS Crash Course](getting_started_materials/README.md)
* [Intro Notebooks to RAPIDS](getting_started_materials/intro_tutorials_and_guides)- covers cuDF, Dask, cuML and XGBoost.
* [Official RAPIDS User Guides](https://docs.rapids.ai/user-guide)
* [10 Minutes to cuDF and Dask cuDF](https://docs.rapids.ai/api/cudf/stable/user_guide/10min/)
* [cuDF for Data Scientists: Functions for Data Wrangling (External)](https://medium.com/@tiraldj/cudf-for-data-scientists-part-1-2-functions-for-data-wrangling-12a8f889b33e#e7ee) - by [Mohammed R. Osman]()
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
  <summary>Deploying RAPIDS</summary>

* [Official RAPIDS Deployment Guide](Deploying RAPIDS — RAPIDS Deployment Documentation documentation)
* [Video- Tutorial of RAPIDS on AWS Sagemaker](https://www.youtube.com/watch?v=BtE4d0v6Css)
* [Video- Tutorial of RAPIDS on AzureML](https://www.youtube.com/watch?v=aqTmVVFnEwI)
* [Bursting Data Science Workloads to GPUs on Google Cloud Platform with Dask Cloud Provider (Blog with Code snippets)](https://medium.com/rapids-ai/bursting-data-science-workloads-to-gpus-on-google-cloud-platform-with-dask-cloud-provider-685be1eff204)
* [Step by Step - Tutorial of RAPIDS on IBM Virtual Server Instance](https://medium.com/@ahmed_82744/deploy-rapids-on-ibm-cloud-virtual-server-for-vpc-ce3e4b3ede1c)- by  [Muhammad Arif](https://www.linkedin.com/in/arifnafees/) in collabaration with [Syed Afzal Ahmed](https://www.linkedin.com/in/syed-ahmed-6927749/)
* [Step by Step - Tutorial of RAPIDS on IBM Kubernetes Service](https://medium.com/@ahmed_82744/deploy-rapids-on-ibm-cloud-kubernetes-service-920de68dc6c4)- by  [Muhammad Arif](https://www.linkedin.com/in/arifnafees/) in collabaration with [Syed Afzal Ahmed](https://www.linkedin.com/in/syed-ahmed-6927749/)



</details>
<details>
  <summary>Multi GPU </summary>

  #### Getting Started 
* [Hello Word to Dask](getting_started_materials/hello_worlds/Dask_Hello_World.ipynb)
* [Intro to Dask](getting_started_materials/intro_tutorials_and_guides/03_Introduction_to_Dask.ipynb)
* [Dask using cuDF](getting_started_materials/intro_tutorials_and_guides/04_Introduction_to_Dask_using_cuDF_DataFrames.ipynb)
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

