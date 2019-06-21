

![RAPIDS](img/rapids_logo.png)



<p float="left">
  <img src="img/nvidia_logo.jpg" width="250" />
  <img src="./img/microsoft_logo.png" width="500" /> 
</p>





# Cloud-Based Data Science at the Speed of Thought Using RAPIDS - the Open GPU Data Science Ecosystem

## <center>KDD-2019 Tutorial</center>



Agenda:

1. Introduction (not hands-on) 
   1. Speaker Introduction
   2. Why RAPIDS
   3. Getting Connected to VMs 
   4. Break - 5 min
2. Tutorial (hands-on)
   1. Classification of astronomical sources - 45 min 
   2. Break - 5 min
   3. Finding commonalities within populations - 45 min
   4. Break - 5 min
   5. Cyber flagging anomalous network communications - 45 min
   6. Break - 5 min
3. Conclusion (not hands-on)
   1. DASK Example (Multi-GPU 
   2. Roadmap 
   3. Conclusion



**Workflow 1**
Classification of astronomical sources in the night sky is important for understanding the universe. It helps us understand the properties of what makes up celestial systems from our solar system to the most distant galaxy and everything in between. The Photometric LSST Astronomical Time-Series Classification Challenge (PLAsTiCC) wanted to revolutionize the field by automatically classifying 10–100x faster than previous methods and provided Kagglers a great dataset for solving this Kaggle problem using machine learning. The workflow comes from the RAPIDS submission to the Kaggle challenge, which came in 8th place out of 1094 submissions.

**Workflow 2**
Finding commonalities within populations, counted in hundreds of millions, and with tens of millions of distinct feature values, is a non-trivial task. Current techniques employ variations of FP-Tree algorithms to extract useful patterns. In this tutorial we will showcase an end-to-end novel approach to finding frequent patterns using cuGraph capabilities available in RAPIDS.

**Workflow 3**
Computer networks generate massive amounts of heterogeneous data as users interact with other users, computers, and services. These interactions can be modeled as large, heterogeneous property graphs, with multidimensional characteristics of the communication embedded on an edge connecting nodes. Current techniques to identify subgraph evolution over time and extract anomalies require lengthy compute times or necessitate significant pre-filtering of the graph. In this tutorial, we showcase an approach to flagging anomalous network communications in a large graph using a combination of structural graph features and graph analytics, running end-to-end in RAPIDS.