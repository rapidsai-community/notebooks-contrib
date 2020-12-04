<center><img width='80%
' src='https://raw.githubusercontent.com/miroenev/rapids/master/rapids_workflow.png'></center>

# Rapids End to End ML Demo

# Contents

1. Motivate rapids [ show coverage of modern data science tools ]

2. Generate a synthetic dataset

* 2.1 - Split into train and test set
* 2.2 - Visualize sub-datasets

3. ETL

* 3.1 - Load data [ csv read ]     
* 3.2 - Transform data [ standard scaler ]

4. Model Building 

* 4.1 - Train CPU and GPU XGBoost classifier models 
* 4.2 - Use trained models for inference
* 4.3 - Compare accuracy
* 4.4 - Visualize sample boosted trees & model predictions

5. Extensions 

* 5.1 - Create an ensemble with a clustering model [ DBScan ]
* 5.2 - Export data to DeepLearning Framework [ PyTorch ]
    
<center><img width='80%
' src='https://raw.githubusercontent.com/miroenev/rapids/master/dataset.png'></center>


# Install & Run Demo 
> [ Video Walkthrough Link](https://www.dropbox.com/s/1qkmsnynog45ox8/rapids_walkthrough_4_25.mp4?dl=0)

## 1 -- clone repository

    git clone https://github.com/miroenev/rapids && cd rapids


### 2 -- build container [ takes 5-10 minutes ]

    sudo docker build -t rapids-demo:v0 .


### 3 -- launch/run the container [ auto starts jupyter notebook ]

    sudo docker run --runtime=nvidia -it --rm -p 8888:8888 -p 8787:8787 rapids-demo:v0

### 4 -- connect to notebook

    i) navigate browser to IP of machine running container at port 8888
        e.g., http://127.0.0.1:8888

    ii) in the rapids folder launch the notebook titled 
        rapids_ml_workflow_demo.ipynb