FROM nvcr.io/nvidia/rapidsai/rapidsai:0.7-cuda10.0-runtime-ubuntu18.04-gcc7-py3.7

RUN apt update && apt -y upgrade

RUN source activate rapids && conda install -y -c conda-forge nodejs

RUN source activate rapids && conda install -y -c conda-forge ipywidgets
RUN source activate rapids && jupyter labextension install @jupyter-widgets/jupyterlab-manager

RUN source activate rapids && conda install -y -c conda-forge ipyvolume

RUN source activate rapids && jupyter labextension install ipyvolume
RUN source activate rapids && jupyter labextension install jupyter-threejs

RUN source activate rapids && conda install -c conda-forge python-graphviz 

RUN apt -y --fix-missing install font-manager unzip git vim htop

RUN git clone https://github.com/miroenev/rapids

# the kaggle survey data is now hardcoded into the figure code -- for anyone interested feel free to use the original sources by uncommenting the three lines below
#RUN cd rapids && mkdir -p kaggle_data/2017 && mv kaggle-survey-2017.zip kaggle_data/2017 && cd kaggle_data/2017 && unzip *.zip
#RUN cd rapids && mkdir -p kaggle_data/2018 && mv kaggle-survey-2018.zip kaggle_data/2018 && cd kaggle_data/2018 && unzip *.zip
#RUN cd rapids && cd kaggle_data && wget -O results.csv https://raw.githubusercontent.com/adgirish/kaggleScape/d291e121b2ece69cac715b4c89f4f19b684d4d02/results/annotResults.csv

# enables demo of ETL with RAPIDS and model building with DL-framework [ optional extension ]
RUN source activate rapids && conda install -y -c pytorch pytorch    

EXPOSE 8888

CMD ["bash", "-c", "source activate rapids && jupyter lab --no-browser --ip=0.0.0.0 --allow-root --NotebookApp.token=''"]
