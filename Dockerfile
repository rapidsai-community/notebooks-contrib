FROM rapidsai/rapidsai:cuda9.2-runtime-ubuntu16.04

SHELL ["/bin/bash", "-c"]
RUN source activate rapids && conda install -y \
        matplotlib \
        scikit-learn \
        seaborn

# ToDo: let user supply kaggle creds
RUN source activate rapids && pip install kaggle

ADD data /data
ADD cpu_comparisons /rapids/notebooks/cpu_comparisons
ADD tutorials /rapids/notebooks/tutorials

WORKDIR /rapids/notebooks
CMD source activate rapids && sh /rapids/notebooks/utils/start-jupyter.sh
