FROM rapidsai/rapidsai:cuda9.2-runtime-ubuntu16.04

RUN conda install -y \
        matplotlib \
        scikit-learn \
        seaborn

# ToDo: let user supply kaggle creds
RUN pip install kaggle

ADD data /data
ADD cpu_comparisons /rapids/notebooks/
ADD tutorials /rapids/notebooks/

WORKDIR /rapids/notebooks
CMD sh /rapids/notebooks/utils/start-jupyter.sh
