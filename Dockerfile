FROM rapidsai/rapidsai:cuda10.0-runtime-ubuntu18.04
USER root

ENV DEBIAN_FRONTEND noninteractive
ENV SCALA_VERSION 2.11
ENV KAFKA_VERSION 2.3.0
ENV KAFKA_HOME /opt/kafka_"$SCALA_VERSION"-"$KAFKA_VERSION"

SHELL ["/bin/bash", "-c"]
RUN source activate rapids && \
    conda install -c conda-forge -c defaults \
                        matplotlib \
                        scikit-learn \
                        seaborn \
                        python-louvain \
                        jinja2 \
                        librdkafka=1.1.0=hb2b7465_0 \
                        streamz=0.5.2=py_0 \
                        python-confluent-kafka=1.1.0=py36h516909a_0 \
                        ujson \
                        cugraph \
                        openjdk=8.0.152 \
                        graphviz \
                        kaggle && \
    pip install graphistry mockito

WORKDIR /rapids/notebooks/contrib

# Add everthing from the local build context (incuding the data folder)
ADD . .

# move /rapids/notebooks/contrib/data to /data, then symlink
# so users can browse the data directory inside JupyterLab
RUN mv data /data && ln -s /data /rapids/notebooks/contrib/data

# Zookeeper
EXPOSE 2181

# Kafka
EXPOSE 9092

# Install Kafka within the container
RUN wget -q https://archive.apache.org/dist/kafka/"$KAFKA_VERSION"/kafka_"$SCALA_VERSION"-"$KAFKA_VERSION".tgz -O /tmp/kafka_"$SCALA_VERSION"-"$KAFKA_VERSION".tgz && \
        tar xfz /tmp/kafka_"$SCALA_VERSION"-"$KAFKA_VERSION".tgz -C /opt && \
        rm /tmp/kafka_"$SCALA_VERSION"-"$KAFKA_VERSION".tgz

CMD ["/rapids/notebooks/contrib/docker-utils/scripts/entry.sh"]