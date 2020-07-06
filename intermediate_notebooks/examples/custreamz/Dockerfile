FROM rapidsai/rapidsai:cuda10.0-runtime-ubuntu18.04
USER root

ENV DEBIAN_FRONTEND noninteractive
ENV SCALA_VERSION 2.11
ENV KAFKA_VERSION 2.3.0
ENV KAFKA_HOME /opt/kafka_"$SCALA_VERSION"-"$KAFKA_VERSION"

SHELL ["/bin/bash", "-c"]
RUN source activate rapids && \
    conda install -c conda-forge -c defaults \
                        librdkafka=1.1.0=hb2b7465_0 \
                        streamz=0.5.2=py_0 \
                        python-confluent-kafka=1.1.0=py36h516909a_0 \
                        ujson \
                        openjdk=8.0.152 \
                        graphviz

WORKDIR /rapids/notebooks/contrib

# Add all of the jupyter notebooks in this directory to the container in the jupyter "home" directory.
ADD . .

# Zookeeper
EXPOSE 2181

# Kafka
EXPOSE 9092

# Install Kafka within the container
RUN wget -q https://archive.apache.org/dist/kafka/"$KAFKA_VERSION"/kafka_"$SCALA_VERSION"-"$KAFKA_VERSION".tgz -O /tmp/kafka_"$SCALA_VERSION"-"$KAFKA_VERSION".tgz && \
        tar xfz /tmp/kafka_"$SCALA_VERSION"-"$KAFKA_VERSION".tgz -C /opt && \
        rm /tmp/kafka_"$SCALA_VERSION"-"$KAFKA_VERSION".tgz

CMD ["/rapids/notebooks/contrib/scripts/entry.sh"]