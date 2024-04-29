FROM python:3.11-slim-bullseye as runtime

RUN pip install numpy pandas PyYAML cloudpickle==3.0.0 && \
    rm -rf /root/.cache/pip/*

WORKDIR /app

FROM runtime as dev

COPY /src/requirements.txt /app/

RUN pip install --no-cache-dir -r /app/requirements.txt && \
    rm -rf /root/.cache/pip/*


FROM runtime as prod

RUN apt-get update && \
    apt-get install -y --no-install-recommends openjdk-17-jre-headless gcc libc6-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME /usr/lib/jvm/java-17-openjdk-amd64

RUN pip install torch==2.2.1 -f https://download.pytorch.org/whl/cpu/torch-2.1.1-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl#sha256=d6227060f268894f92c61af0a44c0d8212e19cb98d05c20141c73312d923bc0a \
    torchserve==0.10.0 \
    scikit-learn && \
    rm -rf /root/.cache/pip/*

COPY . /app

COPY build/docker/entrypoint.sh /
RUN chmod +x /entrypoint.sh

COPY build/docker/config.dev.properties /app/model-server/config.properties
COPY build/docker/log4j.properties /app/model-server/log4j.properties

ENTRYPOINT ["/entrypoint.sh"]