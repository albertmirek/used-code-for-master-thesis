FROM ghcr.io/mlflow/mlflow:v2.11.0 as runtime

WORKDIR /app

RUN apt-get update && apt-get install -y \
    default-libmysqlclient-dev \
    build-essential \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

RUN pip install mysqlclient psycopg2-binary boto3


# ---- Build Stage ----
FROM runtime AS prod


COPY . /app


ENTRYPOINT mlflow server --backend-store-uri mysql://${DB_USER}:${DB_PASSWORD}@${DB_HOST}:${DB_PORT}/${DB_NAME} --artifacts-destination s3://${BUCKET_NAME} --host 0.0.0.0 --port $APP_PORT
