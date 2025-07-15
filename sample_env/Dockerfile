FROM python:3.11-slim

WORKDIR /app

COPY env_server.py .
COPY env_server.proto .

RUN apt-get update && apt-get install -y \
    build-essential \
    swig \
    cmake \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install --no-cache-dir grpcio grpcio-tools gymnasium[all]

RUN python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. env_server.proto

EXPOSE 8000

CMD ["python", "env_server.py"]
