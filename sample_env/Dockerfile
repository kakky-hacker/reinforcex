FROM python:3.11-slim

WORKDIR /app

COPY env_server.py .

RUN apt-get update && apt-get install -y \
    build-essential \
    swig \
    cmake \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

RUN pip install --no-cache-dir fastapi uvicorn gymnasium[all]

EXPOSE 8000

CMD ["uvicorn", "env_server:app", "--host", "0.0.0.0", "--port", "8000"]
