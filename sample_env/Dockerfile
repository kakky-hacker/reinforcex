FROM python:3.11-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir fastapi uvicorn gymnasium numpy

EXPOSE 8000

CMD ["uvicorn", "env_server:app", "--host", "0.0.0.0", "--port", "8000"]
