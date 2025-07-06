FROM python:3.11-slim

WORKDIR /app

COPY env_server.py .

RUN pip install --no-cache-dir fastapi uvicorn gymnasium

EXPOSE 8000

CMD ["uvicorn", "env_server:app", "--host", "0.0.0.0", "--port", "8000"]
