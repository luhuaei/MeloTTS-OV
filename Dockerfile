FROM python:3.9-slim
WORKDIR /app
COPY . /app

RUN apt-get update && apt-get install -y \
    build-essential libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv
RUN uv sync
RUN uv run python -m unidic download
RUN uv run python melo/init_downloads.py

CMD ["uv", "run", "python", "./melo/app.py", "--host", "0.0.0.0", "--port", "8888"]
