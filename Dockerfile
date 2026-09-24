FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements/ ./requirements/

RUN pip install --no-cache-dir -r requirements/runtime.txt

COPY app/ ./app/
COPY src/ ./src/
COPY config/ ./config/
COPY scripts/ ./scripts/
COPY gx/ ./gx/

RUN mkdir -p logs models/cache mlruns artifacts

COPY scripts/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["/entrypoint.sh"]
