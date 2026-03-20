FROM python:3.11-slim AS base

LABEL maintainer="Shahin Hasanov"
LABEL description="Trade Intelligence Graph - Graph-based trade fraud detection"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ src/
COPY config/ config/
COPY setup.py .
COPY README.md .

# Install package
RUN pip install --no-cache-dir -e .

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser -d /app appuser
RUN chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/graphql || exit 1

EXPOSE 8000

CMD ["uvicorn", "graph_intel.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
