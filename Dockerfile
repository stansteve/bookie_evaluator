FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VERSION=1.7.1 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_CREATE=false

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="$POETRY_HOME/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy poetry configuration files
COPY pyproject.toml poetry.lock* ./

# Install dependencies
RUN poetry install --no-interaction --no-ansi

# Copy project files
COPY . .

# Create directory for persistent data
RUN mkdir -p /data/matches
VOLUME ["/data"]

# Set environment variables for data paths
ENV DATA_DIR="/data"

# Expose Jupyter port
EXPOSE 8888

# Create entrypoint script
RUN echo '#!/bin/bash\n\
if [ "$1" = "jupyter" ]; then\n\
  exec poetry run jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root\n\
elif [ "$1" = "bash" ]; then\n\
  exec /bin/bash\n\
else\n\
  exec poetry run python main.py "$@"\n\
fi' > /app/entrypoint.sh \
&& chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]

# Default command (if no arguments provided)
CMD ["--help"]