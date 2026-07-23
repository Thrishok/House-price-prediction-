FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies needed for scientific stack / matplotlib and pandas build
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       gcc \
       python3-dev \
       python3-wheel \
       gfortran \
       libopenblas-dev \
       liblapack-dev \
       libfreetype6-dev \
       libpng-dev \
       libjpeg-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (use webui requirements)
COPY webui/requirements.txt /app/webui/requirements.txt
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r /app/webui/requirements.txt \
    && pip install --no-cache-dir gunicorn

# Copy project files
COPY . /app

WORKDIR /app/webui

# Expose default port (Render provides $PORT at runtime)
ENV PORT=7070
EXPOSE ${PORT}

# Run with gunicorn and bind to the dynamic $PORT provided by the platform
# Use shell form so $PORT is expanded at container start
CMD sh -c "gunicorn --bind 0.0.0.0:$PORT app:app --workers 1 --timeout 120"
