FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies needed for scientific stack / matplotlib
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       gcc \
       libfreetype6-dev \
       libpng-dev \
       libjpeg-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (use webui requirements)
COPY webui/requirements.txt /app/webui/requirements.txt
RUN pip install --no-cache-dir -r /app/webui/requirements.txt \
    && pip install --no-cache-dir gunicorn

# Copy project files
COPY . /app

WORKDIR /app/webui

# Expose Flask port
EXPOSE 7070

# Run with gunicorn for a simple production-ish server
CMD ["gunicorn", "--bind", "0.0.0.0:7070", "app:app", "--workers", "1", "--timeout", "120"]
