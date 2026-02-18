FROM python:3.12-slim

# System deps for OpenCV and PDF conversion
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps (bust cache: v3-openai)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

COPY . .

RUN mkdir -p uploads comparisons

EXPOSE 8899

# Use longer timeout for AI analysis calls
CMD ["gunicorn", "--bind", "0.0.0.0:8899", "--workers", "2", "--timeout", "300", "app:app"]
