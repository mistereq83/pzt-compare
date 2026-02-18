FROM python:3.12-slim

# System deps for OpenCV and PDF conversion
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

COPY . .

# Ensure dirs exist
RUN mkdir -p uploads comparisons

EXPOSE 8899

CMD ["gunicorn", "--bind", "0.0.0.0:8899", "--workers", "2", "--timeout", "120", "app:app"]
