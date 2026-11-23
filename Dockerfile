# Production Dockerfile for Render / Cloud Containers
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and UI assets
COPY . .

# Set default port for container environments
ENV PORT=10000
EXPOSE 10000

# Start FastAPI server
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT:-10000}"]
