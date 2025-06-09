FROM python:3.11-slim as builder

WORKDIR /app

# Install system dependencies required for PyTorch & other libraries
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc python3-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy only requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --user torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --user -r requirements.txt

# Final stage (smaller image)
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local
COPY . .

# Ensure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

CMD ["python", "server.py"]