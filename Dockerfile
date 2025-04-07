FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libpq-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY quant_ml_trader/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies for the dashboard
RUN pip install --no-cache-dir streamlit plotly

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=true

# Expose ports
EXPOSE 8501

# Set entry point
ENTRYPOINT ["streamlit", "run", "quant_ml_trader/dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
