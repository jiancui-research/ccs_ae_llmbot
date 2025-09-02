# Use Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    unzip \
    git \
    gcc \
    python3-dev \
    curl \
    nano \
    vim \
    htop \
    tree \
    bash-completion \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python packages using pip
RUN pip install --no-cache-dir -r requirements.txt

# Install additional tools for data download
RUN pip install --no-cache-dir gdown

# Copy the entire project
COPY . .

# Make startup script executable
RUN chmod +x scripts/startup.sh

# Set up enhanced bash environment
RUN chmod +x scripts/bash_setup.sh && \
    echo "source /workspace/scripts/bash_setup.sh" >> /root/.bashrc

# Create necessary directories
RUN mkdir -p measurement_data \
    && mkdir -p publisher_response/results \
    && mkdir -p bot_preference/results \
    && mkdir -p mem_analysis/results \
    && mkdir -p chatgpt_casestudy/results

# Set environment variables
ENV PYTHONPATH=/workspace
ENV JUPYTER_ENABLE_LAB=yes

# Expose Jupyter port
EXPOSE 8888

# Use startup script that downloads data and starts Jupyter
CMD ["./scripts/startup.sh"] 