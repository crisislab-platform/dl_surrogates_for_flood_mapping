FROM ubuntu:20.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies including Python and GDAL
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    gdal-bin \
    libgdal-dev \
    libproj-dev \
    libgeos-dev \
    libspatialite-dev \
    sqlite3 \
    git \
    wget \
    curl \
    vim \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Set GDAL environment variables
ENV GDAL_DATA=/usr/share/gdal
ENV PROJ_LIB=/usr/share/proj

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Copy requirements file first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install -r requirements.txt

# Copy the application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/data /app/output /app/logs

# Set Python path
ENV PYTHONPATH=/app:$PYTHONPATH

# Expose port for Jupyter
EXPOSE 8888

# Default command
CMD ["bash"]
