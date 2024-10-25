# Use an official Ubuntu as a parent image
FROM ubuntu:20.04

# Set environment variables to prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    git \
    bzip2 \
    ca-certificates \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh && \
    $CONDA_DIR/bin/conda clean --all --yes

# Update PATH environment variable
ENV PATH=$CONDA_DIR/bin:$PATH

# Create the Conda environment
RUN conda create --name layer_skip python=3.10 -y

# Activate the Conda environment and install PyTorch and dependencies via Conda
RUN echo "source activate layer_skip" >> ~/.bashrc
ENV CONDA_DEFAULT_ENV=layer_skip
ENV PATH=$CONDA_DIR/envs/layer_skip/bin:$PATH

# Install PyTorch CPU-only and other dependencies using Conda
RUN conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 cpuonly -c pytorch -y

# Copy and install Python dependencies via pip (excluding torch, torchvision, torchaudio)
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# Copy the entire project into the Docker image
COPY . /app

# Set the working directory
WORKDIR /app

# Copy the entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Expose any necessary ports (if applicable)
# EXPOSE 8000

# Set the entrypoint
ENTRYPOINT ["/entrypoint.sh"]
