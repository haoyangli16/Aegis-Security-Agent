FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget git curl ca-certificates bzip2 libx11-6 build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install Miniconda
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh
ENV PATH=$CONDA_DIR/bin:$PATH

# Copy environment.yaml into container
COPY ../environment.yaml /tmp/environment.yaml

# Create conda environment
RUN conda env create -f /tmp/environment.yaml && conda clean -afy

# Set default conda env name
ARG CONDA_ENV_NAME
ENV CONDA_ENV_NAME=${CONDA_ENV_NAME:-base}

# Activate conda env for future RUNs
SHELL ["conda", "run", "-n", "${CONDA_ENV_NAME}", "/bin/bash", "-c"]

# Set working dir
WORKDIR /workspace

# (Optional) Default command
CMD ["python"]
