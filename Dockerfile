FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PATH="/opt/conda/bin:$PATH"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    build-essential \
    ca-certificates \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -a -y && \
    mkdir -p /etc/profile.d && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

# Install dependencies with pip instead of relying on conda environment
RUN conda install -y python=3.9 pip && \
    conda clean -afy

# Set up Python environment
RUN pip install torch==2.0.0 torchvision==0.15.0 --index-url https://download.pytorch.org/whl/cu117 && \
    pip install omegaconf torchmetrics==0.10.3 fvcore iopath aim && \
    pip install git+https://github.com/facebookincubator/submitit && \
    pip install --extra-index-url https://pypi.nvidia.com cuml-cu11 && \
    pip install xformers==0.0.18

# Set working directory
WORKDIR /workspace/dinov2

# Copy the repository
COPY . /workspace/dinov2/

# Install the package in development mode
RUN pip install -e .
RUN pip install -r requirements-extras.txt

# Set the entrypoint
# ENTRYPOINT ["python"]

# Default command
CMD ["--help"]
