# Use a specific x86-64 Ubuntu image
FROM ubuntu:22.04

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies from the workflow
RUN apt-get update && apt-get install -y \
    git \
    clang \
    llvm \
    gdb \
    build-essential \
    cmake \
    ninja-build \
    libopenblas-dev \
    pybind11-dev \
    python3-pip \
    python3-dev \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Install Python 3.11
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.11 python3.11-dev python3.11-venv && \
    rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as the default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Set up the working directory
WORKDIR /app

# Copy requirements and install dependencies first to leverage caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Build the shared library and install the package in editable mode
RUN pip install -e .

# Start a bash session by default
CMD ["/bin/bash"]
