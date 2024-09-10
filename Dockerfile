# Use CUDA 12.2 base image with Python 3.11
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

# Install Python 3.11 and other dependencies
RUN apt-get update && apt-get install -y python3.11 python3-pip git

# Create symlinks for python3 and python (if they don't already exist)
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 || true
RUN ln -sf /usr/bin/python3.11 /usr/bin/python || true

# Set the working directory in the container
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt /app/requirements.txt

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Print Python version and pip version
RUN python3 --version && python3 -m pip --version

# Install PyTorch separately to ensure CUDA compatibility
RUN python3 -m pip install torch --index-url https://download.pytorch.org/whl/cu121

# Print installed packages
RUN python3 -m pip list

# Install other requirements
RUN python3 -m pip install -r requirements.txt

# Print installed packages again
RUN python3 -m pip list

# Verify torch installation
RUN python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

# Clone and install lm-evaluation-harness
RUN git clone https://github.com/EleutherAI/lm-evaluation-harness && \
    cd lm-evaluation-harness && \
    python3 -m pip install -e .

# Copy the agent_eval directory and run.sh script into the container
COPY agent_eval /app/agent_eval
COPY run.sh /app/run.sh

# Make run.sh executable
RUN chmod +x /app/run.sh

# Add lm-evaluation-harness to PATH
ENV PATH="/app/lm-evaluation-harness:${PATH}"

# Set the entrypoint to run.sh
ENTRYPOINT ["/app/run.sh"]

# Use CMD to provide default arguments, which can be overridden
CMD ["", ""]
