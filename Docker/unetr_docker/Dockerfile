
# Use the NVIDIA CUDA base image
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

# Install dependencies
RUN apt-get update && \
    apt-get install -y python3-pip && \
    pip3 install --upgrade pip

# Install Jupyter Lab
RUN pip install jupyterlab && \
    pip install notebook

# Install compatible Python packages
RUN pip install torch==2.0.1 \
                torchvision==0.15.2 \
                monai==1.2.0 \
                lightning==2.0.7 \
                einops==0.6.1 \
                numpy==1.25.2 \
                matplotlib==3.7.2 \
                rasterio==1.3.8 \
                scikit-learn==1.3.0 \
                opendatasets==0.1.22 \
                tiffile

# Set the working directory
WORKDIR /Workspace

# Expose the necessary ports
EXPOSE 8888 9453

# Command to run Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
