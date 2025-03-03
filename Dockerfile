# Use the ROCm PyTorch image as the base image  
FROM rocm/megatron-lm:latest@sha256:1e6ed9bdc3f4ca397300d5a9907e084ab5e8ad1519815ee1f868faf2af1e04e2

# Update and install necessary system packages  
RUN apt-get update && \  
    apt-get install -y --no-install-recommends \  
    git \  
    build-essential && \  
    rm -rf /var/lib/apt/lists/*

# Set the default shell to use the conda environment  
# /opt/conda/envs/py_3.10/bin/python
SHELL ["conda", "run", "-n", "py_3.10", "/bin/bash", "-c"]  

# Install Python packages in the existing conda environment
# From loguru package we have things for evaluation kit = lmms-eval
RUN pip install \  
    transformers==4.44.0  \
    accelerate==1.1.1  \
    datasets==2.21.0  \
    deepspeed==0.15.4 \
    peft==0.12.0 \
    regex==2024.7.24 \
    timm==1.0.12 \
    boto3==1.35.63 \
    botocore==1.35.97 \
    loguru==0.7.3 \
    sacrebleu==2.5.1 \
    evaluate==0.4.3 \
    sqlitedict==2.1.0 \
    openai==1.63.2 \
    hf-transfer==0.1.9 \
    decord==0.6.0 \
    av==14.1.0 \
    pytablewriter==1.2.1 \
    openpyxl==3.1.5

# Specify the working directory
COPY . /InstellaVL

# Change to the correct directory and install the package  
RUN cd /InstellaVL && \
    git clone https://github.com/mosaicml/streaming.git && \  
    cp assets/patches/encodings.py streaming/streaming/base/format/mds/. && \
    cp assets/patches/reader.py streaming/streaming/base/format/base/. && \
    cp assets/patches/stream.py streaming/streaming/base/. && \
    cd streaming && \ 
    git checkout 6af3216e5608899e97f2f75be47c21d4106392fb && \ 
    pip install -e '.[all]'


# Build bitsandbytes with HSA_OVERRIDE_GFX_VERSION=9.4.2 for -DBNB_ROCM_ARCH="gfx942" for MI300
RUN cd /InstellaVL && \
    git clone --recurse https://github.com/ROCm/bitsandbytes.git && \ 
    cd bitsandbytes && \ 
    git checkout rocm_enabled && \ 
    pip install -r requirements-dev.txt && \ 
    export TORCH_BLAS_PREFER_HIPBLASLT=1 && \ 
    cmake -DCOMPUTE_BACKEND=hip -S . -DBNB_ROCM_ARCH="gfx942" -DCMAKE_HIP_COMPILER=/opt/rocm-6.3.0/lib/llvm/bin/clang++ && \ 
    make && \ 
    pip install .
    

# Clone the AMD VLM repository  
RUN cd /InstellaVL && \
    pip install -e . --no-deps

# Entry point  
CMD ["bash"]  
