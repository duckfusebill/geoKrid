FROM rocm/pytorch:rocm6.1.3_ubuntu22.04_py3.10_pytorch_release-2.1.2

# upgrade PyTorch to >=2.4 (required by transformers 5.x) using ROCm 6.1 wheels
RUN pip install --no-cache-dir \
    torch==2.6.0 torchvision \
    --index-url https://download.pytorch.org/whl/rocm6.1

# project dependencies
RUN pip install --no-cache-dir \
    transformers \
    geoclip \
    geopy \
    pandas \
    Pillow \
    tqdm \
    numpy

WORKDIR /home/pro/proj/eel4935_proj/geovit384
