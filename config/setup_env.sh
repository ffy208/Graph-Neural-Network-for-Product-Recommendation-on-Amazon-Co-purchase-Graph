#!/bin/bash

# Install and activate Conda (Colab only)
pip install -q condacolab
python -c "import condacolab; condacolab.install()"

# Create a clean environment
conda create -n pinsage_env python=3.10 -y

# Activate environment
source activate pinsage_env

# Install PyTorch 2.1.0 with CUDA 11.8
conda run -n pinsage_env pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0 triton==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# Install DGL 2.0.0 with CUDA 11.8
conda install -n pinsage_env -c dglteam/label/cu118 dgl=2.0.0 -y

# Install compatible NumPy version (<2.0) to avoid ABI issues
conda run -n pinsage_env pip install "numpy<2.0"

# Install other dependencies
conda run -n pinsage_env pip install \
    pandas==2.1.3 \
    scipy \
    tqdm \
    torchtext==0.15.2 \
    torchdata==0.6.1 \
    dask[dataframe] \
    filelock \
    fsspec \
    jinja2 \
    networkx==3.2 \
    sympy \
    requests \
    pillow \
    typing_extensions \
    ipython

# Install scikit-learn and matplotlib
conda run -n pinsage_env pip install scikit-learn matplotlib

# Optional: Torch Geometric (PyG)
conda run -n pinsage_env pip install \
    torch-scatter torch-sparse torch-cluster torch-geometric \
    -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

# Sanity check
conda run -n pinsage_env python -c \
    "import torch, dgl, pandas, sympy, sklearn; print('✅ Torch:', torch.__version__, '| CUDA:', torch.cuda.is_available(), '| DGL:', dgl.__version__, '| Sklearn:', sklearn.__version__)"
