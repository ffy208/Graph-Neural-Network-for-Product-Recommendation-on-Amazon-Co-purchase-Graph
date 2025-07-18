#!/bin/bash

# Install and activate Conda
pip install -q condacolab
python -c "import condacolab; condacolab.install()"

# Create env
conda create -n pinsage_env python=3.10 -y

# Install PyTorch 2.1.0 with CUDA 11.8
conda run -n pinsage_env pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# Install DGL 2.0.0 with CUDA 11.8
conda install -n pinsage_env -c dglteam/label/cu118 dgl=2.0.0 -y

# Install other pip deps
conda run -n pinsage_env pip install numpy==1.24.4 pandas==2.1.3 scipy tqdm torchtext==0.15.2 dask[dataframe]
conda run -n pinsage_env pip install filelock fsspec jinja2 networkx sympy triton==2.1.0 requests torchdata==0.6.1 pillow typing_extensions

# scikit-learn and matplotlib
conda run -n pinsage_env pip install scikit-learn matplotlib

# Optional: Torch Geometric
conda run -n pinsage_env pip install torch-scatter torch-sparse torch-cluster torch-geometric -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

# Sanity check
conda run -n pinsage_env python -c "import torch, dgl, pandas, sympy, sklearn; print('✅ Torch:', torch.__version__, '| CUDA:', torch.cuda.is_available(), '| DGL:', dgl.__version__, '| Sklearn:', sklearn.__version__)"
