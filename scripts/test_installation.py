print("🔍 Starting environment test for DGL PinSAGE example...\n")

# ==== Core modules ====
import os
import re
import pickle
import argparse
print("✅ Built-in modules loaded (os, re, pickle, argparse)")

# ==== Data + array handling ====
import numpy as np
import pandas as pd
import scipy.sparse as ssp
import dask.dataframe as dd
print("✅ Data modules loaded (numpy, pandas, scipy, dask)")

# ==== Deep learning ====
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
print(f"✅ Torch loaded (version: {torch.__version__}, CUDA: {torch.cuda.is_available()})")

# ==== Graph learning ====
import dgl
import dgl.function as fn
import dgl.nn.pytorch as dglnn
print(f"✅ DGL loaded (version: {dgl.__version__})")

# ==== Text processing ====
import torchtext
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
print("✅ TorchText loaded (tokenizer, vocab)")

# ==== Progress bar ====
import tqdm
print("✅ tqdm loaded")

# ==== Pandas data type helpers ====
from pandas.api.types import is_categorical_dtype, is_numeric_dtype
print("✅ pandas.api.types helpers loaded")

# ==== Project-local modules (test for expected failure if not present) ====
missing_local_modules = []
for module in ['sampler', 'evaluation', 'layers', 'builder', 'data_utils']:
    try:
        __import__(module)
        print(f"✅ Local module found: {module}.py")
    except ModuleNotFoundError:
        print(f"⚠️  Local module NOT found: {module}.py (expected if not copied yet)")
        missing_local_modules.append(module)

# ==== Final report ====
print("\n🎉 Environment test complete!")
if missing_local_modules:
    print("⚠️  Note: Some local scripts were not found. If you're setting up the full repo, be sure to include:")
    for mod in missing_local_modules:
        print(f"  - {mod}.py")
else:
    print("✅ All required modules are present and importable.")
