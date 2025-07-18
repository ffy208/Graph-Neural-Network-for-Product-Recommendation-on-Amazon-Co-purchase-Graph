#!/bin/bash
#SBATCH --job-name=gat_c4
#SBATCH --output=/home/hice1/fyang365/scratch/Group-Project/logs/train_c4_%j.log
#SBATCH --partition=ice-gpu
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu-a100|gpu-h100|gpu-h200|gpu-l40s
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:00:00

# ----- Directories and script paths -----
LOGDIR=/home/hice1/fyang365/scratch/Group-Project/logs
mkdir -p $LOGDIR
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE=$LOGDIR/train_c4_${TIMESTAMP}.log
TRAIN_SCRIPT=/home/hice1/fyang365/scratch/Group-Project/GAT/train_gat_graphsage_adv_nsfix.py

# ----- Hyper-parameters -----
EPOCHS=150
HIDDEN_DIM=320
HEADS=2
NUM_LAYERS=2
LR=0.005
WEIGHT_DECAY=0.0003
DROPOUT=0.015
NEG_RATIO=0.30
POS_WEIGHT=2
MIX_ALPHA=0.0

echo "=========================================="
echo "GAT Training Script - c4"
echo "Job ID: $SLURM_JOB_ID  Nodes: $SLURM_NODELIST"
echo "Log File: $LOGFILE"
echo "=========================================="
echo "Params: --epochs $EPOCHS --hidden_dim $HIDDEN_DIM --heads $HEADS --num_layers $NUM_LAYERS \
        --lr $LR --weight_decay $WEIGHT_DECAY --dropout $DROPOUT --neg_ratio $NEG_RATIO \
        --pos_weight $POS_WEIGHT --mix_alpha $MIX_ALPHA --scheduler cosine"
echo "=========================================="

# ----- Modules -----
module purge
module load anaconda3/2023.03 cuda/11.8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1

# ----- Conda environment -----
if ! conda env list | grep -q "gat_env"; then
  conda create -n gat_env python=3.9 -y
fi
source ~/.bashrc
conda activate gat_env

# ----- Dependency check -----
python - << 'PY'
import importlib, subprocess, sys
requirements = ['torch==2.2.0+cu118', 'torch_geometric', 'pandas', 'numpy', 'scikit-learn', 'matplotlib', 'seaborn']
for pkg in requirements:
    try:
        importlib.import_module(pkg.split('==')[0].replace('-', '_'))
    except ImportError:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg, '--progress-bar', 'off'])
PY

# ----- Training -----
cd /home/hice1/fyang365/scratch/Group-Project
mkdir -p logs models
python -u $TRAIN_SCRIPT \
  --epochs $EPOCHS --hidden_dim $HIDDEN_DIM --heads $HEADS \
  --num_layers $NUM_LAYERS --lr $LR --weight_decay $WEIGHT_DECAY \
  --dropout $DROPOUT --neg_ratio $NEG_RATIO --pos_weight $POS_WEIGHT \
  --mix_alpha $MIX_ALPHA --scheduler cosine \
  2>&1 | tee "$LOGFILE"

echo "===== Training Completed ====="
echo "Log saved to: $LOGFILE" 