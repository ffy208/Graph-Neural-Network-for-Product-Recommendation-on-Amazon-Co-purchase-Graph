import torch
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support

def compute_metrics(logits:torch.Tensor, labels:torch.Tensor, thr:float=0.5):
    p = torch.sigmoid(logits).cpu().numpy()
    y = labels.cpu().numpy().astype(int)
    auc = roc_auc_score(y, p) if len(set(y))>1 else float("nan")
    ap  = average_precision_score(y, p) if len(set(y))>1 else float("nan")
    pred = (p >= thr).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    acc = (pred==y).mean()
    return {"auc":auc, "ap":ap, "precision":prec, "recall":rec, "f1":f1, "accuracy":acc}
