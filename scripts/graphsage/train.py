import os, torch
from torch.cuda.amp import autocast, GradScaler
from .metrics import compute_metrics

def train_one_epoch(model, loader, optimizer, criterion, device, amp=True):
    model.train(); scaler = GradScaler(enabled=amp); total=0.0; count=0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=amp):
            logits = model(batch.x, batch.edge_index, batch.edge_label_index)
            loss = criterion(logits.view(-1), batch.edge_label.float())
        scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        total += float(loss) * batch.edge_label.numel(); count += batch.edge_label.numel()
    return total / max(count, 1)

@torch.no_grad()
def evaluate(model, loader, criterion, device, thr=0.5):
    model.eval(); tot=0.0; n=0; L=[]; Y=[]
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.edge_label_index)
        loss = criterion(logits.view(-1), batch.edge_label.float())
        tot += float(loss)*batch.edge_label.numel(); n += batch.edge_label.numel()
        L.append(logits.detach()); Y.append(batch.edge_label.detach())
    L = torch.cat(L); Y = torch.cat(Y).float()
    m = compute_metrics(L, Y, thr); m["loss"] = tot / max(n,1); return m

def save_ckpt(model, opt, path, epoch, best):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"state": model.state_dict(), "opt": opt.state_dict(),
                "epoch": epoch, "best": best}, path)

def fit(model, train_loader, val_loader, optimizer, criterion, device,
        epochs=100, patience=20, ckpt_path="results/graphsage_ckpts/best.pt", amp=True):
    best_auc = -1.0; wait = 0; hist=[]
    for ep in range(1, epochs+1):
        tl = train_one_epoch(model, train_loader, optimizer, criterion, device, amp)
        val = evaluate(model, val_loader, criterion, device)
        hist.append({"epoch":ep, "train_loss":tl, **{f"val_{k}":v for k,v in val.items()}})
        if val["auc"] > best_auc:
            best_auc = val["auc"]; wait = 0; save_ckpt(model, optimizer, ckpt_path, ep, best_auc)
        else:
            wait += 1
        if wait >= patience: break
    return hist, ckpt_path
