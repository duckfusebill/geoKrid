import math
import os
import signal
import sys
import torch
import torch.nn as nn
from tqdm import tqdm

"""haversine"""
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = torch.deg2rad(lat2 - lat1)
    dlon = torch.deg2rad(lon2 - lon1)
    a = (torch.sin(dlat / 2) ** 2
         + torch.cos(torch.deg2rad(lat1))
         * torch.cos(torch.deg2rad(lat2))
         * torch.sin(dlon / 2) ** 2)
    return 2 * R * torch.asin(torch.clamp(torch.sqrt(a), 0.0, 1.0))


"""mid-epoch checkpoint"""
def _save_mid(model, optimizer, scheduler, scaler, epoch, batch_idx,
              total_loss, n_batches, path):
    torch.save({
        "model":      model.state_dict(),
        "optimizer":  optimizer.state_dict(),
        "scheduler":  scheduler.state_dict(),
        "scaler":     scaler.state_dict(),
        "epoch":      epoch,
        "batch_idx":  batch_idx,
        "total_loss": total_loss,
        "n_batches":  n_batches,
        "mid_epoch":  True,
    }, path)


"""training loop"""
def train_epoch(loader, model, optimizer, scheduler, epoch, device,
                save_every=500, mid_ckpt_path=None, resume_batch=0):
    model.train()
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda")
    total_loss = 0.0
    n_batches  = 0

    total_batches = resume_batch + len(loader)

    _interrupted = [False]
    def _handle_stop(sig, frame):
        _interrupted[0] = True
    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT,  _handle_stop)

    bar = tqdm(loader, total=total_batches, initial=resume_batch,
               desc=f"epoch {epoch}")

    for batch in bar:
        if batch is None:
            continue

        imgs, cell_ids = batch
        imgs     = imgs.to(device, non_blocking=True)
        cell_ids = cell_ids.to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.amp.autocast("cuda"):
            logits = model(imgs)
            loss   = criterion(logits, cell_ids)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        n_batches  += 1
        absolute   = resume_batch + n_batches
        bar.set_postfix(loss=f"{loss.item():.4f}")

        if mid_ckpt_path and save_every > 0 and absolute % save_every == 0:
            _save_mid(model, optimizer, scheduler, scaler, epoch, absolute,
                      total_loss, n_batches, mid_ckpt_path)

        if _interrupted[0]:
            if mid_ckpt_path:
                _save_mid(model, optimizer, scheduler, scaler, epoch, absolute,
                          total_loss, n_batches, mid_ckpt_path)
                print(f"\npaused at batch {absolute}/{total_batches}  "
                      f"({absolute/total_batches*100:.1f}%)")
                print(f"resume: python3 runner.py --resume {mid_ckpt_path}")
            sys.exit(0)

    # epoch complete — delete mid-epoch checkpoint if it exists
    if mid_ckpt_path and os.path.exists(mid_ckpt_path):
        os.remove(mid_ckpt_path)

    if scheduler is not None:
        scheduler.step()

    return total_loss / max(n_batches, 1)


"""scheduler"""
def warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / max(warmup_epochs, 1)
        t = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + math.cos(math.pi * t))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


"""checkpointing"""
def save_checkpoint(model, optimizer, epoch, path):
    torch.save({
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch":     epoch,
    }, path)


def load_checkpoint(model, optimizer, path, device, scheduler=None):
    """
    Loads a checkpoint. Returns (epoch, resume_batch, extra).
    For mid-epoch checkpoints, resume_batch > 0 and extra has loss accumulators.
    For end-of-epoch checkpoints, resume_batch == 0.
    """
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])

    if ckpt.get("mid_epoch"):
        if scheduler is not None and "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        return ckpt["epoch"], ckpt["batch_idx"], {
            "total_loss": ckpt["total_loss"],
            "n_batches":  ckpt["n_batches"],
        }

    return ckpt["epoch"], 0, {}
