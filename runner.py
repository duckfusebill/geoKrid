import os
import argparse
import pandas as pd
import torch
import dataset as D
import model as M
import train as T
import eval as E

"""config"""
CFG = {
    # data
    "train_csv":     "/mnt/b/datasets/geovit/us_train_cls.csv",
    "val_csv":       "/mnt/b/datasets/geovit/us_val_cls.csv",
    "osv_root":      "",          # image_path in csv is absolute
    "cells_csv":     "cells.csv",

    # output
    "ckpt_dir":      "checkpoints",
    "resume":        None,
    "save_every":    500,   # save mid-epoch checkpoint every N batches (0 = disable)

    # training
    "epochs":        10,
    "batch_size":    128,
    "lr":            3e-4,
    "weight_decay":  1e-2,
    "warmup_epochs": 1,
    "num_workers":   8,
    "device":        "cuda",

    # eval
    "val_every":     1,
}


def build_model(cfg):
    cells = pd.read_csv(cfg["cells_csv"])
    n_cells = len(cells)
    cell_centers = cells[["lat_center", "lon_center"]].values  # (N, 2)
    m = M.GeoClassifier(n_cells, cell_centers)
    return m.to(cfg["device"])


def build_optimizer(model, cfg):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])


def main(cfg):
    os.makedirs(cfg["ckpt_dir"], exist_ok=True)
    device      = cfg["device"]
    mid_ckpt    = os.path.join(cfg["ckpt_dir"], "mid_epoch.pt")

    model     = build_model(cfg)
    optimizer = build_optimizer(model, cfg)
    scheduler = T.warmup_cosine_scheduler(optimizer, cfg["warmup_epochs"], cfg["epochs"])

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {trainable:,} / {total:,}")

    start_epoch   = 1
    resume_batch  = 0

    if cfg["resume"]:
        epoch, resume_batch, _ = T.load_checkpoint(
            model, optimizer, cfg["resume"], device, scheduler
        )
        if resume_batch > 0:
            start_epoch = epoch
            print(f"resumed mid-epoch {epoch} at batch {resume_batch}")
        else:
            start_epoch = epoch + 1
            print(f"resumed from {cfg['resume']} at epoch {epoch}")

    for epoch in range(start_epoch, cfg["epochs"] + 1):
        skip = resume_batch if epoch == start_epoch else 0
        train_loader, val_loader = D.build_loaders(cfg, epoch=epoch, skip_batches=skip)

        loss = T.train_epoch(
            train_loader, model, optimizer, scheduler, epoch, device,
            save_every=cfg["save_every"],
            mid_ckpt_path=mid_ckpt,
            resume_batch=skip,
        )
        print(f"epoch {epoch}  avg loss {loss:.4f}")

        if epoch % cfg["val_every"] == 0:
            E.eval_model(val_loader, model, device)

        ckpt_path = os.path.join(cfg["ckpt_dir"], f"epoch_{epoch:03d}.pt")
        T.save_checkpoint(model, optimizer, epoch, ckpt_path)

    print("training complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv",     default=CFG["train_csv"])
    parser.add_argument("--val-csv",       default=CFG["val_csv"])
    parser.add_argument("--osv-root",      default=CFG["osv_root"])
    parser.add_argument("--cells-csv",     default=CFG["cells_csv"])
    parser.add_argument("--ckpt-dir",      default=CFG["ckpt_dir"])
    parser.add_argument("--resume",        default=CFG["resume"])
    parser.add_argument("--epochs",        type=int,   default=CFG["epochs"])
    parser.add_argument("--batch-size",    type=int,   default=CFG["batch_size"])
    parser.add_argument("--lr",            type=float, default=CFG["lr"])
    parser.add_argument("--weight-decay",  type=float, default=CFG["weight_decay"])
    parser.add_argument("--warmup-epochs", type=int,   default=CFG["warmup_epochs"])
    parser.add_argument("--num-workers",   type=int,   default=CFG["num_workers"])
    parser.add_argument("--device",        default=CFG["device"])
    parser.add_argument("--val-every",     type=int,   default=CFG["val_every"])
    parser.add_argument("--save-every",    type=int,   default=CFG["save_every"])
    args = parser.parse_args()

    cfg = dict(CFG)
    for k, v in vars(args).items():
        key = k.replace("-", "_")
        if key in cfg:
            cfg[key] = v

    main(cfg)
