"""Train the UNet-ResNet50 flood member on Kuro Siwo. Run on GPU with torchrun:

    torchrun --nproc_per_node=2 train.py --config configs/unet_r50.yaml

Outputs in <out_dir>/<run_name>/:
  best.pt, model.onnx, norm.json, splits.json, history.json, metrics.json, model_card.json
"""

import argparse
import csv
import json
import math
import os
import random
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

ALLOWED_DATASETS = {"kurosiwo", "sen1floods11"}


# ----------------------------------------------------------------------------- data


def load_meta(data_dir: Path) -> list[dict]:
    rows = []
    for meta in sorted(data_dir.glob("*_meta.csv")):
        tag = meta.name[: -len("_meta.csv")]
        with open(meta) as fh:
            for i, r in enumerate(csv.DictReader(fh)):
                r["chunk"], r["i"] = tag, i
                r["actid"] = int(r["actid"])
                for k in ("pflood", "pwater", "valid_frac", "relief_m", "slope_deg"):
                    r[k] = float(r[k])
                rows.append(r)
    return rows


def make_splits(rows: list[dict], cfg: dict) -> dict[str, list[int]]:
    """Event-level splits plus a mountain hold-out fold chosen by DEM relief only."""
    by_event: dict[int, list[float]] = {}
    for r in rows:
        if math.isfinite(r["relief_m"]):
            by_event.setdefault(r["actid"], []).append(r["relief_m"])
    relief = {a: float(np.median(v)) for a, v in by_event.items()}
    k = cfg["splits"]["mountain_holdout_events"]
    mountain = sorted(relief, key=lambda a: -relief[a])[:k]
    sp = cfg["splits"]
    out: dict[str, list[int]] = {"train": [], "val": [], "test": [], "mountain": []}
    for idx, r in enumerate(rows):
        a = r["actid"]
        if r["valid_frac"] < 0.5:
            continue
        if a in mountain:
            out["mountain"].append(idx)
        elif a in sp["train_acts"]:
            out["train"].append(idx)
        elif a in sp["val_acts"]:
            out["val"].append(idx)
        elif a in sp["test_acts"]:
            out["test"].append(idx)
    out_meta = {
        "event_relief_m": {str(a): round(v, 1) for a, v in sorted(relief.items())},
        "mountain_events": mountain,
        "counts": {s: len(v) for s, v in out.items()},
    }
    return out | {"_meta": out_meta}  # type: ignore[return-value]


class KuroSiwo(Dataset):
    def __init__(self, data_dir: Path, rows, idx, norm, aug: dict | None):
        self.data_dir, self.rows, self.idx, self.aug = data_dir, rows, idx, aug
        self.mean = np.array(norm["mean"], np.float32)[:, None, None]
        self.std = np.array(norm["std"], np.float32)[:, None, None]
        self._x: dict[str, np.ndarray] = {}
        self._y: dict[str, np.ndarray] = {}

    def _arr(self, tag: str):
        if tag not in self._x:
            self._x[tag] = np.load(self.data_dir / f"{tag}_x.npy", mmap_mode="r")
            self._y[tag] = np.load(self.data_dir / f"{tag}_y.npy", mmap_mode="r")
        return self._x[tag], self._y[tag]

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, j: int):
        r = self.rows[self.idx[j]]
        xa, ya = self._arr(r["chunk"])
        x = np.array(xa[r["i"]], dtype=np.float32)  # (6,H,W) dB: post, pre1, pre2 × (VV,VH)
        y = np.array(ya[r["i"]], dtype=np.int64)
        a = self.aug
        if a:
            if random.random() < a["swap_pre"]:
                x[[2, 3, 4, 5]] = x[[4, 5, 2, 3]]
            if random.random() < a["pre2_dropout"]:
                x[[4, 5]] = x[[2, 3]]
            g = a["gain_jitter_db"]
            for d in range(3):
                x[2 * d : 2 * d + 2] += np.float32(random.gauss(0, g))
            x += np.random.normal(0, a["speckle_db"], x.shape).astype(np.float32)
            if a["flips"]:
                if random.random() < 0.5:
                    x, y = x[:, :, ::-1], y[:, ::-1]
                if random.random() < 0.5:
                    x, y = x[:, ::-1], y[::-1]
                k = random.randint(0, 3)
                x, y = np.rot90(x, k, (1, 2)), np.rot90(y, k)
        x = (x - self.mean) / self.std
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(y))


def compute_norm(data_dir: Path, rows, idx, n: int = 2000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    pick = rng.choice(idx, size=min(n, len(idx)), replace=False)
    ds = KuroSiwo(data_dir, rows, list(pick), {"mean": [0] * 6, "std": [1] * 6}, None)
    acc = []
    for j in range(len(ds)):
        x, y = ds[j]
        m = (y != 255).numpy()
        acc.append(x.numpy()[:, m][:, ::16])
    a = np.concatenate(acc, axis=1)
    return {"mean": a.mean(1).round(4).tolist(), "std": a.std(1).round(4).tolist(), "units": "dB"}


# ----------------------------------------------------------------------------- model / loss


def build_model(cfg: dict):
    import segmentation_models_pytorch as smp

    m = cfg["model"]
    if m["encoder_weights"] and not m["allow_pretrained_weights"]:
        raise RuntimeError(
            "Pretrained encoder weights are a gray area; set allow_pretrained_weights"
        )
    return smp.Unet(
        encoder_name=m["encoder"],
        encoder_weights=m["encoder_weights"],
        in_channels=m["in_channels"],
        classes=m["classes"],
    )


def dice_loss(logits, y, n_cls: int):
    valid = (y != 255).unsqueeze(1)
    p = logits.float().softmax(1) * valid
    oh = F.one_hot(y.clamp(max=n_cls - 1), n_cls).permute(0, 3, 1, 2).float() * valid
    inter = (p * oh).sum((0, 2, 3))
    den = p.sum((0, 2, 3)) + oh.sum((0, 2, 3))
    return 1 - ((2 * inter + 1) / (den + 1)).mean()


# ----------------------------------------------------------------------------- metrics


class Confusion:
    def __init__(self, n: int = 3):
        self.n, self.m = n, np.zeros((n, n), np.int64)

    def add(self, pred: np.ndarray, y: np.ndarray):
        v = y != 255
        self.m += np.bincount(self.n * y[v] + pred[v], minlength=self.n**2).reshape(self.n, self.n)

    def scores(self) -> dict:
        tp = np.diag(self.m).astype(float)
        fp, fn = self.m.sum(0) - tp, self.m.sum(1) - tp
        iou = tp / np.maximum(tp + fp + fn, 1)
        prec, rec = tp / np.maximum(tp + fp, 1), tp / np.maximum(tp + fn, 1)
        f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
        names = ["no_water", "perm_water", "flood"]
        out = {f"iou_{k}": round(float(iou[i]), 4) for i, k in enumerate(names)}
        out |= {f"f1_{k}": round(float(f1[i]), 4) for i, k in enumerate(names)}
        out |= {
            "precision_flood": round(float(prec[2]), 4),
            "recall_flood": round(float(rec[2]), 4),
        }
        out["miou"] = round(float(iou.mean()), 4)
        out["n_pixels"] = int(self.m.sum())
        return out


@torch.no_grad()
def predict_logits(model, loader, device, amp: bool):
    model.eval()
    for x, y in loader:
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            lg = model(x.to(device, non_blocking=True))
        yield lg.float(), y.to(device)


def evaluate(model, loader, device, amp, rows=None, idx=None, temperature: float = 1.0):
    """Overall + per-event confusion; flood-probability calibration bins (ECE)."""
    conf = Confusion()
    per_event: dict[int, Confusion] = {}
    bins = np.zeros((10, 3))  # count, sum p, sum y  (flood vs not)
    k = 0
    for lg, y in predict_logits(model, loader, device, amp):
        p = (lg / temperature).softmax(1)
        pred = p.argmax(1).cpu().numpy()
        yy = y.cpu().numpy()
        pf = p[:, 2].cpu().numpy()
        conf.add(pred.ravel(), yy.ravel())
        v = yy != 255
        b = np.clip((pf[v] * 10).astype(int), 0, 9)
        np.add.at(bins, (b, 0), 1)
        np.add.at(bins, (b, 1), pf[v])
        np.add.at(bins, (b, 2), (yy[v] == 2))
        if rows is not None and idx is not None:
            for s in range(len(yy)):
                a = rows[idx[k + s]]["actid"]
                per_event.setdefault(a, Confusion()).add(pred[s].ravel(), yy[s].ravel())
        k += len(yy)
    ece = float(np.sum(np.abs(bins[:, 1] - bins[:, 2])) / max(bins[:, 0].sum(), 1))
    out = conf.scores() | {"ece_flood": round(ece, 4)}
    if per_event:
        ev = {str(a): c.scores() for a, c in sorted(per_event.items())}
        f1s = [s["f1_flood"] for s in ev.values() if s["n_pixels"] > 0]
        out["per_event"] = ev
        out["event_mean_f1_flood"] = round(float(np.mean(f1s)), 4) if f1s else None
        out["event_f1_flood_ci95"] = bootstrap_ci(f1s)
    return out


def bootstrap_ci(vals: list[float], n: int = 2000, seed: int = 0):
    if len(vals) < 2:
        return None
    rng = np.random.default_rng(seed)
    a = np.array(vals)
    means = rng.choice(a, (n, len(a)), replace=True).mean(1)
    return [round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)]


def fit_temperature(model, loader, device, amp) -> float:
    """Scalar temperature minimising NLL on the validation set (Guo et al. 2017)."""
    logits, labels = [], []
    for lg, y in predict_logits(model, loader, device, amp):
        v = y != 255
        lg = lg.permute(0, 2, 3, 1)[v][::50]
        logits.append(lg.cpu())
        labels.append(y[v][::50].cpu())
    lg, yy = torch.cat(logits), torch.cat(labels)
    t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(lg / t.exp(), yy)
        loss.backward()
        return loss

    opt.step(closure)
    return float(t.exp())


# ----------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--limit", type=int, help="debug: cap samples per split")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    if cfg["dataset"] not in ALLOWED_DATASETS:
        raise RuntimeError(f"training dataset {cfg['dataset']!r} is not an allowed source")
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs

    ddp = "RANK" in os.environ
    if ddp:
        dist.init_process_group("nccl", timeout=timedelta(hours=2))  # rank 0 validates alone
        rank = dist.get_rank()
    else:
        rank = 0
    device = torch.device("cuda", rank % max(torch.cuda.device_count(), 1))
    torch.cuda.set_device(device)
    seed = cfg["seed"] + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    data_dir = Path(cfg["data_dir"])
    run = Path(cfg["out_dir"]) / cfg["run_name"]
    run.mkdir(parents=True, exist_ok=True)
    rows = load_meta(data_dir)
    splits = make_splits(rows, cfg)
    if args.limit:
        for s in ("train", "val", "test", "mountain"):
            splits[s] = splits[s][: args.limit]
    norm_path = run / "norm.json"
    if rank == 0:
        Path(run / "splits.json").write_text(json.dumps(splits["_meta"], indent=1))
        print("splits", splits["_meta"]["counts"], "mountain", splits["_meta"]["mountain_events"])
        if not norm_path.exists():
            Path(norm_path).write_text(json.dumps(compute_norm(data_dir, rows, splits["train"])))
    if ddp:
        dist.barrier()
    norm = json.loads(norm_path.read_text())

    tcfg = cfg["train"]
    train_ds = KuroSiwo(data_dir, rows, splits["train"], norm, cfg["augment"])
    sampler = DistributedSampler(train_ds, shuffle=True, seed=cfg["seed"]) if ddp else None
    train_dl = DataLoader(
        train_ds,
        batch_size=tcfg["batch_size"],
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=tcfg["num_workers"],
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )

    def eval_loader(name):
        ds = KuroSiwo(data_dir, rows, splits[name], norm, None)
        return DataLoader(ds, batch_size=32, shuffle=False, num_workers=tcfg["num_workers"])

    model = build_model(cfg).to(device)
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device.index])
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    steps = tcfg["epochs"] * len(train_dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=tcfg["lr"], total_steps=steps, pct_start=0.05
    )
    scaler = torch.amp.GradScaler(enabled=tcfg["amp"])
    ce_w = torch.tensor(tcfg["ce_weights"], device=device)
    history, best, start_epoch = [], -1.0, 0
    core = model.module if ddp else model
    last = run / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location=device, weights_only=False)
        core.load_state_dict(ck["state_dict"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        history, best, start_epoch = ck["history"], ck["best"], ck["epoch"] + 1
        if rank == 0:
            print(f"resumed from epoch {ck['epoch']}", flush=True)

    for epoch in range(start_epoch, tcfg["epochs"]):
        if sampler:
            sampler.set_epoch(epoch)
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for x, y in train_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=tcfg["amp"]):
                lg = model(x)
                loss = F.cross_entropy(lg.float(), y, weight=ce_w, ignore_index=255)
                loss = loss + tcfg["dice_weight"] * dice_loss(lg, y, 3)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot, n = tot + float(loss), n + 1
        if rank == 0:
            val = evaluate(core, eval_loader("val"), device, tcfg["amp"])
            rec = {"epoch": epoch, "loss": round(tot / n, 4), "sec": round(time.time() - t0)} | {
                k: val[k] for k in ("miou", "iou_flood", "f1_flood", "iou_perm_water")
            }
            history.append(rec)
            print(json.dumps(rec), flush=True)
            Path(run / "history.json").write_text(json.dumps(history, indent=1))
            if val["iou_flood"] > best:
                best = val["iou_flood"]
                torch.save(
                    {"state_dict": core.state_dict(), "epoch": epoch, "cfg": cfg}, run / "best.pt"
                )
        if ddp:
            dist.barrier()

    if rank == 0:
        ck = torch.load(run / "best.pt", map_location=device, weights_only=False)
        core.load_state_dict(ck["state_dict"])
        temp = fit_temperature(core, eval_loader("val"), device, tcfg["amp"])
        metrics = {"best_epoch": ck["epoch"], "temperature": round(temp, 4)}
        for name in ("val", "test", "mountain"):
            if splits[name]:
                metrics[name] = evaluate(
                    core, eval_loader(name), device, tcfg["amp"], rows, splits[name], temp
                )
                print(
                    name, {k: v for k, v in metrics[name].items() if k != "per_event"}, flush=True
                )
        Path(run / "metrics.json").write_text(json.dumps(metrics, indent=1))
        export_onnx(core, run, device)
        card = {
            "model": f"{cfg['model']['arch']}-{cfg['model']['encoder']}",
            "run_name": cfg["run_name"],
            "training_data": "Kuro Siwo labelled GRD (Bountos et al., NeurIPS 2024), CC BY",
            "encoder_weights": cfg["model"]["encoder_weights"] or "random init",
            "inputs": "6 x 224 x 224: dB sigma0 [post VV, post VH, pre1 VV, pre1 VH, pre2 VV, pre2 VH], "
            "standardised with norm.json",
            "classes": ["no_water", "permanent_water", "flood"],
            "temperature": round(temp, 4),
            "norm": norm,
            "splits": splits["_meta"],
            "seed": cfg["seed"],
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        Path(run / "model_card.json").write_text(json.dumps(card, indent=1))
    if ddp:
        dist.destroy_process_group()


def export_onnx(model, run: Path, device) -> None:
    model.eval().float()
    dummy = torch.zeros(1, 6, 224, 224, device=device)
    torch.onnx.export(
        model,
        dummy,
        run / "model.onnx",
        input_names=["x"],
        output_names=["logits"],
        dynamic_axes={"x": {0: "n", 2: "h", 3: "w"}, "logits": {0: "n", 2: "h", 3: "w"}},
        opset_version=17,
        dynamo=False,
    )


if __name__ == "__main__":
    main()
