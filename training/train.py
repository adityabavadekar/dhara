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
import torch.nn as nn
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
    mountain = cfg["splits"].get("mountain_events") or sorted(relief, key=lambda a: -relief[a])[:k]
    sp = cfg["splits"]
    out: dict[str, list[int]] = {"train": [], "val": [], "test": [], "mountain": []}
    for idx, r in enumerate(rows):
        a = r["actid"]
        if r.get("source") == "sen1floods11":
            out["train"].append(
                idx
            )  # Sen1Floods11 train/valid chips; its test and Bolivia stay out
            continue
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
    missing = [s for s in ("train", "val", "test", "mountain") if not out[s]]
    if missing:
        # No silent tile-level fallback: scoring on tiles from training events is meaningless.
        present = sorted({r["actid"] for r in rows})
        raise RuntimeError(
            f"split(s) {missing} are empty with the events present {present}; convert more "
            "Kuro Siwo shards (prep_kurosiwo.py --max-shards 99) or edit the event lists."
        )
    cap = int(sp.get("max_tiles_per_event") or 0)
    if cap:
        # Balance events so a few large activations do not dominate training or the scores.
        rng = random.Random(cfg["seed"])
        for name, idx in out.items():
            by: dict[int, list[int]] = {}
            for i in idx:
                by.setdefault(rows[i]["actid"], []).append(i)
            out[name] = sorted(
                j for v in by.values() for j in (rng.sample(v, cap) if len(v) > cap else v)
            )

    out_meta = {
        "event_relief_m": {str(a): round(v, 1) for a, v in sorted(relief.items())},
        "mountain_events": mountain,
        "counts": {s: len(v) for s, v in out.items()},
        "events": {s: sorted({rows[i]["actid"] for i in v}) for s, v in out.items()},
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
        """Per chunk: (path, data offset, sample shape, dtype) for x and y. Samples are read with
        plain file reads, not memory maps: with mmap every loader worker gradually maps the whole
        dataset, and the container's memory use grows epoch after epoch."""
        if tag not in self._x:
            for store, kind in ((self._x, "x"), (self._y, "y")):
                path = self.data_dir / f"{tag}_{kind}.npy"
                with open(path, "rb") as fh:
                    version = np.lib.format.read_magic(fh)
                    read = (
                        np.lib.format.read_array_header_1_0
                        if version == (1, 0)
                        else np.lib.format.read_array_header_2_0
                    )
                    shape, _, dtype = read(fh)
                    store[tag] = (str(path), fh.tell(), tuple(shape[1:]), np.dtype(dtype))
        return self._x[tag], self._y[tag]

    @staticmethod
    def _read(spec, i: int) -> np.ndarray:
        path, off, shape, dtype = spec
        n = int(np.prod(shape)) * dtype.itemsize
        fd = os.open(path, os.O_RDONLY)
        try:
            buf = os.pread(fd, n, off + i * n)
            os.posix_fadvise(
                fd, off + i * n, n, os.POSIX_FADV_DONTNEED
            )  # do not keep in page cache
        finally:
            os.close(fd)
        return np.frombuffer(buf, dtype=dtype).reshape(shape)

    @staticmethod
    def _db(raw: np.ndarray) -> np.ndarray:
        """float dB from a stored tile: float16 as is, uint8 as quantised by prep_kurosiwo.py."""
        if raw.dtype == np.uint8:
            return raw.astype(np.float32) / 4.0 - 45.0  # DB_STEPS = 4, DB_MIN = -45
        return raw.astype(np.float32)

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, j: int):
        r = self.rows[self.idx[j]]
        xs, ys = self._arr(r["chunk"])
        x = self._db(self._read(xs, r["i"]))  # (6,H,W) dB: post, pre1, pre2 × (VV,VH)
        y = self._read(ys, r["i"]).astype(np.int64)
        a = self.aug
        if a:
            if random.random() < a["swap_pre"]:
                x[[2, 3, 4, 5]] = x[[4, 5, 2, 3]]
            if random.random() < a["pre2_dropout"]:
                x[[4, 5]] = x[[2, 3]]
            if random.random() < a.get("no_pre", 0.0):
                # No usable pre-event pass (e.g. Sen1Floods11): both pre slots = the post image.
                # The target stays the same, so the model learns water vs not without change.
                x[2:6] = np.concatenate([x[0:2], x[0:2]])
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


class PhysicsGuidedSARModel(nn.Module):
    """Deep Neural Segmentation Network augmented with a Physics-Informed SAR Residual Head.

    Extracts explicit physical differential radar features:
      1. Specular backscatter drop Δσ0_VV = Post VV - Pre1 VV
      2. Cross-pol volume scattering ratio Δσ0_VH = Post VH - Pre1 VH
      3. Co/Cross polarization ratio = Post VV - Post VH (dB)
    and passes them through a physics gating residual unit to refine the deep feature logits.
    """

    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base = base_model
        # Physics feature extractor: 3 physical radar channels
        self.physics_head = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, kernel_size=1),
        )
        self.gate = nn.Parameter(torch.tensor([0.2]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is (B, 6, H, W): [post_vv, post_vh, pre1_vv, pre1_vh, pre2_vv, pre2_vh]
        delta_vv = x[:, 0:1] - x[:, 2:3]
        delta_vh = x[:, 1:2] - x[:, 3:4]
        pol_ratio = x[:, 0:1] - x[:, 1:2]
        physics_feat = torch.cat([delta_vv, delta_vh, pol_ratio], dim=1)

        base_logits = self.base(x)
        physics_residual = self.physics_head(physics_feat)

        return base_logits + self.gate * physics_residual


def build_model(cfg: dict) -> nn.Module:
    import segmentation_models_pytorch as smp

    m = cfg["model"]
    # ImageNet encoders are allowed (organizer answer, 2026-10-13); other weights need the flag.
    if m["encoder_weights"] not in (None, "imagenet") and not m["allow_pretrained_weights"]:
        raise RuntimeError(
            f"Encoder weights {m['encoder_weights']!r} are not allowed; set allow_pretrained_weights"
        )
    arch_name = m.get("arch", "Unet")
    model_cls = getattr(smp, arch_name, None)
    if model_cls is None:
        for name in dir(smp):
            if name.lower() == arch_name.lower():
                model_cls = getattr(smp, name)
                break
    if model_cls is None:
        model_cls = smp.Unet
    base_model = model_cls(
        encoder_name=m["encoder"],
        encoder_weights=m["encoder_weights"],
        in_channels=m["in_channels"],
        classes=m["classes"],
    )
    if m.get("physics_head", False):
        return PhysicsGuidedSARModel(base_model)
    return base_model


WATER_ANY = 3  # label for "water, flood or permanent unknown" (Sen1Floods11 pixels)


def any_water_loss(logits, y) -> torch.Tensor:
    """-log(p_perm + p_flood) on pixels labelled WATER_ANY, summed and divided by all labelled
    pixels in the batch, so each pixel weighs as much as in the cross-entropy term. (Averaging over
    the WATER_ANY pixels alone, as v5 did, made a few pixels outweigh the batch and taught the
    model to call most post-only input water.)"""
    m = y == WATER_ANY
    if not m.any():
        return logits.sum() * 0.0
    logp = logits.float().log_softmax(1)
    lw = torch.logsumexp(logp[:, 1:3], dim=1)
    return -lw[m].sum() / (y != 255).sum().clamp(min=1)


def dice_loss(logits, y, n_cls: int):
    valid = (y != 255).unsqueeze(1)
    p = logits.float().softmax(1) * valid
    oh = F.one_hot(y.clamp(max=n_cls - 1), n_cls).permute(0, 3, 1, 2).float() * valid
    inter = (p * oh).sum((0, 2, 3))
    den = p.sum((0, 2, 3)) + oh.sum((0, 2, 3))
    return 1 - ((2 * inter + 1) / (den + 1)).mean()


def focal_loss(logits, y, n_cls: int, gamma: float = 2.0, alpha=None):
    valid = y != 255
    log_p = F.log_softmax(logits.float(), dim=1)
    p = torch.exp(log_p)
    target = y.clamp(max=n_cls - 1)
    ce = F.nll_loss(log_p, target, weight=alpha, ignore_index=255, reduction="none")
    p_t = p.gather(1, target.unsqueeze(1)).squeeze(1)
    focal_weight = (1.0 - p_t) ** gamma
    return (focal_weight * ce)[valid].mean()


def lovasz_softmax_loss(logits, y, n_cls: int):
    """Lovász-softmax (Berman et al., CVPR 2018): a convex surrogate of per-class IoU, averaged
    over the classes present in the batch. Pixels labelled 255 are ignored."""
    p = logits.float().softmax(1).permute(0, 2, 3, 1).reshape(-1, n_cls)
    t = y.reshape(-1)
    v = t != 255
    p, t = p[v], t[v]
    losses = []
    for c in range(n_cls):
        fg = (t == c).float()
        if fg.sum() == 0:
            continue
        err, order = (fg - p[:, c]).abs().sort(descending=True)
        g = fg[order]
        gts = g.sum()
        jac = 1.0 - (gts - g.cumsum(0)) / (gts + (1.0 - g).cumsum(0))
        jac = torch.cat([jac[:1], jac[1:] - jac[:-1]])
        losses.append(torch.dot(err, jac))
    return torch.stack(losses).mean() if losses else logits.sum() * 0.0


def focal_tversky_loss(
    logits, y, classes=(1, 2), w_fn: float = 0.7, w_fp: float = 0.3, gamma: float = 0.75
):
    """Focal Tversky (Abraham & Khan 2019) on the water classes: Tversky index TP / (TP + w_fn FN
    + w_fp FP), w_fn > w_fp so missed water costs more than false water; (1 - TI)^gamma."""
    valid = (y != 255).unsqueeze(1).float()
    p = logits.float().softmax(1) * valid
    out = []
    for c in classes:
        pc, gc = p[:, c], (y == c).float()
        tp = (pc * gc).sum()
        fn = ((1 - pc) * gc * valid[:, 0]).sum()
        fp = (pc * (1 - gc)).sum()
        ti = (tp + 1.0) / (tp + w_fn * fn + w_fp * fp + 1.0)
        out.append((1.0 - ti) ** gamma)
    return torch.stack(out).mean()


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
                # Pixels labelled WATER_ANY (Sen1Floods11) are scored by any_water_loss only.
                y3 = torch.where(y == WATER_ANY, torch.full_like(y, 255), y)
                loss = F.cross_entropy(lg.float(), y3, weight=ce_w, ignore_index=255)
                loss = loss + tcfg["dice_weight"] * dice_loss(lg, y3, 3)
                loss = loss + tcfg.get("any_water_weight", 1.0) * any_water_loss(lg, y)
                if tcfg.get("focal_weight", 0.0) > 0:
                    loss = loss + tcfg["focal_weight"] * focal_loss(lg, y3, 3, alpha=ce_w)
                if tcfg.get("lovasz_weight", 0.0) > 0:
                    loss = loss + tcfg["lovasz_weight"] * lovasz_softmax_loss(lg, y3, 3)
                if tcfg.get("tversky_weight", 0.0) > 0:
                    loss = loss + tcfg["tversky_weight"] * focal_tversky_loss(lg, y3)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scale_before = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            scale_after = scaler.get_scale()
            if scale_before <= scale_after:
                sched.step()
            tot, n = tot + float(loss.detach().item()), n + 1
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
            torch.save(
                {
                    "state_dict": core.state_dict(),
                    "epoch": epoch,
                    "cfg": cfg,
                    "opt": opt.state_dict(),
                    "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(),
                    "history": history,
                    "best": best,
                },
                run / "last.pt",
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
