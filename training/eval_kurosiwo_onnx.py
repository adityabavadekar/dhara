"""Score exported ONNX models on the same Kuro Siwo event splits (test, held-out mountain fold), so
models trained with different splits can be compared on equal terms.

Usage on the GPU box: python eval_kurosiwo_onnx.py --config configs/unet_r34_imagenet.yaml \
    --model old=/path/old_dir --model new=/path/new_dir   (each dir: model.onnx + model_card.json)

--tune: also fit per-class probability weights on the validation events (argmax of w_c * p_c,
grid search, flood F1 on validation only) and score test/mountain with them as "<name>+w".
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import train
import yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", action="append", required=True, help="name=dir")
    ap.add_argument("--out", default="/tmp/kurosiwo_compare.json")
    ap.add_argument(
        "--tune", action="store_true", help="fit class weights on the validation events"
    )
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    rows = train.load_meta(Path(cfg["data_dir"]))
    splits = train.make_splits(rows, cfg)
    report: dict = {"events": splits["_meta"]["events"], "models": {}}
    models = []
    for spec in args.model:
        name, d = spec.split("=", 1)
        card = json.loads((Path(d) / "model_card.json").read_text())
        sess = ort.InferenceSession(
            str(Path(d) / "model.onnx"), providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        models.append((name, card, sess))
    base = [m[0] for m in models] + (["ensemble"] if len(models) > 1 else [])

    def batches(split):
        """Yield (tile indices, {name: class probabilities (B,3,H,W)}, labels) over one split."""
        datasets = {
            name: train.KuroSiwo(Path(cfg["data_dir"]), rows, splits[split], card["norm"], None)
            for name, card, _ in models
        }
        n_tiles = len(splits[split])
        for j in range(0, n_tiles, 16):
            ks = range(j, min(j + 16, n_tiles))
            probs, y = {}, None
            for name, card, sess in models:
                batch = [datasets[name][k] for k in ks]
                x = np.stack([b[0].numpy() for b in batch]).astype(np.float32)
                y = np.stack([b[1].numpy() for b in batch])
                (lg,) = sess.run(None, {"x": x})
                z = lg / float(card.get("temperature", 1.0))
                z = z - z.max(1, keepdims=True)
                p = np.exp(z)
                probs[name] = p / p.sum(1, keepdims=True)
            if len(models) > 1:
                probs["ensemble"] = np.mean([probs[m[0]] for m in models], axis=0)
            yield ks, probs, y

    # Class weights (no water fixed at 1): a coarse log grid, chosen on validation events only.
    weights: dict[str, np.ndarray] = {}
    if args.tune:
        grid = [round(float(v), 3) for v in np.exp(np.linspace(np.log(1 / 3), np.log(3.0), 13))]
        combos = list(itertools.product(grid, grid))  # (permanent water, flood)
        gconf = {n: [train.Confusion() for _ in combos] for n in base}
        for _, probs, y in batches("val"):
            for n in base:
                for c, (wp, wf) in enumerate(combos):
                    w = np.array([1.0, wp, wf], np.float32)[None, :, None, None]
                    gconf[n][c].add((probs[n] * w).argmax(1).ravel(), y.ravel())
        for n in base:
            sc = [c.scores() for c in gconf[n]]
            best = max(range(len(combos)), key=lambda c: sc[c]["f1_flood"] or 0.0)
            ref = combos.index((1.0, 1.0)) if (1.0, 1.0) in combos else None
            weights[n] = np.array([1.0, *combos[best]], np.float32)
            report.setdefault("tuning", {})[n] = {
                "weights": [1.0, *combos[best]],
                "val_f1_flood": sc[best]["f1_flood"],
                "val_f1_perm_water": sc[best]["f1_perm_water"],
                "val_f1_flood_unweighted": sc[ref]["f1_flood"] if ref is not None else None,
            }
            print(n, "tuned on val", report["tuning"][n], flush=True)
    names = base + [f"{n}+w" for n in weights]
    for split in ("test", "mountain"):
        conf = {n: train.Confusion() for n in names}
        per: dict = {n: {} for n in names}
        for ks, probs, y in batches(split):
            for n in names:
                if n.endswith("+w"):
                    pred = (probs[n[:-2]] * weights[n[:-2]][None, :, None, None]).argmax(1)
                else:
                    pred = probs[n].argmax(1)
                conf[n].add(pred.ravel(), y.ravel())
                for s_, k in enumerate(ks):
                    a = rows[splits[split][k]]["actid"]
                    per[n].setdefault(a, train.Confusion()).add(pred[s_].ravel(), y[s_].ravel())
        for n in names:
            ev = {str(a): c.scores()["f1_flood"] for a, c in sorted(per[n].items())}
            f1s = [v for v in ev.values() if v is not None]
            res = conf[n].scores() | {
                "event_mean_f1_flood": round(float(np.mean(f1s)), 4),
                "event_f1_flood_ci95": train.bootstrap_ci(f1s),
                "per_event_f1_flood": ev,
            }
            report["models"].setdefault(n, {})[split] = res
            print(
                n,
                split,
                {
                    k: res[k]
                    for k in (
                        "f1_flood",
                        "iou_flood",
                        "precision_flood",
                        "recall_flood",
                        "f1_perm_water",
                        "event_mean_f1_flood",
                        "event_f1_flood_ci95",
                    )
                },
                flush=True,
            )
            print("   per event", ev, flush=True)
    Path(args.out).write_text(json.dumps(report, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
