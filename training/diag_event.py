"""Where do a model's flood errors go on one Kuro Siwo event? For the event's labelled pixels,
the confusion (rows = truth, cols = prediction) per model, plus mean dB of each date for flood pixels.
Usage on the GPU box: python diag_event.py --config ... --event 562 --model name=dir [...]"""

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import train
import yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--event", type=int, required=True)
    ap.add_argument("--model", action="append", required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    rows = train.load_meta(Path(cfg["data_dir"]))
    idx = [i for i, r in enumerate(rows) if r["actid"] == a.event]
    raw = train.KuroSiwo(Path(cfg["data_dir"]), rows, idx, {"mean": [0] * 6, "std": [1] * 6}, None)
    db = np.stack([raw[k][0].numpy() for k in range(len(idx))])
    y = np.stack([raw[k][1].numpy() for k in range(len(idx))])
    fl = y == 2
    names = ["post VV", "post VH", "pre1 VV", "pre1 VH", "pre2 VV", "pre2 VH"]
    print("tiles", len(idx), "flood px", int(fl.sum()), "perm px", int((y == 1).sum()))
    print(
        "mean dB on flood px:",
        {n: round(float(db[:, c][fl].mean()), 2) for c, n in enumerate(names)},
    )
    for spec in a.model:
        name, d = spec.split("=", 1)
        card = json.loads((Path(d) / "model_card.json").read_text())
        sess = ort.InferenceSession(str(Path(d) / "model.onnx"), providers=["CPUExecutionProvider"])
        ds = train.KuroSiwo(Path(cfg["data_dir"]), rows, idx, card["norm"], None)
        pred = np.concatenate(
            [
                sess.run(
                    None,
                    {"x": np.stack([ds[k][0].numpy() for k in range(j, min(j + 16, len(idx)))])},
                )[0].argmax(1)
                for j in range(0, len(idx), 16)
            ]
        )
        m = np.zeros((3, 3), int)
        for t in range(3):
            for p in range(3):
                m[t, p] = int(((y == t) & (pred == p)).sum())
        share = (m / np.maximum(m.sum(1, keepdims=True), 1)).round(3)
        print(
            name,
            "truth flood -> pred [none, perm, flood]:",
            share[2].tolist(),
            "| truth perm ->",
            share[1].tolist(),
            "| truth none ->",
            share[0].tolist(),
        )


if __name__ == "__main__":
    main()
