"""ML member: UNet-ResNet50 trained on Kuro Siwo only (training/train.py), run with ONNX Runtime.

Input: σ⁰ (linear) for post, pre1 and pre2 passes on one relative orbit, as dB, standardised with
the training norm. Output: temperature-calibrated class probabilities
(0 no water, 1 permanent water, 2 flood). Sliding 224 px windows with blended overlaps.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from dhara.config import get_settings

MODEL_DIR = Path(__file__).resolve().parents[3] / "models" / "flood_unet"
TILE, STRIDE = 224, 160


@dataclass
class Model:
    session: Any  # onnxruntime.InferenceSession
    mean: np.ndarray
    std: np.ndarray
    temperature: float
    card: dict

    @property
    def card_summary(self) -> dict:
        keys = (
            "model",
            "run_name",
            "training_data",
            "encoder_weights",
            "temperature",
            "created_utc",
        )
        return {k: self.card.get(k) for k in keys}


@dataclass
class MLResult:
    p_flood: np.ndarray
    p_perm_water: np.ndarray


def model_available(model_dir: Path = MODEL_DIR) -> bool:
    return (model_dir / "model.onnx").exists() and (model_dir / "model_card.json").exists()


def load_model(model_dir: Path = MODEL_DIR) -> Model | None:
    onnx, card_path = model_dir / "model.onnx", model_dir / "model_card.json"
    if not (onnx.exists() and card_path.exists()):
        return None
    card = json.loads(card_path.read_text())
    weights = str(card.get("encoder_weights", "random init"))
    # ImageNet encoders are allowed (organizer answer, 2026-10-13); anything else is a gray area.
    if weights not in ("random init", "imagenet") and not get_settings().allow_foundation_weights:
        raise RuntimeError(
            f"Model uses pretrained encoder weights ({weights}); only random init or ImageNet are "
            "allowed by default (set ALLOW_FOUNDATION_WEIGHTS=true to allow others)."
        )
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 0
    sess = ort.InferenceSession(str(onnx), opts, providers=["CPUExecutionProvider"])
    norm = card["norm"]
    return Model(
        session=sess,
        mean=np.array(norm["mean"], np.float32)[:, None, None],
        std=np.array(norm["std"], np.float32)[:, None, None],
        temperature=float(card.get("temperature", 1.0)),
        card=card,
    )


def _window(n: int = TILE) -> np.ndarray:
    w = np.hanning(n + 2)[1:-1]
    return np.outer(w, w).astype(np.float32) + 1e-3


def _starts(size: int) -> list[int]:
    if size <= TILE:
        return [0]
    s = list(range(0, size - TILE, STRIDE))
    return [*s, size - TILE]


def predict(
    model: Model, post: np.ndarray, pre1: np.ndarray, pre2: np.ndarray, batch: int = 8
) -> MLResult:
    """post/pre1/pre2: (2, H, W) linear σ⁰ [VV, VH] on the same grid. Returns probabilities (H, W)."""
    x = np.concatenate([post, pre1, pre2]).astype(np.float32)
    valid = np.isfinite(x).all(0)
    x = 10.0 * np.log10(np.clip(np.nan_to_num(x, nan=1e-5), 1e-5, None))
    x = (x - model.mean) / model.std
    x[:, ~valid] = 0.0
    _, H, W = x.shape
    ph, pw = max(H, TILE), max(W, TILE)
    if (ph, pw) != (H, W):
        x = np.pad(x, ((0, 0), (0, ph - H), (0, pw - W)))
    acc = np.zeros((3, ph, pw), np.float32)
    wsum = np.zeros((ph, pw), np.float32)
    win = _window()
    coords = [(r, c) for r in _starts(ph) for c in _starts(pw)]
    for i in range(0, len(coords), batch):
        chunk = coords[i : i + batch]
        xb = np.stack([x[:, r : r + TILE, c : c + TILE] for r, c in chunk])
        (logits,) = model.session.run(None, {"x": xb})
        z = logits / model.temperature
        z = z - z.max(1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(1, keepdims=True)
        for (r, c), pk in zip(chunk, p, strict=True):
            acc[:, r : r + TILE, c : c + TILE] += pk * win
            wsum[r : r + TILE, c : c + TILE] += win
    prob = acc[:, :H, :W] / wsum[:H, :W]
    prob[:, ~valid] = np.nan
    return MLResult(p_flood=prob[2], p_perm_water=prob[1])
