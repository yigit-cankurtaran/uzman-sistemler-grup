#!/usr/bin/env python3
# inference.py – dead-simple, mismatch-proof loader/predictor

import argparse, json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn


# ---------- util helpers ---------------------------------------------------


def _weight_keys(state: dict) -> List[str]:
    """grab all *.weight keys in order"""
    return [k for k in state if k.endswith(".weight")]


def _build_mlp(state: dict) -> nn.Module:
    """rebuild exact mlp from weight shapes"""
    layers = []
    w_keys = _weight_keys(state)
    for i, k in enumerate(w_keys):
        w = state[k]
        in_dim, out_dim = w.shape[1], w.shape[0]
        layers.append(nn.Linear(in_dim, out_dim))
        if i < len(w_keys) - 1:  # no activation after final linear
            layers.append(nn.ReLU())
    net = nn.Sequential(*layers)
    net.load_state_dict(state, strict=False)
    net.eval()
    return net


def _dict_to_list(raw: dict) -> List[str]:
    """convert any {label:idx} or {idx:label} into idx-ordered list"""
    if all(isinstance(v, int) for v in raw.values()):  # label → id
        n = max(raw.values()) + 1
        out = [""] * n
        for lbl, idx in raw.items():
            out[idx] = lbl
    else:                                              # id → label
        n = max(map(int, raw.keys())) + 1
        out = [""] * n
        for idx, lbl in raw.items():
            out[int(idx)] = lbl
    return out


# ---------- checkpoint loader ----------------------------------------------


def load_ckpt(path: Path) -> Tuple[nn.Module, List[str], List[str]]:
    ckpt = torch.load(path, map_location="cpu")

    # labels
    raw_enc = json.loads(ckpt["label_encoder"])
    labels: List[str] = raw_enc if isinstance(raw_enc, list) else _dict_to_list(raw_enc)

    # features
    feat_cols: List[str] = json.loads(ckpt["feature_cols"])

    # model
    state = ckpt["model_state_dict"]
    model = _build_mlp(state)

    # pad labels to net output size
    out_dim = state[_weight_keys(state)[-1]].shape[0]
    if len(labels) < out_dim:
        labels += [f"class_{i}" for i in range(len(labels), out_dim)]

    return model, labels, feat_cols


# ---------- vectorizer & pretty print --------------------------------------


def make_vec(traits: Dict[str, float], cols: List[str]) -> torch.Tensor:
    return torch.tensor([[traits.get(c, 0.0) for c in cols]], dtype=torch.float32)


def blurb(traits: Dict[str, float]) -> str:
    top2 = sorted(traits.items(), key=lambda kv: kv[1], reverse=True)[:2]
    return f"high {top2[0][0]} & {top2[1][0]}" if len(top2) == 2 else "n/a"


# ---------- main predict fn -------------------------------------------------


def predict(model_path: Path, json_path: Path, top_k: int) -> None:
    traits = json.loads(Path(json_path).read_text())
    model, labels, feat_cols = load_ckpt(model_path)

    with torch.no_grad():
        probs = torch.softmax(model(make_vec(traits, feat_cols)), dim=1).numpy().ravel()

    # pad labels *again* if trainer later expanded head
    if len(labels) < probs.size:
        labels.extend(f"class_{i}" for i in range(len(labels), probs.size))

    SOC_MAJOR = {
        11: "management",
        13: "biz / finance",
        15: "comp & math",
        17: "architecture / engineering",
        19: "science",
        21: "community & social svc",
        23: "legal",
        25: "education / library",
        27: "arts / media",
        29: "healthcare",
        31: "health support",
        33: "protective service",
        35: "food prep & serving",
        37: "building / grounds",
        39: "personal care",
        41: "sales",
        43: "office / admin",
        45: "farming / fishing",
        47: "construction",
        49: "install / repair",
        51: "production",
        53: "transportation",
    }

    top_idx = probs.argsort()[-top_k:][::-1]
    print("recommended careers →")
    for rank, idx in enumerate(top_idx, 1):
        nice = SOC_MAJOR.get(int(labels[idx]), labels[idx])
        print(f"{rank}. {nice} — {blurb(traits)} (p={probs[idx]:.2f})")


# ---------- cli -------------------------------------------------------------


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="career inference on weighted-RIASEC")
    ap.add_argument("--json_path", type=Path, required=True)
    ap.add_argument("--model_path", type=Path, default="model.pth")
    ap.add_argument("--top_k", type=int, default=3)
    args = ap.parse_args()
    predict(args.model_path, args.json_path, args.top_k)
