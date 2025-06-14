# komut: python inference.py --json_path cevaplar.json --model_path model.pth

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

def build_mlp_from_state(state: dict, n_classes: int) -> nn.Module:
    """
    reconstruct the exact-layer MLP purely from weight shapes
    expects state dict keys like 'net.0.weight', 'net.0.bias', ...
    """
    # pull ordered weight/bias pairs
    layers, idx = [], 0
    while f"net.{idx}.weight" in state:
        w = state[f"net.{idx}.weight"]
        b = state[f"net.{idx}.bias"]
        in_dim, out_dim = w.shape[1], w.shape[0]
        layers.append(nn.Linear(in_dim, out_dim))
        if f"net.{idx+1}" in state:  # there’s an activation after every linear except last
            layers.append(nn.ReLU())
        idx += 2  # skip over the following activation in the original sequential
    model = nn.Sequential(*layers)
    # register param shapes then load_full state
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def load_ckpt(model_path: Path) -> Tuple[nn.Module, List[str], List[str]]:
    ckpt = torch.load(model_path, map_location="cpu")
    labels: List[str] = json.loads(ckpt["label_encoder"])
    feats: List[str] = json.loads(ckpt["feature_cols"])
    model = build_mlp_from_state(ckpt["model_state_dict"], len(labels))
    return model, labels, feats


def make_vec(traits: Dict[str, float], cols: List[str]) -> torch.Tensor:
    return torch.tensor([[traits.get(col, 0.0) for col in cols]], dtype=torch.float32)


def trait_blurb(traits: Dict[str, float]) -> str:
    ranked = sorted(traits.items(), key=lambda kv: kv[1], reverse=True)[:2]
    return f"high {ranked[0][0]} & {ranked[1][0]}" if ranked else "n/a"


def predict(model_path: Path, json_path: Path, top_k: int) -> None:
    traits = json.loads(Path(json_path).read_text())
    model, labels, cols = load_ckpt(model_path)

    with torch.no_grad():
        probs = torch.softmax(model(make_vec(traits, cols)), dim=1).numpy().flatten()

    top_idx = probs.argsort()[-top_k:][::-1]
    print("recommended careers →")
    for rank, idx in enumerate(top_idx, 1):
        print(f"{rank}. {labels[idx]} — {trait_blurb(traits)} (p={probs[idx]:.2f})")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="career inference from RIASEC traits")
    p.add_argument("--json_path", type=Path, required=True)
    p.add_argument("--model_path", type=Path, default="model.pth")
    p.add_argument("--top_k", type=int, default=3)
    args = p.parse_args()

    predict(args.model_path, args.json_path, args.top_k)
