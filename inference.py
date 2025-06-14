import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn

TRAITS: List[str] = [
    "realistic",
    "investigative",
    "artistic",
    "social",
    "enterprising",
    "conventional",
]


class MLP(nn.Module):
    """same architecture as in training"""

    def __init__(self, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.net(x)


def load_model(model_path: Path) -> tuple[nn.Module, List[str]]:
    ckpt = torch.load(model_path, map_location="cpu")
    labels = json.loads(ckpt["label_encoder"])
    model = MLP(num_classes=len(labels))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, labels


def make_vector(traits: Dict[str, float]) -> torch.Tensor:
    # order‑consistent vector, fill missing keys with 0.0
    vec = [traits.get(trait, 0.0) for trait in TRAITS]
    return torch.tensor([vec], dtype=torch.float32)


def explain_top_traits(traits: Dict[str, float]) -> str:
    ranked = sorted(traits.items(), key=lambda kv: kv[1], reverse=True)[:2]
    top_names = [name for name, _ in ranked]
    return f"high {top_names[0]} and {top_names[1]}"


def predict(
    model_path: Path,
    json_path: Path,
    top_k: int = 3,
) -> None:
    traits = json.loads(Path(json_path).read_text())
    model, labels = load_model(model_path)
    with torch.no_grad():
        logits = model(make_vector(traits))
        probs = torch.softmax(logits, dim=1).numpy().flatten()
    top_idx = probs.argsort()[-top_k:][::-1]
    print("recommended careers →")
    for rank, idx in enumerate(top_idx, 1):
        print(f"{rank}. {labels[idx]} — {explain_top_traits(traits)} (p={probs[idx]:.2f})")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="career inference from rias ec traits")
    p.add_argument("--json_path", type=Path, required=True)
    p.add_argument("--model_path", type=Path, default="model.pth")
    p.add_argument("--top_k", type=int, default=3)
    args = p.parse_args()
    predict(args.model_path, args.json_path, args.top_k)
