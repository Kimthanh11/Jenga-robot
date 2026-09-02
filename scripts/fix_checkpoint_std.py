"""Rewrite the action standard deviation stored in an rsl_rl checkpoint."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

KEY = "distribution.log_std_param"


def main() -> None:
    parser = argparse.ArgumentParser(description="Reset the action std in a checkpoint.")
    parser.add_argument("checkpoint")
    parser.add_argument("--std", type=float, default=0.2)
    parser.add_argument("--out", default=None, help="Default: <name>_std<value>.pt")
    args = parser.parse_args()

    src = Path(args.checkpoint)
    data = torch.load(src, map_location="cpu", weights_only=False)
    actor = data.get("actor_state_dict")
    if actor is None or KEY not in actor:
        raise SystemExit(f"{KEY} not found in {src}; keys: {sorted(actor or {})}")

    before = actor[KEY].exp()
    actor[KEY] = torch.full_like(actor[KEY], math.log(args.std))
    out = Path(args.out) if args.out else src.with_name(f"{src.stem}_std{args.std:g}.pt")
    torch.save(data, out)
    print(f"std {[round(float(v), 4) for v in before]} -> {args.std}")
    print(f"iteration {data.get('iter')}, wrote {out}")


if __name__ == "__main__":
    main()
