from __future__ import annotations

# =====================================================================================
# Rewrite the action standard deviation stored in a checkpoint.
#
# rsl_rl clamps std with torch.clamp, whose gradient is zero outside the range. Once
# the entropy bonus pushes log_std_param past the upper bound the parameter is dead:
# no gradient can bring it back, a third of sampled actions clip, and PPO keeps
# scoring the unclipped values. Setting learn_std=False does not help on a resume,
# because load_state_dict restores the pinned value from the checkpoint.
#
# This rewrites that one number so a run can be continued from an otherwise healthy
# policy. The mean network is untouched -- it was never the broken part.
# =====================================================================================

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
