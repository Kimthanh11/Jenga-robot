"""Rewrite the action standard deviation stored in an rsl_rl checkpoint.

rsl_rl constrains the policy standard deviation with torch.clamp, whose gradient is
zero outside the permitted range. Once the entropy bonus drives log_std_param past the
upper bound, the parameter cannot recover: no gradient returns it to the interior, a
substantial fraction of sampled actions is clipped by clip_actions, and PPO continues
to evaluate log-probabilities on the unclipped samples.

Setting learn_std=False does not repair this on resume. The configured init_std applies
only when the distribution is constructed; load_state_dict then overwrites it with the
value stored in the checkpoint, and requires_grad=False pins it there permanently.

This script rewrites that single parameter so training can continue from an otherwise
healthy policy. The mean network and the optimizer state are left untouched.

Verify the effect in the training log: `Mean action std` must report the new value from
the first iteration onwards.
"""

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
