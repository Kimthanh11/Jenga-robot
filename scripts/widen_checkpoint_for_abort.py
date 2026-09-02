"""Add the abort action dimension to a trained checkpoint."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

OUTPUT_WEIGHT = "mlp.4.weight"
OUTPUT_BIAS = "mlp.4.bias"
LOG_STD = "distribution.log_std_param"


def _widen(tensor: torch.Tensor, fill: float) -> torch.Tensor:
    """Append one row (or element) to the leading dimension."""
    extra_shape = (1,) + tuple(tensor.shape[1:])
    extra = torch.full(extra_shape, fill, dtype=tensor.dtype, device=tensor.device)
    return torch.cat([tensor, extra], dim=0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add one action dimension to a checkpoint for the abort task."
    )
    parser.add_argument("checkpoint", help="Path to the checkpoint to widen.")
    parser.add_argument(
        "--std",
        type=float,
        default=0.2,
        help="Standard deviation for the new action dimension.",
    )
    parser.add_argument(
        "--bias",
        type=float,
        default=0.0,
        help="Bias of the new output unit. 0 keeps the abort signal centred; a "
        "negative value adds margin to the abort threshold.",
    )
    parser.add_argument("--out", default=None, help="Output path.")
    args = parser.parse_args()

    src = Path(args.checkpoint)
    dst = Path(args.out) if args.out else src.with_name(src.stem + "_abort" + src.suffix)

    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    actor = ckpt["actor_state_dict"]

    before = tuple(actor[OUTPUT_WEIGHT].shape)
    if before[0] != actor[LOG_STD].shape[0]:
        raise SystemExit(
            f"inconsistent checkpoint: {OUTPUT_WEIGHT} has {before[0]} outputs but "
            f"{LOG_STD} has {actor[LOG_STD].shape[0]}"
        )

    old_dim = before[0]

    actor[OUTPUT_WEIGHT] = _widen(actor[OUTPUT_WEIGHT], 0.0)
    actor[OUTPUT_BIAS] = _widen(actor[OUTPUT_BIAS], args.bias)
    actor[LOG_STD] = _widen(actor[LOG_STD], math.log(args.std))

    widened_state = 0
    optimizer = ckpt.get("optimizer_state_dict")
    if optimizer is not None:
        for entry in optimizer["state"].values():
            for key, value in entry.items():
                if torch.is_tensor(value) and value.dim() > 0 and value.shape[0] == old_dim:
                    # Zero moment for the new unit: it has no update history.
                    entry[key] = _widen(value, 0.0)
                    widened_state += 1
        # exp_avg and exp_avg_sq for each of the three parameters.
        if widened_state != 6:
            raise SystemExit(
                f"expected to widen 6 optimizer tensors (exp_avg and exp_avg_sq for "
                f"the output weight, output bias and log std), widened {widened_state}. "
                f"The architecture differs from the one this script was written for; "
                f"inspect optimizer_state_dict before continuing."
            )

    torch.save(ckpt, dst)

    print(f"read  {src}")
    print(f"  {OUTPUT_WEIGHT:28} {before} -> {tuple(actor[OUTPUT_WEIGHT].shape)}")
    print(f"  {OUTPUT_BIAS:28} -> {tuple(actor[OUTPUT_BIAS].shape)}, new value {args.bias}")
    print(f"  {LOG_STD:28} -> {tuple(actor[LOG_STD].shape)}, new std {args.std}")
    print(f"  critic unchanged, observation normalizer unchanged")
    if optimizer is None:
        print(f"  optimizer state absent")
    else:
        print(f"  optimizer state: {widened_state} moment tensors widened, "
              f"new entries zeroed")
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
