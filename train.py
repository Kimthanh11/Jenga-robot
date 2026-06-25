import os


def _maybe_force_cpu() -> None:
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        return
    try:
        import torch

        if torch.cuda.is_available():
            return
    except Exception:
        pass
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


if __name__ == "__main__":
    _maybe_force_cpu()
    import mjlab_jenga
    from mjlab.scripts.train import main

    main()
