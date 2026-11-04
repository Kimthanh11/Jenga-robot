"""Pilot RSL-RL configuration for discrete high-level block selection."""

from dataclasses import asdict

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


def high_level_ppo_runner_cfg(num_block_actions: int) -> RslRlOnPolicyRunnerCfg:
    """Return the first serial masked-categorical PPO configuration."""
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            class_name="mjlab_jenga.masked_categorical:MaskedCategoricalModel",
            hidden_dims=(256, 256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": (
                    "mjlab_jenga.masked_categorical:CategoricalDistribution"
                ),
                "num_categories": num_block_actions,
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(256, 256, 128),
            activation="elu",
            obs_normalization=True,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=5,
            # The initial serial implementation has only one environment. One
            # mini-batch avoids empty or tiny subdivisions of the rollout.
            num_mini_batches=1,
            learning_rate=3.0e-4,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        obs_groups={
            "actor": ("tower_state",),
            "critic": ("tower_state",),
        },
        experiment_name="jenga_high_level",
        run_name="serial_raw_state",
        logger="tensorboard",
        upload_model=False,
        save_interval=10,
        num_steps_per_env=8,
        max_iterations=100,
    )


def high_level_ppo_train_dict(num_block_actions: int) -> dict:
    """Return an RSL-RL-ready dictionary without MJLab's optional ``None`` fields."""
    cfg = asdict(high_level_ppo_runner_cfg(num_block_actions))
    for model_name in ("actor", "critic"):
        model_cfg = cfg[model_name]
        for optional_name in ("cnn_cfg", "distribution_cfg"):
            if model_cfg.get(optional_name) is None:
                model_cfg.pop(optional_name, None)
        if model_cfg.get("rnn_type") is None:
            for recurrent_name in ("rnn_type", "rnn_hidden_dim", "rnn_num_layers"):
                model_cfg.pop(recurrent_name, None)
    cfg["algorithm"].setdefault("rnd_cfg", None)
    cfg["algorithm"].setdefault("symmetry_cfg", None)
    cfg.setdefault("multi_gpu", None)
    return cfg
