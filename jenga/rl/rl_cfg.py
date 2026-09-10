from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)

def jenga_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(64, 64),
      activation="elu",
      # The critic reads 81 raw block coordinates alongside joint angles and contact
      # forces, which span very different scales.
      obs_normalization=True,
      # The standard deviation is fixed rather than learned.
      #
      # rsl_rl defaults std_range to (1e-6, 1e6) and constrains std with torch.clamp,
      # whose gradient is zero outside the range. Either bound is therefore absorbing:
      # once the entropy bonus drives log_std_param past log(1.0) no gradient returns
      # it. Because clip_actions is applied after sampling, PPO continues to evaluate
      # log-probabilities of samples that all execute as the same boundary action.
      #
      # Unbounded, this produced standard deviations of 117 and 743, a mean action norm
      # of exactly 2.0 -- the maximum for four actions in [-1, 1] -- and an episode
      # return of exactly -0.00005 * 4 * 2000 = -0.40, that is the action-magnitude
      # penalty alone.
      #
      # At std 1.0 with actions in [-1, 1] roughly a third of samples per dimension are
      # clipped. 0.2 is the value the policy converged to when the entropy term was not
      # dominating.
      #
      # Note for resumed runs: load_state_dict overwrites init_std with the value
      # stored in the checkpoint, and requires_grad=False then pins it. Use
      # fix_checkpoint_std.py before resuming from a checkpoint with a saturated std.
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 0.2,
        "std_type": "log",
        "std_range": (0.05, 1.0),
        "learn_std": False,
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(64, 64),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      # No effect while the policy standard deviation is fixed: the entropy of a
      # Gaussian depends only on its standard deviation, so with learn_std=False this
      # term contributes a constant and its gradient is identically zero. Retained so
      # that the value is defined if std is made learnable again; 0.002 was the
      # largest setting under which the policy gradient still pulled std down rather
      # than the entropy bonus pushing it into the clip region.
      entropy_coef=0.002,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="jenga",
    clip_actions=1.0,
    # A 12-hour job reaches roughly 900-1400 iterations, so saving every 500 threw
    # away 411 iterations when the first run was cut off at 911. This task needs
    # several chained resume jobs to finish its curriculum, so that loss compounds.
    # Checkpoints are ~220 KB.
    save_interval=100,
    num_steps_per_env=32,
    max_iterations=10000,
  )