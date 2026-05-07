def apply_preset(args):
    if args.preset == "none":
        return args

    default_args = type(args)()

    if args.preset == "docking2d_saferl":
        preset_values = {
            "env_id": "Docking2d",
            "total_timesteps": 400_000,  # 800_000 = 200 iterations * train_batch_size 4000 in SafeRL PPO setup
            "learning_rate": 5e-5,
            "num_envs": 4,
            "num_steps": 1000,  # batch_size 4000 (close to SafeRL train_batch_size)
            "anneal_lr": False,
            "gamma": 0.968559,
            "gae_lambda": 0.928544,
            "num_minibatches": 32,  # minibatch_size 125 (close to SafeRL sgd_minibatch_size=128)
            "update_epochs": 30,  # SafeRL num_sgd_iter=30
            "clip_coef": 0.3,
            "ent_coef": 0.0,
            "vf_coef": 1.0,
            "target_kl": 0.01,
        }
        # Apply preset only for fields left at parser defaults.
        for key, value in preset_values.items():
            if getattr(args, key) == getattr(default_args, key):
                setattr(args, key, value)
        return args

    raise ValueError(f"Unknown preset: {args.preset}")
