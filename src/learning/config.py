from ray import tune


stop = {
    # "training_iteration": 1000,
    "episode_reward_mean": 19,
    "timesteps_total": 5_000_000
}

performance_configs = {
    "laptop": {
        # Performance config
        # "num_cpus_for_driver": 1,
        "num_envs_per_worker": 1,
        "num_workers": 10,
        "num_gpus": 1,
    }, "iridis": {
        # Performance config
        # "num_cpus_for_driver": 1,
        "num_envs_per_worker": 1,
        "num_workers": 13,
        "num_gpus": 1,
    },
}

config = {
    # RLlib configurations for all models
    "common": {
        "framework": "torch",
    },
    # RLlib configurations for ppo trainer
    "ppo": {
        "config": {

            # Model config
            "model": {
                "dim": 11,
                "conv_filters": [[16, [3, 3], 2], [32, [3, 3], 2], [512, [3, 3], 1]],
                "use_lstm": True,
                # To further customize the LSTM auto-wrapper.
                "lstm_cell_size": 64,
            },

            # Trainer parameters
            "lr": tune.loguniform(0.00001, 0.001),
            "lambda": tune.uniform(0.9, 1),
            "gamma": 0.99,
            "rollout_fragment_length": 100,
            "train_batch_size": 5000,
            "sgd_minibatch_size": 500,
            "entropy_coeff": 0.01
        },

        # For population based trainer
        "mutations_config": {
            "lr": tune.loguniform(0.00001, 0.001),
            "lambda": tune.uniform(0.9, 1),
            "gamma": tune.uniform(0.95, 1),

            "rollout_fragment_length": [20, 100, 200],
            "train_batch_size": [4000, 5000],
            "sgd_minibatch_size": [128, 500],

            "observation_filter": ["MeanStdFilter", "NoFilter"]
        }
    }
}
