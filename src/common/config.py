from ray import tune
import numpy as np

from simulation.environment import GridWorldEnv

env_config = {
    "version": GridWorldEnv.VERSION,
    "width": 40,
    "height": 40,
    "num_survivors": 20,
    "num_agents": 3,
    "start_world": [[]],
    "sight": 5,
    "battery": 500,
    "rewards": {
        "rescue": 1,
        "hit tree": 0,
        "exploring": 0.01
    },
    "battery costs": {
        "rotate left": 1,
        "rotate right": 1,
        "advance": 2
    },
    "fire spread": {
        "starting points": 5,
        "covariance": [[3, 0], [0, 3]],
        "rate": 0.1,
    },
    "autogen config": {
        "forest fire": {
            "chance": 1,
            "trees": {
                "scale": 20.0,
                "octaves": 8,
                "persistence": 0.5,
                "lacunarity": 2.0,
                "threshold": 0.07
            },
            "rocks": {
                "scale": 6.0,
                "octaves": 10,
                "persistence": 0.5,
                "lacunarity": 5.0,
                "threshold": 0.20
            },
            "hq": {
                "size": 5,
            },
        }
    }
}

stop = {
    "training_iteration": 1000,
    # "episode_reward_mean": 14,
}

config = {
    "env": GridWorldEnv,
    "framework": "torch",
    "env_config": env_config,

    # Performance config
    # "num_cpus_for_driver": 1,
    "num_envs_per_worker": 1,
    "num_workers": 9,
    "num_gpus": 1,

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
}

mutations_config = {
    "lr": tune.loguniform(0.00001, 0.001),
    "lambda": tune.uniform(0.9, 1),
    "gamma": tune.uniform(0.95, 1),

    "rollout_fragment_length": [20, 100, 200],
    "train_batch_size": [4000, 5000],
    "sgd_minibatch_size": [128, 500],

    "observation_filter": ["MeanStdFilter", "NoFilter"]
}
