from ray import tune
import numpy as np

from simulation.environment import GridWorldEnv

env_config = {
    "version": GridWorldEnv.VERSION,
    "width": 60,
    "height": 60,
    "num_survivors": 25,
    "num_agents": 5,
    "start_world": [[]],
    "sight": 8,
    "battery": 200,
    "rewards": {
        "rescue": 1,
        "hit tree": 0
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
                "size": 6,
            },
        }
    }
}

stop = {
    "training_iteration": 300,
    # "episode_reward_mean": 14,
}

config = {
    "env": GridWorldEnv,
    "framework": "torch",
    # "num_cpus_for_driver": 1,
    # "num_envs_per_worker": 1,
    "num_workers": 6,
    "num_gpus": 1,
    # "num_cpus_per_worker": 1,
    # "model": {"fcnet_hiddens": [8, 8]},
    # "train_batch_size": int(4000/8),
    # "rollout_fragment_length": int(200/8),
    # "sgd_minibatch_size": 128,
    "lr": tune.grid_search([0.01, 0.001]),
    "env_config": env_config,

    # "exploration_config": {
    #     "type": "Curiosity",
    #     "eta": 0.1,
    #     "lr": 0.001,
    #     # No actual feature net: map directly from observations to feature
    #     # vector (linearly).
    #     "feature_net_config": {
    #         "fcnet_hiddens": [],
    #         "fcnet_activation": "relu",
    #     },
    #     "sub_exploration": {
    #         "type": "StochasticSampling",
    #     },
    #     "forward_net_activation": "relu",
    #     "inverse_net_activation": "relu",
    # }

}