from ray import tune
import numpy as np

from simulation.environment import GridWorldEnv

env_config = {
    "version": GridWorldEnv.VERSION,
    "width": 40,
    "height": 40,
    "num_survivors": 20,
    "num_agents": 2,
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
    "num_cpus_for_driver": 1,
    # "num_envs_per_worker": 1,
    "num_workers": 13,
    "num_gpus": 1,
    # "num_cpus_per_worker": 3,
    # "model": {"fcnet_hiddens": [8, 8]},
    # "train_batch_size": int(4000/8),
    # "rollout_fragment_length": int(200/8),
    # "sgd_minibatch_size": 128,
    "lr": tune.grid_search([0.001, 0.0003]),
    "env_config": env_config,
    # "use_lstm": tune.grid_search([True, False])

    # "exploration_config": {
    #     "type": "Curiosity",  # <- Use the Curiosity module for exploring.
    #     "eta": 1.0,  # Weight for intrinsic rewards before being added to extrinsic ones.
    #     "lr": 0.001,  # Learning rate of the curiosity (ICM) module.
    #     "feature_dim": 288,  # Dimensionality of the generated feature vectors.
    #     # Setup of the feature net (used to encode observations into feature (latent) vectors).
    #     "feature_net_config": {
    #         "fcnet_hiddens": [],
    #         "fcnet_activation": "relu",
    #     },
    #     "inverse_net_hiddens": [256],  # Hidden layers of the "inverse" model.
    #     "inverse_net_activation": "relu",  # Activation of the "inverse" model.
    #     "forward_net_hiddens": [256],  # Hidden layers of the "forward" model.
    #     "forward_net_activation": "relu",  # Activation of the "forward" model.
    #     "beta": 0.2,  # Weight for the "forward" loss (beta) over the "inverse" loss (1.0 - beta).
    #     # Specify, which exploration sub-type to use (usually, the algo's "default"
    #     # exploration, e.g. EpsilonGreedy for DQN, StochasticSampling for PG/SAC).
    #     "sub_exploration": {
    #         "type": "StochasticSampling",
    #     }
    # }

}