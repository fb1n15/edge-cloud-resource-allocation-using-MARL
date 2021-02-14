from ray import tune

from simulation.environment import GridWorldEnv

stop = {
    "training_iteration": 5,
    # "episode_reward_mean": 14,
}

env_config = {
    "version": GridWorldEnv.VERSION,
    "width": 60,
    "height": 60,
    "num_survivors": 25,
    "num_agents": 4,
    "start_world": [[]],
    "sight": 8,
    "battery": 300,
    "rewards": {
        "rescue": 1,
        "hit tree": -3
    },
    "battery costs": {
        "rotate left": 1,
        "rotate right": 1,
        "advance": 2
    },
    "autogen config": {
        "forest fire": {
            "chance": 1,
            "trees": {
                "scale": 20.0,
                "octaves": 6,
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
            "fire spread": {
                "speed": 10,
                "starting points": 5
            }
        }
    }
}

config = {
    "env": GridWorldEnv,
    "framework": "torch",
    "num_gpus": 1,
    "num_cpus_for_driver": 1,
    # "num_cpus_per_worker": 1,
    # "lr": 0.01,
    # "model": {"fcnet_hiddens": [8, 8]},
    "num_envs_per_worker": 2,
    # "train_batch_size": int(4000/8),
    # "rollout_fragment_length": int(200/8),
    # "sgd_minibatch_size": 128,
    "num_workers": 3,
    "lr": tune.grid_search([0.01, 0.001]),
    "env_config": env_config

}