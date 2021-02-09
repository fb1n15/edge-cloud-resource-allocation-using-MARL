from ray import tune

from simulation.environment import GridWorldEnv

stop = {
    "training_iteration": 1000,
    # "episode_reward_mean": 14,
}
config = {
    "env": GridWorldEnv,
    "framework": "torch",
    # "num_gpus": 1,
    # "num_workers": 0,
    # "num_cpus_for_driver": 1,
    # "num_cpus_per_worker": 1,
    # "lr": 0.01,
    # "model": {"fcnet_hiddens": [8, 8]},
    "lr": tune.grid_search([0.01, 0.001, 0.0001, 0.00001]),
    "env_config": {
        "version": GridWorldEnv.VERSION,
        "width": 40,
        "height": 40,
        "num_survivors": 25,
        "num_agents": 4,
        "start_world": [[]],
        "sight": 8,
        "battery": 300,
        "rewards": {
            "rescue": 1,
            "hit tree": tune.grid_search([-5, -2, -0.5])
        },
        "battery costs": {
            "rotate left": 1,
            "rotate right": 1,
            "advance": 2
        },
        "autogen config": {
            "forest fire 1": {
                "trees": {
                    "scale": 20.0,
                    "octaves": 6,
                    "persistence": 0.5,
                    "lacunarity": 2.0,
                    "threshold": 0.07
                }
            }
        }
    }
}