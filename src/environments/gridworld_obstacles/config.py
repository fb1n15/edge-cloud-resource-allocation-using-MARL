from environments.gridworld_obstacles.simulation.environment import GridWorldEnv

gridworld_obstacles_env_config = {
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