import ray
from datetime import datetime

from simulation.environment import GridWorldEnv
from common.checkpoint_handler import save_checkpoints

from ray import tune
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()

stop = {
    "training_iteration": 5,
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
    "lr": tune.grid_search([0.001, 0.0001, 0.00001]),
    "env_config": {
        "version": GridWorldEnv.VERSION,
        "width": 50,
        "height": 50,
        "num_survivors": 25,
        "num_agents": 3,
        "start_world": [[]],
        "sight": 6,
        "battery": 200,
        "rewards": {
            "rescue": 1,
            "hit tree": -5
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

ray.init()


def get_name():
    time = str(datetime.now()).replace(":", "-").replace(".", "-")
    return f"DroneRescue {time}"


def run_same_policy():
    """Use the same policy for both agents"""

    analysis = tune.run("PPO", name=get_name(),
                        config=config,
                        stop=stop,
                        verbose=2,
                        checkpoint_freq=1,
                        checkpoint_at_end=True
                        )

    checkpoints = analysis.get_trial_checkpoints_paths(
        trial=analysis.get_best_trial("episode_reward_mean", mode="max"),
        metric="episode_reward_mean")

    return checkpoints


def main(restore=None):
    # TODO add restore checkpoint option
    checkpoints = run_same_policy()
    save_checkpoints(checkpoints)
    print(checkpoints)


if __name__ == "__main__":
    main()
