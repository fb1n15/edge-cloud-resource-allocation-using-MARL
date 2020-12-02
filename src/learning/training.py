import argparse
import os

import ray

from simulation.environment import GridWorldEnv

from ray import tune
from ray.rllib.utils.framework import try_import_tf, try_import_torch

tf1, tf, tfv = try_import_tf()

stop = {
    "training_iteration": 100,
    "episode_reward_mean": 9.5,
}
config = {
    "env": GridWorldEnv,
    "framework": "tf",
    "num_gpus": 0,
    "num_workers": 5,
    # "num_cpus_for_driver": 1,
    # "num_cpus_per_worker": 1,
    # "lr": 0.01,
    # "model": {"fcnet_hiddens": [8, 8]},
    "env_config": {
        "width": 20,
        "height": 20,
        "num_survivors": 10,
        "num_agents": 2,
        "start_world": [[]],
        "sight": 4,
        "battery": 200,
    }
}

ray.init()


def run_same_policy():
    """Use the same policy for both agents (trivial case)."""

    analysis = tune.run("PPO", name="drone_rescue", config=config, stop=stop, verbose=1, checkpoint_freq=1, checkpoint_at_end=True)

    checkpoints = analysis.get_trial_checkpoints_paths(
        trial=analysis.get_best_trial("episode_reward_mean", mode="max"),
        metric="episode_reward_mean")

    return checkpoints


def save_checkpoints(checkpoints):
    path = os.path.expanduser(os.path.join("~", "Gridworld"))
    if not os.path.exists(path):
        os.makedirs(path)
    with open(os.path.join(path, "checkpoints.txt"), "w+") as f:
        f.write(str(checkpoints))


def load_checkpoints():
    with open(os.path.join(os.path.expanduser("~"), "Gridworld", "checkpoints.txt"), "r") as f:
        return eval(f.readline())


def main():
    checkpoints = run_same_policy()
    save_checkpoints(checkpoints)
    print(checkpoints)


if __name__ == "__main__":
    main()
