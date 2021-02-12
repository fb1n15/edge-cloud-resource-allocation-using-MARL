import ray
from datetime import datetime

from common.config import stop, config
from common.checkpoint_handler import save_checkpoints

from ray import tune
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()


def get_name():
    time = str(datetime.now()).replace(":", "-").replace(".", "-")
    return f"DroneRescue {time}"


def run_same_policy():
    """Use the same policy for both agents"""

    analysis = tune.run("PPO", name=get_name(),
                        config=config,
                        stop=stop,
                        verbose=2,
                        checkpoint_freq=100,
                        checkpoint_at_end=True
                        )

    checkpoints = analysis.get_trial_checkpoints_paths(
        trial=analysis.get_best_trial("episode_reward_mean", mode="max"),
        metric="episode_reward_mean")

    return checkpoints


def main(restore=None):
    # TODO add restore checkpoint option
    ray.init()
    checkpoints = run_same_policy()
    save_checkpoints(checkpoints)
    print(checkpoints)


if __name__ == "__main__":
    main()
