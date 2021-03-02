import ray
from datetime import datetime

from common.config import stop, config

from ray import tune
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()


def get_name():
    time = str(datetime.now()).replace(":", "-").replace(".", "-")
    return f"DroneRescue {time}"


def run_same_policy():
    """Use the same policy for all agents"""

    analysis = tune.run(
        # "PPO", name=get_name(),
        "PPO",
        name="DroneRescue 2021-03-02 13-07-52-039234",
        restore=r"C:\Users\Jack\PycharmProjects\marl-disaster-relief\src\results\DroneRescue 2021-03-02 "
                r"13-07-52-039234\PPO_GridWorldEnv_4b97a_00001_1_lr=0.001_2021-03-02_13-29-07\checkpoint_100"
                r"\checkpoint-100",
        local_dir="results/",
        config=config,
        stop=stop,
        verbose=3,
        checkpoint_freq=50,
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
    print(checkpoints)


if __name__ == "__main__":
    main()
