from typing import Dict

import ray
from datetime import datetime

from ray.rllib import RolloutWorker, BaseEnv, Policy, SampleBatch
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.evaluation import MultiAgentEpisode

from common.config import stop, config

from ray import tune
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()


def get_name():
    time = str(datetime.now()).replace(":", "-").replace(".", "-")
    return f"DroneRescue {time}"


def run_same_policy():
    """Use the same policy for all agents"""
    _config = config
    _config["callbacks"] = CustomCallbacks
    analysis = tune.run(
        "PPO",
        name=get_name(),
        # name="DroneRescue 2021-03-02 13-07-52-039234",
        # restore=r"C:\Users\Jack\PycharmProjects\marl-disaster-relief\src\results\DroneRescue 2021-03-02 "
        #         r"13-07-52-039234\PPO_GridWorldEnv_4b97a_00001_1_lr=0.001_2021-03-02_13-29-07\checkpoint_100"
        #         r"\checkpoint-100",
        local_dir="results/",
        config=config,
        stop=stop,
        verbose=3,
        checkpoint_freq=20,
        checkpoint_at_end=True,
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


class CustomCallbacks(DefaultCallbacks):

    def on_episode_end(self, *, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[str, Policy], episode: MultiAgentEpisode,
                       env_index: int, **kwargs):
        assert len(base_env.get_unwrapped()) == 1
        env = base_env.get_unwrapped()[0]
        episode.custom_metrics["Survivors Rescued"] = env.get_survivors_rescued()
        episode.custom_metrics["Agents Crashed"] = env.num_agents_crashed()
        episode.custom_metrics["Map Explored"] = env.get_map_explored()
