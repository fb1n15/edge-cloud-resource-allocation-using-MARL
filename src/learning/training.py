from pprint import pprint
from typing import Dict

import ray
from datetime import datetime

from ray.rllib import RolloutWorker, BaseEnv, Policy
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.evaluation import MultiAgentEpisode
from ray.tune.schedulers import PopulationBasedTraining

from environments import environment_map

from ray import tune
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()


def format_name(name):
    time = str(datetime.now()).replace(":", "-").replace(".", "-")
    return f"{name} {time}"


def train(config):
    """
    :param config:
    :return: Analysis object
    """

    # _config = config["common"]
    # _config = merge_dicts(_config, config[trainer]["config"])
    # _config = merge_dicts(_config, performance_configs[platform])
    # _config["env_config"] = environments[env]["env_config"]
    # _config["env"] = environments[env]["env"]
    pprint(config)

    trainer_config = config["trainer-config"]
    trainer_config["env_config"] = config["env-config"]
    trainer_config["env"] = environment_map(config["env"])["env"]

    # Add callbacks for custom metrics
    trainer_config["callbacks"] = CustomCallbacks

    scheduler = None
    if "scheduler" in config:
        if config["scheduler"] == "pbt":
            scheduler = PopulationBasedTraining(**config["scheduler-config"])

    analysis = tune.run(
        "PPO",
        name=config["name"],
        # name="DroneRescue 2021-03-02 13-07-52-039234",
        # restore=r"C:\Users\Jack\PycharmProjects\marl-disaster-relief\src\results\DroneRescue 2021-03-02 "
        #         r"13-07-52-039234\PPO_GridWorldEnv_4b97a_00001_1_lr=0.001_2021-03-02_13-29-07\checkpoint_100"
        #         r"\checkpoint-100",
        scheduler=scheduler,
        config=trainer_config,
        stop=config["stop"],
        local_dir="results/",
        verbose=3,
        checkpoint_freq=20,
        checkpoint_at_end=True,
        num_samples=20,
    )

    # checkpoints = analysis.get_trial_checkpoints_paths(
    #     trial=analysis.get_best_trial("episode_reward_mean", mode="max"),
    #     metric="episode_reward_mean")

    return analysis


def main(config):
    ray.init()
    analysis = train(config)
    print(analysis)


class CustomCallbacks(DefaultCallbacks):

    def on_episode_end(self, *, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[str, Policy], episode: MultiAgentEpisode,
                       env_index: int, **kwargs):
        assert len(base_env.get_unwrapped()) == 1
        env = base_env.get_unwrapped()[0]
        episode.custom_metrics["Survivors Rescued"] = env.get_survivors_rescued()
        episode.custom_metrics["Agents Crashed"] = env.num_agents_crashed()
        episode.custom_metrics["Map Explored"] = env.get_map_explored()
