from pprint import pprint
from typing import Dict

import ray
from datetime import datetime

from gym.spaces import Tuple
from ray.rllib import RolloutWorker, BaseEnv, Policy
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import GroupAgentsWrapper
from ray.rllib.evaluation import MultiAgentEpisode
from ray.tune import register_env
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

    trainer_config = config["trainer-config"]
    trainer_config["env_config"] = config["env-config"]

    # Choose environment, with groupings
    env = environment_map(config["env"])["env"]
    if "grouping" not in config:
        trainer_config["env"] = env

        trainer_config["multiagent"] = {
            "policies": {
                "default": (None, env.get_observation_space(config["env-config"]), env.get_action_space(), {}),
            },
            "policy_mapping_fn": lambda agent_id: "default"
        }

    elif config["grouping"] == "all_same":
        obs_space = Tuple([env.get_observation_space(config["env-config"]) for i in range(config["env-config"]["num_agents"])])
        act_space = Tuple([env.get_action_space() for i in range(config["env-config"]["num_agents"])])
        grouping = {
            "group_1": ["drone_"+str(i) for i in range(config["env-config"]["num_agents"])],
        }

        # Register the environment with Ray, and use this in the config
        register_env(config["env"], lambda env_cfg: env.with_agent_groups(env(env_cfg), grouping))
        trainer_config["env"] = config["env"]

        trainer_config["multiagent"] = {
            "policies": {
                "default": (None, obs_space, act_space, {}),
            },
            "policy_mapping_fn": lambda agent_id: "default"
        }

    # Add callbacks for custom metrics
    trainer_config["callbacks"] = CustomCallbacks

    # Add scheduler, as specified by config
    scheduler = None
    if "scheduler" in config:
        if config["scheduler"] == "pbt":
            scheduler = PopulationBasedTraining(**config["scheduler-config"])

    analysis = tune.run(
        config["trainer"],
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
        num_samples=config.get("samples", 1),
    )

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
        if isinstance(env, GroupAgentsWrapper):
            # If the environment is using a group wrapper (for Q-Mix)
            # Access the inner environment
            env = env.env
        episode.custom_metrics["Survivors Rescued"] = env.get_survivors_rescued()
        episode.custom_metrics["Agents Crashed"] = env.num_agents_crashed()
        episode.custom_metrics["Map Explored"] = env.get_map_explored()
