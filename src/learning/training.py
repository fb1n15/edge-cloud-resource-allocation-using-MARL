from typing import Dict

import ray
from datetime import datetime

from gym.spaces import Tuple
from ray.rllib import RolloutWorker, BaseEnv, Policy
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import GroupAgentsWrapper
from ray.rllib.evaluation import MultiAgentEpisode
from ray.rllib.models import ModelCatalog
from ray.tune import register_env
from ray.tune.schedulers import PopulationBasedTraining

from environments import environment_map

from ray import tune
from ray.rllib.utils.framework import try_import_torch

from learning.models.centralised_model import CentralisedModel
from learning.models.centralised_model2 import CentralisedModelFC
from learning.models.convolutional_model import ConvolutionalModel
from learning.models.fc_model import FCModel

torch, nn = try_import_torch()


def format_name(name):
    time = str(datetime.now()).replace(":", "-").replace(".", "-")
    return f"{name} {time}"


def get_trainer_config(config):
    trainer_config = config["trainer-config"]
    trainer_config["env_config"] = config["env-config"]

    # Choose environment, with groupings
    env = environment_map(config["env"])["env"]
    print()
    print(env)
    if "grouping" not in config:
        trainer_config["env"] = env

        trainer_config["multiagent"] = {
            "policies": {
                "default": (None, env.get_observation_space(config["env-config"]), env.get_action_space(config["env-config"]), {}),
            },
            "policy_mapping_fn": lambda agent_id: "default"
        }

    elif config["grouping"] == "all_same":
        print(env)
        obs_space = Tuple([env.get_observation_space(config["env-config"]) for i in range(config["env-config"]["num_agents"])])
        act_space = Tuple([env.get_action_space() for i in range(config["env-config"]["num_agents"])])
        grouping = {
            "group_1": ["drone_"+str(i) for i in range(config["env-config"]["num_agents"])],
        }

        # Register the environment with Ray, and use this in the config
        register_env(config["env"], lambda env_cfg: env(env_cfg).with_agent_groups(grouping, obs_space=obs_space, act_space=act_space))
        trainer_config["env"] = config["env"]
        # trainer_config["env"] = env

        # trainer_config["multiagent"] = {
        #     "policies": {
        #         "default": (None, env.get_observation_space(config["env-config"]), env.get_action_space(), {}),
        #     },
        #     "policy_mapping_fn": lambda agent_id: "default"
        # }

        trainer_config["multiagent"] = {
            "policies": {
                "default": (None, obs_space, act_space, {}),
            },
            "policy_mapping_fn": lambda agent_id: "default"
        }
    elif config["grouping"] == "centralised":
        print(env)
        obs_space = Tuple([env.get_observation_space(config["env-config"]) for i in range(config["env-config"]["num_agents"])])
        act_space = Tuple([env.get_action_space() for i in range(config["env-config"]["num_agents"])])
        grouping = {
            "group_1": ["drone_"+str(i) for i in range(config["env-config"]["num_agents"])],
        }

        # Register the environment with Ray, and use this in the config
        register_env(config["env"], lambda env_cfg: env(env_cfg).with_agent_groups(grouping, obs_space=obs_space, act_space=act_space))
        trainer_config["env"] = config["env"]
        # trainer_config["env"] = env

        # trainer_config["multiagent"] = {
        #     "policies": {
        #         "default": (None, env.get_observation_space(config["env-config"]), env.get_action_space(), {}),
        #     },
        #     "policy_mapping_fn": lambda agent_id: "default"
        # }

        trainer_config["multiagent"] = {
            "policies": {
                "default": (None, obs_space, act_space, {}),
            },
            "policy_mapping_fn": lambda agent_id: "default"
        }

    elif config["grouping"] == "radar-rescue":
        trainer_config["env"] = env

        trainer_config["multiagent"] = {
            "policies": {
                "radar": (None, env.get_observation_space(config["env-config"], "radar"), env.get_action_space("radar"), {}),
                "rescue": (None, env.get_observation_space(config["env-config"], "rescue"), env.get_action_space("rescue"), {}),
            },
            "policy_mapping_fn": lambda agent_id: agent_id.split("_")[0]
        }

    # ModelCatalog.register_custom_model("CustomVisionNetwork", CustomVisionNetwork)
    ModelCatalog.register_custom_model("ConvolutionalModel", ConvolutionalModel)
    ModelCatalog.register_custom_model("CentralisedModel", CentralisedModel)
    ModelCatalog.register_custom_model("CentralisedModelFC", CentralisedModelFC)
    ModelCatalog.register_custom_model("FCModel", FCModel)

    # Add callbacks for custom metrics
    trainer_config["callbacks"] = CustomCallbacks

    return trainer_config


def train(config):
    """
    :param config:
    :return: Analysis object
    """

    trainer_config = get_trainer_config(config)

    # Add scheduler, as specified by config
    scheduler = None
    if "scheduler" in config:
        if config["scheduler"] == "pbt":
            scheduler = PopulationBasedTraining(**config["scheduler-config"])
    print("start training")
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
        checkpoint_at_end=True,
        num_samples=config.get("samples", 1),
        fail_fast='raise'
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
