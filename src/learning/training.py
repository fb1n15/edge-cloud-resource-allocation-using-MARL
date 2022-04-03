from pprint import pprint
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
from learning.models.centralised_critic_fc_model import CentralisedCriticFCModel
from learning.models.convolutional_model import ConvolutionalModel
from learning.models.fc_model import FCModel

torch, nn = try_import_torch()


def format_name(name):
    time = str(datetime.now()).replace(":", "-").replace(".", "-")
    return f"{name} {time}"


def get_trainer_config(config):
    # return the processed trainer configuration
    trainer_config = config["trainer-config"]
    trainer_config["env_config"] = config["env-config"]

    # Choose environment, with groupings
    env = environment_map(config["env"])["env"]
    print()
    print(env)
    # # configure logging
    # fmtStr = "%(asctime)s: %(levelname)s: %(funcName)s() -> %(message)s"
    #
    # logging.basicConfig(level=logging.DEBUG, filename='resource_allocation.log',
    #                     filemode='w', format=fmtStr)
    # independent agents
    if "grouping" not in config:
        trainer_config["env"] = env
        # logging.debug(f"env = {env}")
        # {None (i.e., uses default policy), observation_space, action_space, config: dict}
        trainer_config["multiagent"] = {
            "policies": {
                "default": (
                    None, env.get_observation_space(config["env-config"]),
                    env.get_action_space(config["env-config"])
                    , {}),
                },
            # all agents are bound to the 'default' policy
            "policy_mapping_fn": lambda agent_id: "default"
            }

    # agents using a centralised model
    elif config["grouping"] == "all_same":
        print("environment = ", env)
        # combines the observations and actions of all the agents in to one
        obs_space = Tuple(
            [env.get_observation_space(config["env-config"]) for i in
             range(config["env-config"]["num_agents"])])
        # act_space = Tuple([env.get_action_space(config["env-config"]) for i in
        act_space = Tuple([env.get_action_space() for i in
                           range(config["env-config"]["num_agents"])])

        print(f"observation space = {obs_space}")
        print(f"action space = {act_space}")
        grouping = {
            "group_1": ["drone_" + str(i) for i in
                        range(config["env-config"]["num_agents"])],
            }

        # RLlib treats agent groups like a single agent with a Tuple action and observation space.
        # Register the environment with Ray, and use this in the config
        register_env(config["env"],
                     lambda env_cfg: env(env_cfg).with_agent_groups(grouping,
                                                                    obs_space=obs_space,
                                                                    act_space=act_space))
        trainer_config["env"] = config["env"]
        print(f"grouping = {grouping}")

        trainer_config["multiagent"] = {
            "policies": {
                "default": (None, obs_space, act_space, {}),
                },
            "policy_mapping_fn": lambda agent_id: "default"
            }
    # the same as before, using a centralised critic
    elif config["grouping"] == "centralised":
        print(env)
        obs_space = Tuple(
            [env.get_observation_space(config["env-config"]) for i in
             range(config["env-config"]["num_agents"])])
        act_space = Tuple([env.get_action_space() for i in
                           range(config["env-config"]["num_agents"])])
        grouping = {
            "group_1": ["drone_" + str(i) for i in
                        range(config["env-config"]["num_agents"])],
            }

        # Register the environment with Ray, and use this in the config
        register_env(config["env"],
                     lambda env_cfg: env(env_cfg).with_agent_groups(grouping,
                                                                    obs_space=obs_space,
                                                                    act_space=act_space))
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

    # not useful for me
    elif config["grouping"] == "radar-rescue":
        trainer_config["env"] = env

        trainer_config["multiagent"] = {
            "policies": {
                "radar": (
                    None,
                    env.get_observation_space(config["env-config"], "radar"),
                    env.get_action_space("radar"), {}),
                "rescue": (
                    None,
                    env.get_observation_space(config["env-config"], "rescue"),
                    env.get_action_space("rescue"), {}),
                },
            "policy_mapping_fn": lambda agent_id: agent_id.split("_")[0]
            }

    # register the customed model so that they can be used like built-in ones
    ModelCatalog.register_custom_model("ConvolutionalModel", ConvolutionalModel)
    ModelCatalog.register_custom_model("CentralisedModel", CentralisedModel)
    ModelCatalog.register_custom_model("CentralisedModelFC", CentralisedModelFC)
    ModelCatalog.register_custom_model("CentralisedCriticFCModel",
                                       CentralisedCriticFCModel)
    ModelCatalog.register_custom_model("FCModel", FCModel)

    # Add callbacks for custom metrics
    trainer_config["callbacks"] = CustomCallbacks

    return trainer_config


def train(config):
    """
    param config:
    return: Analysis object
    """

    trainer_config = get_trainer_config(config)

    # Add scheduler, as specified by config
    scheduler = None
    if "scheduler" in config:
        if config["scheduler"] == "pbt":
            scheduler = PopulationBasedTraining(**config["scheduler-config"])
    print("start training")
    # pprint(f"trainer_config = {trainer_config}")
    # training the agent
    analysis = tune.run(
        run_or_experiment=config["trainer"],  # this is the model to train
        name=config["name"],  # name of the checkpoint
        scheduler=scheduler,  # Scheduler for executing the checkpoint (default: FIFO)
        config=trainer_config,  # algorithm-specific configuration (e.g., num_workers)
        stop=config["stop"],  # stopping criteria
        # # If your trainable is slow to initialize, consider setting reuse_actors=True to reduce actor creation overheads.
        # Trainable.setup took 145.656 seconds. If your trainable is slow to initialize, consider setting reuse_actors=True to reduce actor creation overheads.
        # reuse_actors=False,
        reuse_actors=True,
        local_dir="./results",
        # local directory to save training results to
        # Verbosity mode. 0 = silent, 1 = only status updates, 2 = status and brief trial results, 3 = status and detailed trial results. Defaults to 3.
        verbose=3,
        # Whether to checkpoint at the end of the checkpoint regardless of the checkpoint_freq
        # (When training deep learning models, the checkpoint is the weights of the model. These weights can be used to make predictions as is, or used as the basis for ongoing training.)
        checkpoint_at_end=True,  # whether to checkpoint at the end of the checkpoint
        # number of times to sample from the hyperparameter space.
        num_samples=config.get("samples", 1),  # to take multiple random samples
        fail_fast=True,
        # whether to fail upon the first error, should be used with caution
        ## how to checkpoint?
        # https://docs.ray.io/en/latest/tune/user-guide.html#checkpointing
        # name="my_experiment",
        restore=config.get("restore", None),  # checkpoint to restore from
        # resume=True  # You can then call tune.run() with resume=True to continue this run in the future:
        )

    # Object for checkpoint analysis.
    return analysis


def main(config):
    # to stop the warnings, they are too many (To disable ray workers from logging the output.)
    # https://github.com/ray-project/ray/issues/5048
    ray.init(log_to_driver=True, include_dashboard=False)
    analysis = train(config)
    print(analysis)


class CustomCallbacks(DefaultCallbacks):
    # callbacks can be used for custom metrics
    # Runs when an episode is done.
    def on_episode_end(self, *, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[str, Policy], episode: MultiAgentEpisode,
                       env_index: int, benchmarks=False, **kwargs):
        assert len(base_env.get_unwrapped()) == 1
        env = base_env.get_unwrapped()[0]
        if isinstance(env, GroupAgentsWrapper):
            # If the environment is using a group wrapper (for Q-Mix)
            # Access the inner environment
            env = env.env
        # episode.custom_metrics["Survivors Rescued"] = env.get_survivors_rescued()
        # episode.custom_metrics["Agents Crashed"] = env.num_agents_crashed()
        # episode.custom_metrics["Map Explored"] = env.get_map_explored()
        episode.custom_metrics["Social Welfare (PPO)"] = env.get_total_sw()
        if benchmarks:
            episode.custom_metrics[
                "Social Welfare (Online Myopic)"] = env.get_total_sw_online_myopic()
            episode.custom_metrics[
                "Social Welfare (Random Allocation)"] = env.get_total_sw_random_allocation()
            episode.custom_metrics[
                "Social Welfare (All Bidding Zero)"] = env.get_total_sw_bidding_zero()
            episode.custom_metrics[
                "Social Welfare (Offline Optimal)"] = env.get_total_sw_offline_optimal()
            episode.custom_metrics[
                "Allocated Tasks Number (PPO)"] = env.get_total_allocated_task_num()
            episode.custom_metrics[
                "Bad Allocations Number (PPO)"] = env.get_num_bad_allocations()
