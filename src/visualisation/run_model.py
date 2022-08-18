from pprint import pprint
from random import random

import pandas as pd
import pygame
import ray
import thorpy
import threading

from gym.spaces import Tuple
from ray.tune import register_env

from environments import environment_map
from learning.training import CustomCallbacks, get_trainer_config
from visualisation.gridworld_vis import render_HUD
import ray.rllib.agents.ppo as ppo
import random

WIDTH = 640
HEIGHT = 720


def training_config(config):
    trainer_config = config["trainer-config"]
    trainer_config["env_config"] = config["env-config"]

    # Choose environment, with groupings
    env = environment_map(config["env"])["env"]
    if "grouping" not in config:
        trainer_config["env"] = env

        trainer_config["multiagent"] = {
            "policies": {
                "default": (None, env.get_observation_space(config["env-config"]),
                            env.get_action_space(), {}),
                },
            "policy_mapping_fn": lambda agent_id: "default"
            }

    elif config["grouping"] == "all_same":
        obs_space = Tuple(
            [env.get_observation_space(config["env-config"]) for i in
             range(config["env-config"]["num_agents"])])
        act_space = Tuple(
            [env.get_action_space() for i in range(config["env-config"]["num_agents"])])
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

    elif config["grouping"] == "radar-rescue":
        trainer_config["env"] = env

        trainer_config["multiagent"] = {
            "policies": {
                "radar": (
                    None, env.get_observation_space(config["env-config"], "radar"),
                    env.get_action_space("radar"), {}),
                "rescue": (
                    None, env.get_observation_space(config["env-config"], "rescue"),
                    env.get_action_space("rescue"), {}),
                },
            "policy_mapping_fn": lambda agent_id: agent_id.split("_")[0]
            }

    # ModelCatalog.register_custom_model("CustomVisionNetwork", CustomVisionNetwork)

    # Add callbacks for custom metrics
    trainer_config["callbacks"] = CustomCallbacks

    return trainer_config


class SimulationRunner:
    def __init__(self, experiment, env, config, fake_pct=0):

        # Create logger which doesn't do anything
        # del experiment["best trial"]["config"]["callbacks"]  # Get rid of any callbacks
        # experiment["best trial"]["config"]["explore"] = False
        # # TODO remove sampled stuff from config
        # trainer_config = {"env_config": training_config(config)["env_config"],
        #                   "multiagent": training_config(config)["multiagent"],
        #                   "model": training_config(config)["model"],
        #                   "framework": "torch",
        #                   "explore": False}
        self.env = env(config["env-config"])
        self.fake_pct = fake_pct  # percentage of task types are fake
        # https://docs.ray.io/en/latest/rllib-training.html#accessing-policy-state
        # accessing model obs
        trainer_config = get_trainer_config(config)
        self.agent = ppo.PPOTrainer(config=trainer_config,
                                    env=env)

        self.agent.restore(experiment)  # Restore the last checkpoint
        # self.agent.restore(experiment["best trial"]["path"])  # Restore the last checkpoint

        # self.gridworld = self.env.controller

        self.episode_reward = 0
        self.done = False
        self.obs = self.env.reset()
        self.running = True

        self.speed_callback = None
        self.timestep = 0

        policy_map = self.agent.workers.local_worker().policy_map
        self.model_state = {p: m.get_initial_state() for p, m in policy_map.items()}
        print(self.model_state)

    def set_speed_callback(self, callback):
        self.speed_callback = callback

    def step_simulation(self):
        self.done = {"__all__": False}  # set the done flag to false
        counter = 0  # Reset episodes counter
        while counter < 100:
            episode_reward = 0  # Reset episode reward
            self.timestep += 1
            action = {}
            for agent_id, agent_obs in self.obs.items():
                # for 20% probability
                fake_obs = agent_obs.copy()
                if random.random() < self.fake_pct:
                    # change the valuation coefficient to 1 million
                    fake_obs[0] = 1e6

                policy_id = agent_id.split("_")[0]
                if policy_id not in self.model_state:
                    policy_id = "default"
                action[agent_id], self.model_state[
                    policy_id], _ = self.agent.compute_action(
                    observation=fake_obs,
                    policy_id=policy_id,
                    state=self.model_state[policy_id],
                    full_fetch=True
                    )
            self.obs, self.reward, self.done, self.info = self.env.step(action)
            episode_reward += sum(self.reward.values())

            # print(f"reward: {self.reward}")

            if self.done["__all__"]:
                counter += 1  # Increment episodes counter
                print(f"episode ID = {counter}")  # Print episode ID
                print(f"episode_reward: {episode_reward}")  # Print episode reward
                episode_social_welfare = self.env.get_total_sw()
                print(
                    f"episode_social_welfare: {episode_social_welfare}")  # Print episode social welfare
                social_welfare_online_myopic = self.env.get_total_sw_online_myopic()
                print(f"social_welfare_online_myopic: {social_welfare_online_myopic}")
                social_welfare_bidding_zero = self.env.get_total_sw_bidding_zero()
                print(f"social_welfare_bidding_zero: {social_welfare_bidding_zero}")
                social_welfare_offline_optimal = self.env.get_total_sw_offline_optimal()
                print(f"social_welfare_offline_optimal: {social_welfare_offline_optimal}")
                social_welfare_random_allocation = self.env.get_total_sw_random_allocation()
                print(
                    f"social_welfare_random_allocation: {social_welfare_random_allocation}")
                # save the results to a dataframe
                if counter == 1:
                    self.df = pd.DataFrame({"DAPPO": [episode_social_welfare],
                                            "Offline Optimal": [
                                                social_welfare_offline_optimal],
                                            "Online Greedy": [
                                                social_welfare_online_myopic],
                                            "Bidding Zero": [social_welfare_bidding_zero],
                                            "Random Allocation": [
                                                social_welfare_random_allocation]})
                else:
                    self.df = self.df.append({"DAPPO": episode_social_welfare,
                                              "Offline Optimal": social_welfare_offline_optimal,
                                              "Online Greedy": social_welfare_online_myopic,
                                              "Bidding Zero": social_welfare_bidding_zero,
                                              "Random Allocation": social_welfare_random_allocation},
                                             ignore_index=True)
                self.restart_simulation()

    def restart_simulation(self):
        self.done = False
        self.obs = self.env.reset()
        self.timestep = 0

    def get_time(self):
        return self.timestep

    def get_rescued(self):
        return self.env.get_survivors_rescued()

    def num_agents_crashed(self):
        return self.env.num_agents_crashed()


def start_displaying(runner, env):
    MENU_HEIGHT = 80

    pygame.init()
    pygame.font.init()
    pygame.key.set_repeat(300, 30)
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    screen.fill((255, 255, 255))
    rect = pygame.Rect((0, 0, 50, 50))
    rect.center = screen.get_rect().center
    clock = pygame.time.Clock()

    pygame.draw.rect(screen, (255, 0, 0), rect)
    pygame.display.flip()

    # Declaration of some ThorPy elements ...
    slider = thorpy.SliderX(100, (0.1, 15), "Speed")
    slider.set_value(5)
    runner.set_speed_callback(lambda: slider.get_value())
    button = thorpy.make_button("Restart", func=runner.restart_simulation)
    box = thorpy.Box(elements=[slider, button])
    menu = thorpy.Menu(box)

    # Set the screen as surface for all elements
    for element in menu.get_population():
        element.surface = screen

    box.set_topleft((0, 0))
    box.blit()
    box.update()

    t = threading.Thread(target=runner.step_simulation)
    t.start()

    running = True
    while running:
        clock.tick(60)

        screen.blit(env["render"](runner.gridworld, WIDTH, HEIGHT - MENU_HEIGHT),
                    (0, MENU_HEIGHT))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            menu.react(event)
        hud = render_HUD(runner.get_rescued(), runner.get_time(),
                         runner.num_agents_crashed())
        screen.blit(hud, (WIDTH - hud.get_width(), 0))
        pygame.display.update()

    pygame.quit()
    runner.running = False


def main(experiment, config):
    ray.init()
    env = environment_map(config["env"])
    # for probability in [0.4, 0.6, 0.8]:
    for probability in [0.0, 1.0]:
        runner = SimulationRunner(experiment, env["env"], config, fake_pct=probability)
        runner.step_simulation()
        print(runner.df)
        runner.df.to_csv(f"{experiment}_results_fake_pct={runner.fake_pct}.csv")  # save the results to a csv file
