import logging
from pathlib import Path
from time import strftime
from collections import deque
from itertools import chain
from typing import Tuple

from gym.spaces import Discrete, Box
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict

from benchmarks.all_bidding_zero import all_bidding_zero
from benchmarks.online_myopic_m import online_myopic
from benchmarks.random_allocation import random_allocation
from environments.gridworld_obstacles.simulation.entities import Agent
from environments.gridworld_obstacles.simulation.gridworld_controller import \
    SimulationController
from environments.gridworld_obstacles.visualisation.render import \
    render_gridworld
import copy

import gym
import numpy as np
import pandas as pd
from gym import spaces
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.sample_batch import SampleBatch
from gym.spaces import Box, Dict, Discrete

from common.generate_simulation_data import generate_synthetic_data_edge_cloud

pd.set_option("display.max_rows", None, "display.max_columns", None)


def argmax_earliest(utility_arr, bid_start_time_arr):
    """
    Takes in a list of values and returns the index of the item
    with the highest value (with the smallest start time).  Breaks ties randomly.
    (Because np.argmax always return the first index of highest value)

    Args:
        utility_arr: an array of utilities
        bid_start_time_arr: an array of start times

    Returns:
        int - the index of the highest value in values
    """
    top_value = float("-inf")
    earliest_time = float('inf')
    ties = []
    ties2 = []

    # find the nodes with highest utility to the task
    for i in range(len(utility_arr)):
        # if a value in values is greater than
        # ~the highest value update top and reset ties to zero
        if utility_arr[i] > top_value:
            top_value = utility_arr[i]
            ties = []
        # if a value is equal to top value add the index to ties
        if utility_arr[i] == top_value:
            ties.append(i)

    # find in these nodes the nodes with the earliest start time
    for i in ties:
        if bid_start_time_arr[i] < earliest_time:
            earliest_time = bid_start_time_arr[i]
            ties2 = []
        if bid_start_time_arr[i] == earliest_time:
            ties2.append(i)

    # return a random selection from ties2.
    return np.random.choice(ties2)


def fill_in_actions(info):
    """Callback that saves opponent actions into the agent obs.
    If you don't care about opponent actions you can leave this out."""

    to_update = info["post_batch"][SampleBatch.CUR_OBS]
    my_id = info["agent_id"]
    other_id = 1 if my_id == 0 else 0
    action_encoder = ModelCatalog.get_preprocessor_for_space(Discrete(2))

    # set the opponent actions into the observation
    _, opponent_batch = info["all_pre_batches"][other_id]
    opponent_actions = np.array([
        action_encoder.transform(a)
        for a in opponent_batch[SampleBatch.ACTIONS]
        ])
    to_update[:, -2:] = opponent_actions


class EdgeCloudEnv(MultiAgentEnv):
    """Our edge cloud resource allocation environment"""
    VERSION = 1  # Increment each time there are non-backwards compatible changes m

    def __init__(self, config,
                 seed=0, n_timesteps=10, n_tasks=50,
                 max_steps=11,
                 p_high_value_tasks=0.3, high_value_slackness=0,
                 low_value_slackness=0, resource_ratio=1, valuation_ratio=20,
                 resource_coefficient=0.2,
                 forgiveness_factor=30, logging_level=logging.DEBUG,
                 allow_negative_reward=False,
                 alpha=1.0, lam=1e2, occup_len=4, history_len=3):
        """
        Initialization function for the environment.
        Args:

            seed: seed_value for generating simulation data
            duration The number of timestamps.
            allow_negative_reward: Flag for allowing negative rewards for the bad
            allocation of tasks.
            forgiveness_factor: Tolerance to sequential bad allocation of tasks.
            alpha: Percentage of the total rewards influenced by the prioritisation of
            high valuation tasks.
            lam: Speed of increase in the rewards generated from the prioritisation of
            high valuation tasks.
            not_verbose: A boolean as a flag to logging.debug information about the node
            allocation.
            record_history: whether to put others' action history to the observation of an agent
            occup_len: how many time steps of future occupancy is in observations
        """

        # Set the class variables
        super().__init__()
        # # Remove all handlers associated with the root logger object.
        # for handler in logging.root.handlers[:]:
        #     logging.root.removeHandler(handler)
        #
        # myfile = Path('./logs/edge_cloud_resource_allocation_{}.log'.format(
        #                         strftime('%Y/%m/%d_%T')))
        # myfile.touch(exist_ok=True)
        # config the log of the program
        fmtStr = "%(asctime)s: %(levelname)s: %(funcName)s() -> %(message)s"
        logging.basicConfig(level=logging_level,
                            filename='./logs/edge_cloud_resource_allocation{}.log'.format(
                                strftime('_%Y_%m_%d_%T')), filemode='w', format=fmtStr)

        self.occup_len = config['usage_time_ub']
        self.current_task = None
        self.current_time_slot = None
        self.next_time_slot = None
        self.n_tasks_expensive = 0
        self.winner_id = None
        self.winner_usage_time = None
        self.winner_start_time = None
        self.winner_finish_time = None
        self.resource_capacity_dict = None
        self.future_occup = None
        self.current_task_value = None
        self.df_tasks_relative = None
        self.record_history = config['record_history']
        self.cooperative = config['cooperative']
        self.rewards = {}
        self.sw_increase = None
        self.obs = {}
        self.state = {}
        self.avg_resource_capacity = config["avg_resource_capacity"]
        self.avg_unit_cost = config["avg_unit_cost"]
        self.n_tasks_to_allocate = config[
                                       'n_tasks_to_allocate'] + 1  # the number of tasks to allocate (+1 so that the program can work)
        self.history_len = config['history_len']

        self.n_tasks_in_total = config[
            'n_tasks_in_total']  # the number of tasks for generating the simulation data
        self.duration = config['duration']  # the duration of the allocation
        resource_coefficient = (
                resource_coefficient * self.n_tasks_in_total / self.duration)
        self.seed_value = config['seed']
        self.n_nodes = config['n_nodes']
        self.n_actions = config['n_actions']
        self.p_high_value_tasks = p_high_value_tasks
        self.high_value_slackness = high_value_slackness
        self.low_value_slackness = low_value_slackness
        self.resource_ratio = resource_ratio
        self.valuation_ratio = valuation_ratio
        self.resource_coefficient = resource_coefficient
        self.usage_time_ub = config['usage_time_ub']
        self.auction_type = config['auction_type']
        self.verbose = config["verbose"]
        # record the allocation scheme (task_id: [node_id, start_time, end_time])
        self.allocation_scheme = pd.DataFrame(
            columns=['node_id', 'start_time', 'end_time'])

        (df_tasks, df_nodes, n_time, n_tasks,
         n_nodes) = self.data_for_next_episode()

        self.allocated_tasks = []  # [(task_info, action)]

        self.lam = lam
        self.alpha = alpha
        self.allow_negative_reward = allow_negative_reward
        self.n_resource_type = 3

        # initialise the ndarray of idle resources
        self.full_resource_capacities = np.empty(
            [self.n_nodes, self.n_resource_type, self.duration])

        for node in df_nodes.iterrows():
            self.full_resource_capacities[node[0]] = [
                [df_nodes.loc[node[0], 'CPU'] for _ in range(self.duration)],
                [df_nodes.loc[node[0], 'RAM'] for _ in range(self.duration)],
                [df_nodes.loc[node[0], 'storage'] for _ in range(self.duration)]]

        self.df_tasks = df_tasks
        self.df_nodes = df_nodes

        self.current_task_id = 0
        self.failed = 0

        self.forgiveness_factor = forgiveness_factor

        self.total_social_welfare = 0
        self.total_allocated_tasks_num = 0

        self.processed_tasks = 0
        self.action_history = []

        self.allocation_map = dict((node, []) for node in range(
            self.n_nodes))  # A dict with the shape {node: [(task, start_index,
        # stop_index),]}

        self.idle_resource_capacities = copy.deepcopy(
            self.full_resource_capacities)

        # 10 discrete actions for each agent
        self.action_space = [spaces.Discrete(self.n_actions) for _ in
                             range(self.n_nodes)]

        # observation is the occupancy of future 10 time steps (because the relative
        # deadline of all tasks <= 10)
        # this is the global observation, will modify to local observations in the future
        self.observation_space = [spaces.Box(low=0, high=1,
                                             shape=(37,), dtype=np.float16) for _ in
                                  range(self.n_nodes)]

        self._episode_ended = False
        # an upper bound of the social welfare
        self.total_possible_reward = sum(
            df_tasks.valuation_coefficient * df_tasks.usage_time)

    def seed(self, seed_value):
        self.seed_value = seed_value

    def data_for_next_episode(self):
        logging.debug(f"average resource capacity = {self.avg_resource_capacity}")
        logging.debug(f"average unit cost = {self.avg_unit_cost}")
        logging.debug(f"number of tasks = {self.n_tasks_in_total}")
        logging.debug(f"number of timesteps = {self.duration}")
        logging.debug(f"seed = {self.seed_value}")
        logging.debug(f"number of nodes = {self.n_nodes}")
        logging.debug(f"proportion of HVTs = {self.p_high_value_tasks}")
        logging.debug(f"slackness of HVTs = {self.high_value_slackness}")
        logging.debug(f"slackness of LVTs = {self.low_value_slackness}")
        logging.debug(f"resource demand ratio = {self.resource_ratio}")
        logging.debug(f"value coefficient ratio = {self.valuation_ratio}")
        logging.debug(f"resource coefficient = {self.resource_coefficient}")
        df_tasks, df_nodes, n_time, n_tasks, n_nodes = \
            generate_synthetic_data_edge_cloud(self.avg_resource_capacity,
                                               self.avg_unit_cost,
                                               n_tasks=self.n_tasks_in_total,
                                               n_time=self.duration,
                                               seed=self.seed_value, n_nodes=self.n_nodes,
                                               p_high_value_tasks=self.p_high_value_tasks,
                                               high_value_slackness_lower_limit=self.high_value_slackness,
                                               high_value_slackness_upper_limit=self.high_value_slackness,
                                               low_value_slackness_lower_limit=self.low_value_slackness,
                                               low_value_slackness_upper_limit=self.low_value_slackness,
                                               resource_demand_high=self.resource_ratio,
                                               vc_ratio=self.valuation_ratio,
                                               k_resource=self.resource_coefficient,
                                               usage_time_ub=self.usage_time_ub)
        return df_tasks, df_nodes, n_time, n_tasks, n_nodes

    def reset(self):
        """Reset the tasks and nodes

        Returns the initial global observation
        """
        # occupancy is zero initially
        self.future_occup = np.zeros((self.n_nodes, self.occup_len, 3))
        self._episode_ended = False
        self.current_task_id = 0
        self.processed_tasks = 0
        self.failed = 0
        self.total_social_welfare = 0
        self.total_allocated_tasks_num = 0
        self.n_tasks_expensive = 0

        # generate new tasks for next episode
        (df_tasks, df_nodes, n_time, n_tasks,
         n_nodes) = self.data_for_next_episode()
        self.df_tasks = df_tasks
        self.df_nodes = df_nodes
        self.current_task = df_tasks.iloc[0]
        # make time constraints relative
        self.df_tasks_relative = self.df_tasks.iloc[0:self.n_tasks_to_allocate].copy()
        self.df_tasks_relative["relative_start_time"] = (
                self.df_tasks_relative['start_time'] -
                self.df_tasks_relative['arrive_time'].astype(int) - 1)
        self.df_tasks_relative["relative_deadline"] = (
                self.df_tasks_relative["deadline"] -
                self.df_tasks_relative["start_time"] + 1)
        self.df_tasks_relative.drop('start_time', inplace=True, axis=1)
        self.df_tasks_relative.drop('deadline', inplace=True, axis=1)
        self.df_tasks_relative.drop('arrive_time', inplace=True, axis=1)
        # # "0 max" normalisation
        # self.df_tasks_normalised = (self.df_tasks_relative - 0) / (
        #         self.df_tasks_relative.max() - 0)

        self.seed_value += 1  # may need to make this random in training

        # current timeslot is where the start time of current task in
        self.current_time_slot = int(self.df_tasks.loc[0, "arrive_time"])
        # generate a dict of resource capacity of future 10 time slots
        self.resource_capacity_dict = {}
        for node_id, info in df_nodes.iterrows():
            a = [info.get('CPU'), info.get('RAM'), info.get('storage')]
            self.resource_capacity_dict[node_id] = np.array(a * self.occup_len)

        # the obs is a vector of current task information and the future occupancy
        # const = np.array([1])
        task_info = self.df_tasks_relative.iloc[0].to_numpy()
        # future_occup = self.future_occup.flatten(order="F")

        # reset the idle resource capacities for all nodes
        for node in df_nodes.iterrows():
            self.full_resource_capacities[node[0]] = [
                [df_nodes.loc[node[0], 'CPU'] for _ in range(self.duration)],
                [df_nodes.loc[node[0], 'RAM'] for _ in range(self.duration)],
                [df_nodes.loc[node[0], 'storage'] for _ in range(
                    self.duration)]]

        self.idle_resource_capacities = copy.deepcopy(
            self.full_resource_capacities)

        # the action history of agents
        action_history = np.array([1 for _ in range(self.history_len) for _ in
                                   range(self.n_nodes)])

        # logging.debug(f"record_history?: {self.record_history}")
        for i in range(self.n_nodes):
            future_occup = self.future_occup[i].flatten(order="F")
            if self.record_history:
                agent_state = {
                    'task_info': task_info,
                    'future_occup': future_occup,
                    'action_history': action_history
                    }
            else:
                agent_state = {
                    'task_info': task_info,
                    'future_occup': future_occup,
                    }
            # logging.debug("The original dictionary is : " + str(agent_state))
            # convert dict to list
            agent_obs = list(chain(*agent_state.values()))
            # logging.debug("The Concatenated list values are : " + str(agent_state))

            self.obs[f'drone_{i}'] = agent_obs
            self.obs_dim = len(agent_obs)
            self.state[f'drone_{i}'] = agent_state

        logging.debug(f"Tasks information: \n{df_tasks}")
        logging.debug(f"Tasks information (relative):\n{self.df_tasks_relative}")
        logging.debug(f"Nodes information: \n{df_nodes}")
        logging.debug(f"observation after reset(): \n {self.obs}")

        return self.obs

    def step(self, actions):
        """
        Step function for the environment.
        Args:
            actions: a list of bid actions from all agents (e.g., 0.5 means bid 0.5 VC
            of the current task)

        Returns:
            observation (object): next observation?
            rewards (float): rewards of previous actions
            done (boolean): whether to reset the environment
            info (dict): diagnostic information for debugging.

        """

        # update the information of the current task
        self.current_task = self.df_tasks.loc[self.current_task_id]
        # current timeslot is where the start time of current task in
        self.current_time_slot = int(
            self.df_tasks.loc[self.current_task_id, "arrive_time"])
        self.next_time_slot = int(
            self.df_tasks.loc[self.current_task_id + 1, "arrive_time"])

        logging.debug(f"Current time slot = {self.current_time_slot}")
        logging.debug(
            f"Time slot followed by next task's arrival = {self.next_time_slot}")
        logging.debug(f"Last actions:\n {actions}")

        bids_list = []  # bid price for one time step
        max_usage_time_list = []  # maximum usage time a fog node can offer
        start_time_list = []  # start time according to the planned allocation
        relative_start_time_list = []  # relative start time according to the current task
        sw_increase_list = []
        # calculate the maximum usage time and earliest start time for each agent
        for node_id, (node_name, action) in enumerate(actions.items()):
            max_usage_time, relative_start_time = self.find_max_usage_time(
                node_id)
            # if node_id==0:
            #     logging.debug(f"current task VC = {self.current_task['valuation_coefficient']}")
            #     logging.debug(f"usage time of node 0 = {max_usage_time}")
            start_time = int(
                self.df_tasks.loc[
                    self.current_task_id, 'arrive_time'] + relative_start_time + 1)
            # action is in {1,2,...,9,10}
            # * 0.9 to avoid bidding the same as the value_coefficient
            bids_list.append((action + 0.5) * self.df_tasks.loc[
                self.current_task_id, 'valuation_coefficient'])
            # if action == 1:
            #     bids_list.append(0.8 * self.df_tasks.loc[
            #         self.current_task_id, 'valuation_coefficient'])
            # else:
            #     bids_list.append(0.2 * self.df_tasks.loc[
            #         self.current_task_id, 'valuation_coefficient'])
            max_usage_time_list.append(max_usage_time)
            start_time_list.append(start_time)
            relative_start_time_list.append(relative_start_time)

            cost = self.df_tasks.loc[self.current_task_id, 'CPU'] * self.df_nodes.loc[
                node_id, 'CPU_cost'] + self.df_tasks.loc[self.current_task_id, 'RAM'] * \
                   self.df_nodes.loc[node_id, 'RAM_cost'] + self.df_tasks.loc[
                       self.current_task_id, 'storage'] * self.df_nodes.loc[
                       node_id, 'storage_cost']
            value = self.df_tasks.loc[self.current_task_id, 'valuation_coefficient']
            sw_inc = (value - cost) * max_usage_time
            sw_increase_list.append(sw_inc)

        logging.debug("All nodes have submitted their bids:")
        logging.debug("actions of all nodes:")
        logging.debug(actions)
        logging.debug("bid prices:")
        logging.debug(bids_list)
        logging.debug("max usage times:")
        logging.debug(max_usage_time_list)
        logging.debug("sw increases for different nodes:")
        logging.debug(sw_increase_list)
        logging.debug("start times:")
        logging.debug(start_time_list)
        logging.debug("relative start times:")
        logging.debug(relative_start_time_list)

        # find the winner
        (winner_index, winner_usage_time, winner_revenue, max_utility,
         sw_increase) = self.reverse_auction(bids_list, max_usage_time_list,
                                             start_time_list,
                                             verbose=self.verbose,
                                             auction_type=self.auction_type)

        self.winner_id = winner_index
        self.winner_usage_time = winner_usage_time

        logging.debug(f"winner ID = {self.winner_id}")

        if winner_index is not None:
            # modify the allocation scheme
            winner_start_time = start_time_list[winner_index]
            self.winner_start_time = winner_start_time
            winner_relative_start_time = relative_start_time_list[winner_index]
            winner_finish_time = (winner_start_time + winner_usage_time - 1)
            self.winner_finish_time = winner_finish_time
            if winner_usage_time is not None and winner_usage_time > 0:
                self.allocation_scheme.loc[self.current_task_id] = [
                    winner_index,
                    winner_start_time, winner_finish_time]
            else:  # the task is rejected
                self.allocation_scheme.loc[self.current_task_id] = [None, None,
                                                                    None]

            # modify the occupancy of resources
            self.update_resource_occupency(winner_index, winner_usage_time,
                                           winner_relative_start_time)

            # # if the task is allocated to HCN when it can be allocated to LCNs
            # current_task_usage_time = self.df_tasks.loc[
            #     self.current_task_id, 'usage_time']

            # if task 0 wins the task while it is not gets the higest social welfare
            if sw_increase_list.index(max(sw_increase_list)) != 0:
                if self.winner_id == 0:
                    self.n_tasks_expensive += 1
                    logging.debug("This allocation is more expensive than needed.")
            logging.debug(f"allocation scheme:")
            logging.debug(f"{self.allocation_scheme.loc[self.current_task_id]}")
            logging.debug("idle resource capacities of winner node:")
            logging.debug(self.idle_resource_capacities[self.winner_id][:, 0:5])
            logging.debug("occupancy of future 10 time steps of the winner node")
            logging.debug(self.future_occup[self.winner_id])

        # # update the occupancy of resource (10 future time steps)
        # future_idle = np.divide(self.idle_resource_capacities[:, :,
        # self.next_time_slot + 1: (self.next_time_slot + 11)],
        #     self.full_resource_capacities[:, :,
        #     self.next_time_slot + 1: (self.next_time_slot + 11)])
        # logging.debug("idle resource capacities:")
        # logging.debug(self.idle_resource_capacities)
        # logging.debug("full resource capacities: ")
        # logging.debug(self.full_resource_capacities)

        self.future_occup = (
                1 - np.divide(self.idle_resource_capacities[:, :,
                              self.next_time_slot + 1: (self.next_time_slot + 11)],
                              self.full_resource_capacities[:, :,
                              self.next_time_slot + 1: (self.next_time_slot + 11)]))

        future_occup_len = len(self.future_occup[0][0])
        if future_occup_len < 10:
            # self.future_occup.resize((6, 3, 10))
            z = np.zeros((self.n_nodes, 3, 10 - future_occup_len))
            self.future_occup = np.concatenate((self.future_occup, z),
                                               axis=2)
            # for i in range(self.n_nodes):
            #     for j in range(3):
            #         logging.debug(f"Eco = {self.future_occup[i][j]}")
            #         self.future_occup[i][j] = self.future_occup[i][j].resize(10)
            #         self.future_occup[i][j][future_occup_len - 10:] = 1

        logging.debug("occupancy of future 10 time steps:")
        logging.debug(self.future_occup)

        # calcuate the total value of the current task
        self.current_task_value = self.current_task['valuation_coefficient'] * \
                                  self.current_task['usage_time']
        # logging.debug the updated allocation scheme_nd occupancy of resources
        logging.debug(f"current task ID = {self.current_task_id}")
        logging.debug(f"current task's value = {self.current_task_value}")
        logging.debug("current task info:")
        logging.debug(self.current_task)

        # update the  observation (obs)
        # a list in case different nodes have different rewards
        self.current_task_id += 1
        # reward is the penalty of the value lost
        # every node has the same reward
        #     # reward is the lost value
        #     equal_reward =sw_increase - self.current_task_value
        # reward is the SW increase
        equal_reward = sw_increase
        # only winner has the reward
        if self.cooperative:
            for i in range(self.n_nodes):
                self.rewards[f'drone_{i}'] = equal_reward
        else:
            for i in range(self.n_nodes):
                if i == self.winner_id:
                    self.rewards[f'drone_{i}'] = winner_revenue
                else:
                    self.rewards[f'drone_{i}'] = 0

        # find if this is the last task of the episode
        if self.current_task_id >= self.n_tasks_to_allocate - 1:
            dones = {'__all__': True}
            # log the allocation scheme after each episode
            logging.debug(f"Allocation scheme after an episode:")
            logging.debug(f"\n{self.allocation_scheme}")
        else:  # not the last step of the episode
            dones = {'__all__': False}
            # const = np.array([1])

        task_info = self.df_tasks_relative.iloc[
            self.current_task_id].to_numpy()
        # update the action_history
        logging.debug(f"self.state before updating = \n{self.state}")
        if self.record_history:
            action_history = deque(self.state['drone_0']['action_history'])
            logging.debug(f"previous action_history = {action_history}")
            action_history.rotate(-1)
            # actions_lst = list(actions.values())
            # use max usage times in history
            actions_lst = max_usage_time_list
            for i in range(self.n_nodes):
                action_history[self.history_len * (i + 1) - 1] = actions_lst[i]
            action_history = np.array(action_history)
            logging.debug(f"updated action_history = {action_history}")
        # generate the next observation
        for i in range(self.n_nodes):
            future_occup = self.future_occup[i].flatten(order="F")
            if self.record_history:
                agent_state = {
                    'task_info': task_info,
                    'future_occup': future_occup,
                    'action_history': action_history
                    }
            else:
                agent_state = {
                    'task_info': task_info,
                    'future_occup': future_occup,
                    }
            # logging.debug("The original dictionary is : " + str(agent_state))
            agent_obs = list(chain(*agent_state.values()))
            # logging.debug("The Concatenated list values are : " + str(agent_obs))

            self.state[f'drone_{i}'] = copy.deepcopy(agent_state)
            self.obs[f'drone_{i}'] = copy.deepcopy(agent_obs)

        logging.debug(f"self.state updated:\n{self.state}")
        # calculate rewards (may need to be changed to list of rewards in the future)
        self.sw_increase = sw_increase

        logging.debug("tasks (relative):")
        logging.debug(self.df_tasks_relative.head())
        logging.debug(f"next task ID = {self.current_task_id}")
        logging.debug("The info of the next task")
        logging.debug(task_info)
        logging.debug("social welfare increase")
        logging.debug(self.sw_increase)
        logging.debug("next global observation")
        logging.debug(self.obs)
        logging.debug(f"rewards for agents = {self.rewards}")
        logging.debug(f"Is the episode over? {dones}")
        logging.debug("\n\n")

        # update the total social welfare
        self.total_social_welfare += sw_increase
        if sw_increase > 0:
            self.total_allocated_tasks_num += 1

        # if this is the end of an episode
        if dones['__all__']:
            logging.debug(
                f"The number of expensive allocations = {self.n_tasks_expensive}")
            logging.debug(f"The total social welfare = {self.total_social_welfare}")
        # infos = {'node_0': f'social welfare increase = {sw_increase}'}
        infos = {}
        # info part is None for now
        # logging.debug(f"observation after step() = {self.obs}")
        return self.obs, self.rewards, dones, infos

    def render(self):
        logging.debug(f"current time slot = {self.current_time_slot}")
        logging.debug(f"next task ID = {self.current_task_id}")
        logging.debug(f"winner of last reverse_auction = {self.winner_id}")
        logging.debug(
            f"social welfare increase of last reverse_auction = {self.sw_increase}")
        logging.debug(f"rewards for all agents = {self.rewards}")
        logging.debug(f"Total social welfare = {self.total_social_welfare}")
        logging.debug(f"Next global observation = {self.obs}")

    def get_total_sw(self):
        return self.total_social_welfare

    def get_total_sw_online_myopic(self):
        return self.total_social_welfare

    def get_total_allocated_task_num(self):
        return self.total_allocated_tasks_num

    def get_num_bad_allocations(self):
        return self.n_tasks_expensive

    @staticmethod
    def render_method():
        return None

    @staticmethod
    def get_observation_space(config):
        if config['record_history']:
            obs_dim = 7 + config["usage_time_ub"] * 3 + config["history_len"] * 3
        else:
            obs_dim = 7 + config["usage_time_ub"] * 3

        return spaces.Box(low=0, high=10_000, shape=(obs_dim,),
                          dtype=np.float16)

    # @staticmethod
    # def get_action_space(config):
    #     return spaces.Discrete(config["n_actions"])

    @staticmethod
    def get_action_space(config):
        return spaces.Discrete(config['n_actions'])

    def update_resource_occupency(self, winner_index, winner_usage_time,
                                  winner_relative_start_time):
        """update idle resource capacities according to the reverse_auction result"""
        for i in range(winner_usage_time):
            self.idle_resource_capacities[winner_index][0][
                self.current_time_slot + winner_relative_start_time + 1 + i] -= (
                self.df_tasks.loc[self.current_task_id, 'CPU'])
            self.idle_resource_capacities[winner_index][1][
                self.current_time_slot + winner_relative_start_time + 1 + i] -= (
                self.df_tasks.loc[self.current_task_id, 'RAM'])
            self.idle_resource_capacities[winner_index][2][
                self.current_time_slot + winner_relative_start_time + 1 + i] -= (
                self.df_tasks.loc[self.current_task_id, 'storage'])

    def find_max_usage_time(self, node_id):
        """find the largest number of time steps this node can offer

                Returns:
                    max_time_length: maximum time steps this task can run on this agent
                    start_time: the start time of the task according to its allocation
                    scheme
                """

        # calculate idle resource capacity of future 10 time steps
        resource_capacity = self.resource_capacity_dict[node_id]
        logging.debug(
            f"resource capacity of this node = {self.resource_capacity_dict[node_id]}")
        logging.debug(f"future occupancy of this node = {self.future_occup[node_id]}")
        # ‘F’ means to flatten in column-major (Fortran- style) order.
        occup_resource_future = resource_capacity * self.future_occup[
            node_id].flatten(order="F")
        idle_resource_future = resource_capacity - occup_resource_future
        logging.debug(f"node {node_id}'s idle resource:")
        logging.debug(idle_resource_future)

        max_time_length = 0

        # Try time length from the requested number of time steps till it is possible
        # to allocate
        for time_length in reversed(
                range(1, int(self.current_task["usage_time"] + 1))):
            for start_time in range(int(self.current_task['start_time'] -
                                        self.current_task['arrive_time']),
                                    int(self.current_task['deadline'] -
                                        self.current_task['arrive_time'] -
                                        time_length + 2)):

                # logging.debug(f"time length: {time_length}, start time: {start_time}")
                # logging.debug(f"remaining resource: {remaining_resource}")
                can_allocate = True  # can allocate in this case?
                for i in range(0, time_length):
                    if idle_resource_future[(start_time + i) * 3] < \
                            self.current_task['CPU']:
                        can_allocate = False
                    if idle_resource_future[(start_time + i) * 3 + 1] < \
                            self.current_task['RAM']:
                        can_allocate = False
                    if idle_resource_future[(start_time + i) * 3 + 2] < \
                            self.current_task['storage']:
                        can_allocate = False
                if can_allocate:
                    max_time_length = time_length
                    break
            else:
                continue
            break

        return max_time_length, start_time

    def reverse_auction(self, bid_price_arr, bid_usage_time_arr,
                        bid_start_time_arr,
                        verbose=False, auction_type="first-price"):
        """Decides the reverse_auction result of this task

        Args:
        :param verbose: whether logging.debug the procedure
        :param bid_start_time_arr: array of start times
        :param bid_price_arr: array of bids
        :param bid_usage_time_arr: array of maximum usage times
        :param auction_type: "second-price" or "first-price" reverse reverse_auction
        """

        logging.debug("The reverse auction starts:")
        # find the winner of the reverse_auction
        valuation_coefficient = self.df_tasks.loc[
            self.current_task_id, "valuation_coefficient"]
        utility_arr = np.multiply(bid_usage_time_arr,
                                  (valuation_coefficient - bid_price_arr))
        max_utility = np.amax(utility_arr)  # maximum utility for this task
        if max_utility - 0.01 < 0:  # if bidding prices are all too high, basically means max_utility <= 0
            winner_index = None
            winner_usage_time = None
            winner_revenue = None
            sw_increase = 0
            logging.debug(
                "All bidding prices are higher than the unit value of the task!")
        else:
            # which FN wins this task
            winner_index = argmax_earliest(utility_arr=utility_arr,
                                           bid_start_time_arr=bid_start_time_arr)
            winner_usage_time = bid_usage_time_arr[
                winner_index]  # no. of time steps for this task
            winner_cost = (self.df_nodes.loc[winner_index, 'CPU_cost'] *
                           self.df_tasks.loc[self.current_task_id, 'CPU']
                           + self.df_nodes.loc[winner_index, 'RAM_cost'] *
                           self.df_tasks.loc[self.current_task_id, "RAM"]
                           + self.df_nodes.loc[winner_index, 'storage_cost'] *
                           self.df_tasks.loc[
                               self.current_task_id, "storage"]) * winner_usage_time
            # logging.debug(f"cost of the winner is {winner_cost}")

            # get winner's rewards
            if auction_type == "first-price":
                winner_revenue = (winner_usage_time * bid_price_arr[
                    winner_index] - winner_cost)
            elif auction_type == "second-price":
                bids = {'prices': bid_price_arr,
                        'times': bid_usage_time_arr}
                df_bids = pd.DataFrame(bids, columns=['prices', 'times'])
                df_bids = df_bids.sort_values('prices')
                second_price = bid_price_arr[winner_index]
                for i in range(self.n_nodes):
                    if df_bids.iloc[i, 0] > second_price and df_bids.iloc[
                        i, 1] > 0:  # if the price is greater than winner's and the
                        # usage time is not zero
                        second_price = df_bids.iloc[i, 0]
                        break
                winner_revenue = (
                        winner_usage_time * second_price - winner_cost)
                logging.debug("Dataframe of the bids:")
                logging.debug(df_bids)
                logging.debug(f"second-price={second_price}")

            else:
                raise ValueError("unrecognised reverse_auction type")
            logging.debug(f"ID of the winner = {winner_index}")
            logging.debug(f"operational cost of the winner = {winner_cost}")
            logging.debug(f"winner's revenue = {winner_revenue}")

            # update social welfare
            sw_increase = valuation_coefficient * winner_usage_time - winner_cost
        return winner_index, winner_usage_time, winner_revenue, max_utility, sw_increase


class EdgeCloudEnv1(EdgeCloudEnv):
    """an Environment with low-dimensional states(observations)"""

    def reset(self):
        """Reset the tasks and nodes

        Returns the initial global observation
        """
        # occupancy is zero initially
        self.future_occup = np.zeros((self.n_nodes, self.occup_len, 3))
        self._episode_ended = False
        self.current_task_id = 0
        self.processed_tasks = 0
        self.failed = 0
        self.total_social_welfare = 0
        self.total_allocated_tasks_num = 0
        self.n_tasks_expensive = 0

        # generate new tasks for next episode
        (df_tasks, df_nodes, n_time, n_tasks,
         n_nodes) = self.data_for_next_episode()
        self.df_tasks = df_tasks
        self.df_nodes = df_nodes
        # for benchmark
        self.df_tasks_bench = copy.deepcopy(df_tasks.iloc[0:self.n_tasks_to_allocate])
        self.df_nodes_bench = copy.deepcopy(df_nodes)
        logging.debug(f"df_tasks for benchmark:\n{self.df_tasks_bench}")
        # self.df_tasks_bench = self.df_tasks_bench.rename(columns={"storage": "DISK"})
        # self.df_nodes_bench = self.df_nodes_bench.rename(
        #     columns={"storage": "DISK", "storage_cost": "DISK_cost"})
        self.n_time_bench = n_time
        self.n_tasks_bench = self.n_tasks_to_allocate - 1
        self.n_nodes_bench = n_nodes
        self.current_task = df_tasks.iloc[0]
        # make time constraints relative
        self.df_tasks_relative = self.df_tasks.iloc[0:self.n_tasks_to_allocate].copy()
        self.df_tasks_relative["relative_start_time"] = (
                self.df_tasks_relative['start_time'] -
                self.df_tasks_relative['arrive_time'].astype(int) - 1)
        self.df_tasks_relative["relative_deadline"] = (
                self.df_tasks_relative["deadline"] -
                self.df_tasks_relative["start_time"] + 1)
        self.df_tasks_relative.drop('start_time', inplace=True, axis=1)
        self.df_tasks_relative.drop('deadline', inplace=True, axis=1)
        self.df_tasks_relative.drop('arrive_time', inplace=True, axis=1)
        # # "0 max" normalisation
        # self.df_tasks_normalised = (self.df_tasks_relative - 0) / (
        #         self.df_tasks_relative.max() - 0)

        self.seed_value += 1  # may need to make this random in training

        # current timeslot is where the start time of current task in
        self.current_time_slot = int(self.df_tasks.loc[0, "arrive_time"])
        # generate a dict of resource capacity of future 4 time slots
        self.resource_capacity_dict = {}
        for node_id, info in df_nodes.iterrows():
            a = [info.get('CPU'), info.get('RAM'), info.get('storage')]
            self.resource_capacity_dict[node_id] = np.array(a * self.occup_len)

        # the obs is a vector of current task information and the future occupancy
        # const = np.array([1])
        task_info = self.df_tasks_relative.iloc[0].to_numpy()
        # future_occup = self.future_occup.flatten(order="F")

        # reset the idle resource capacities for all nodes
        for node in df_nodes.iterrows():
            self.full_resource_capacities[node[0]] = [
                [df_nodes.loc[node[0], 'CPU'] for _ in range(self.duration)],
                [df_nodes.loc[node[0], 'RAM'] for _ in range(self.duration)],
                [df_nodes.loc[node[0], 'storage'] for _ in range(
                    self.duration)]]

        self.idle_resource_capacities = copy.deepcopy(
            self.full_resource_capacities)

        # the action history of agents
        action_history = np.array([1 for _ in range(self.history_len) for _ in
                                   range(self.n_nodes)])

        # logging.debug(f"record_history?: {self.record_history}")
        for i in range(self.n_nodes):
            future_occup = self.future_occup[i].flatten(order="F")
            if self.record_history:
                agent_state = {
                    'task_info': task_info,
                    'future_occup': future_occup,
                    'action_history': action_history
                    }
            else:
                agent_state = {
                    'task_info': task_info,
                    'future_occup': future_occup,
                    }
            # logging.debug("The original dictionary is : " + str(agent_state))
            # convert dict to list
            agent_obs = list(chain(*agent_state.values()))
            # logging.debug("The Concatenated list values are : " + str(agent_state))

            self.obs[f'drone_{i}'] = agent_obs
            self.obs_dim = len(agent_obs)
            self.state[f'drone_{i}'] = agent_state

        logging.debug(f"Tasks information: \n{df_tasks}")
        logging.debug(f"Tasks information (relative):\n{self.df_tasks_relative}")
        logging.debug(f"Nodes information: \n{df_nodes}")
        logging.debug(f"observation after reset(): \n {self.obs}")

        return self.obs

    def step(self, actions):
        """
        Step function for the environment.
        Args:
            actions: a list of bid actions from all agents (e.g., 0.5 means bid 0.5 VC
            of the current task)

        Returns:
            observation (object): next observation?
            rewards (float): rewards of previous actions
            done (boolean): whether to reset the environment
            info (dict): diagnostic information for debugging.

        """

        # update the information of the current task
        global action_history
        self.current_task = self.df_tasks.loc[self.current_task_id]
        # current timeslot is where the start time of current task in
        self.current_time_slot = int(
            self.df_tasks.loc[self.current_task_id, "arrive_time"])
        self.next_time_slot = int(
            self.df_tasks.loc[self.current_task_id + 1, "arrive_time"])

        logging.debug(f"Current time slot = {self.current_time_slot}")
        logging.debug(
            f"Time slot followed by next task's arrival = {self.next_time_slot}")
        logging.debug(f"Last actions:\n {actions}")

        bids_list = []  # bid price for one time step
        max_usage_time_list = []  # maximum usage time a fog node can offer
        start_time_list = []  # start time according to the planned allocation
        relative_start_time_list = []  # relative start time according to the current task
        sw_increase_list = []
        # calculate the maximum usage time and earliest start time for each agent
        for node_id, (node_name, action) in enumerate(actions.items()):
            max_usage_time, relative_start_time = self.find_max_usage_time(
                node_id)
            # if node_id==0:
            #     logging.debug(f"current task VC = {self.current_task['valuation_coefficient']}")
            #     logging.debug(f"usage time of node 0 = {max_usage_time}")
            start_time = int(
                self.df_tasks.loc[
                    self.current_task_id, 'arrive_time'] + relative_start_time + 1)
            # action is in {0,1,2,...,9}
            # the smalles action means bidding 0
            if action == 0:
                bid = 0
                bids_list.append(bid)
            else:  # the biggest action means bidding the maximum VC
                bid = (action + 1) / self.n_actions * self.df_tasks.loc[
                    self.current_task_id, 'valuation_coefficient']
                bids_list.append(bid)
            # if action == 1:
            #     bids_list.append(0.8 * self.df_tasks.loc[
            #         self.current_task_id, 'valuation_coefficient'])
            # else:
            #     bids_list.append(0.2 * self.df_tasks.loc[
            #         self.current_task_id, 'valuation_coefficient'])
            max_usage_time_list.append(max_usage_time)
            start_time_list.append(start_time)
            relative_start_time_list.append(relative_start_time)

            cost = self.df_tasks.loc[self.current_task_id, 'CPU'] * self.df_nodes.loc[
                node_id, 'CPU_cost'] + self.df_tasks.loc[self.current_task_id, 'RAM'] * \
                   self.df_nodes.loc[node_id, 'RAM_cost'] + self.df_tasks.loc[
                       self.current_task_id, 'storage'] * self.df_nodes.loc[
                       node_id, 'storage_cost']
            value = self.df_tasks.loc[self.current_task_id, 'valuation_coefficient']
            logging.debug(f"node ID = {node_id}")
            logging.debug(f"valuation coefficient for current task = {value}")
            logging.debug(f"cost for current task for this node = {cost}")

            sw_inc = (value - cost) * max_usage_time
            sw_increase_list.append(sw_inc)

        logging.debug("All nodes have submitted their bids:")
        logging.debug("actions of all nodes:")
        logging.debug(actions)
        logging.debug("bid prices:")
        logging.debug(bids_list)
        logging.debug("max usage times:")
        logging.debug(max_usage_time_list)
        logging.debug("sw increases for different nodes:")
        logging.debug(sw_increase_list)
        logging.debug("start times:")
        logging.debug(start_time_list)
        logging.debug("relative start times:")
        logging.debug(relative_start_time_list)

        # find the winner
        (winner_index, winner_usage_time, winner_revenue, max_utility,
         sw_increase) = self.reverse_auction(bids_list, max_usage_time_list,
                                             start_time_list,
                                             verbose=self.verbose,
                                             auction_type=self.auction_type)

        self.winner_id = winner_index
        self.winner_usage_time = winner_usage_time

        logging.debug(f"winner ID = {self.winner_id}")

        if winner_index is not None:
            # modify the allocation scheme
            winner_start_time = start_time_list[winner_index]
            self.winner_start_time = winner_start_time
            winner_relative_start_time = relative_start_time_list[winner_index]
            winner_finish_time = (winner_start_time + winner_usage_time - 1)
            self.winner_finish_time = winner_finish_time
            if winner_usage_time is not None and winner_usage_time > 0:
                self.allocation_scheme.loc[self.current_task_id] = [
                    winner_index,
                    winner_start_time, winner_finish_time]
            else:  # the task is rejected
                self.allocation_scheme.loc[self.current_task_id] = [None, None,
                                                                    None]

            # modify the occupancy of resources
            self.update_resource_occupency(winner_index, winner_usage_time,
                                           winner_relative_start_time)

            # # if the task is allocated to HCN when it can be allocated to LCNs
            # current_task_usage_time = self.df_tasks.loc[
            #     self.current_task_id, 'usage_time']

            # if task 0 wins the task while it is not gets the higest social welfare
            if sw_increase_list.index(max(sw_increase_list)) != 0:
                if self.winner_id == 0:
                    self.n_tasks_expensive += 1
                    logging.debug("This allocation is more expensive than needed.")
            logging.debug(f"allocation scheme:")
            logging.debug(f"{self.allocation_scheme.loc[self.current_task_id]}")
            logging.debug("idle resource capacities of winner node:")
            logging.debug(self.idle_resource_capacities[self.winner_id][:, 0:5])
            logging.debug("occupancy of future 10 time steps of the winner node")
            logging.debug(self.future_occup[self.winner_id])

        # # update the occupancy of resource (10 future time steps)
        # future_idle = np.divide(self.idle_resource_capacities[:, :,
        # self.next_time_slot + 1: (self.next_time_slot + 11)],
        #     self.full_resource_capacities[:, :,
        #     self.next_time_slot + 1: (self.next_time_slot + 11)])
        # logging.debug("idle resource capacities:")
        # logging.debug(self.idle_resource_capacities)
        # logging.debug("full resource capacities: ")
        # logging.debug(self.full_resource_capacities)

        self.future_occup = (
                1 - np.divide(self.idle_resource_capacities[:, :,
                              self.next_time_slot + 1: (
                                      self.next_time_slot + self.occup_len + 1)],
                              self.full_resource_capacities[:, :,
                              self.next_time_slot + 1: (
                                      self.next_time_slot + self.occup_len + 1)]))

        future_occup_len = len(self.future_occup[0][0])
        if future_occup_len < self.occup_len:
            # self.future_occup.resize((6, 3, 10))
            z = np.zeros((self.n_nodes, 3, self.occup_len - future_occup_len))
            self.future_occup = np.concatenate((self.future_occup, z),
                                               axis=2)
            # for i in range(self.n_nodes):
            #     for j in range(3):
            #         logging.debug(f"Eco = {self.future_occup[i][j]}")
            #         self.future_occup[i][j] = self.future_occup[i][j].resize(10)
            #         self.future_occup[i][j][future_occup_len - 10:] = 1

        logging.debug(f"occupancy of future {self.occup_len} time steps:")
        logging.debug(self.future_occup)

        # calcuate the total value of the current task
        self.current_task_value = self.current_task['valuation_coefficient'] * \
                                  self.current_task['usage_time']
        # logging.debug the updated allocation scheme_nd occupancy of resources
        logging.debug(f"current task ID = {self.current_task_id}")
        logging.debug(f"current task's value = {self.current_task_value}")
        logging.debug("current task info:")
        logging.debug(self.current_task)

        # update the  observation (obs)
        # a list in case different nodes have different rewards
        self.current_task_id += 1
        # reward is the penalty of the value lost
        # every node has the same reward
        #     # reward is the lost value
        #     equal_reward =sw_increase - self.current_task_value
        # reward is the SW increase
        equal_reward = sw_increase
        if self.cooperative:
            for i in range(self.n_nodes):
                self.rewards[f'drone_{i}'] = equal_reward
        else:  # only winner has the reward
            for i in range(self.n_nodes):
                if i == self.winner_id:
                    self.rewards[f'drone_{i}'] = winner_revenue
                else:
                    self.rewards[f'drone_{i}'] = 0

        # find if this is the last task of the episode
        if self.current_task_id >= self.n_tasks_to_allocate - 1:
            dones = {'__all__': True}
            # log the allocation scheme after each episode
            logging.debug(f"Allocation scheme after an episode:")
            logging.debug(f"\n{self.allocation_scheme}")
        else:  # not the last step of the episode
            dones = {'__all__': False}
            # const = np.array([1])

        task_info = self.df_tasks_relative.iloc[
            self.current_task_id].to_numpy()
        # update the action_history
        logging.debug(f"self.state before updating = \n{self.state}")
        if self.record_history:
            action_history = deque(self.state['drone_0']['action_history'])
            logging.debug(f"previous action_history = {action_history}")
            action_history.rotate(-1)
            # actions_lst = list(actions.values())
            # use max usage times in history
            actions_lst = max_usage_time_list
            for i in range(self.n_nodes):
                action_history[self.history_len * (i + 1) - 1] = actions_lst[i]
            action_history = np.array(action_history)
            logging.debug(f"updated action_history = {action_history}")
        # generate the next observation
        for i in range(self.n_nodes):
            future_occup = self.future_occup[i].flatten(order="F")
            if self.record_history:
                agent_state = {
                    'task_info': task_info,
                    'future_occup': future_occup,
                    'action_history': action_history
                    }
            else:
                agent_state = {
                    'task_info': task_info,
                    'future_occup': future_occup,
                    }
            # logging.debug("The original dictionary is : " + str(agent_state))
            agent_obs = list(chain(*agent_state.values()))
            # logging.debug("The Concatenated list values are : " + str(agent_obs))

            self.state[f'drone_{i}'] = copy.deepcopy(agent_state)
            self.obs[f'drone_{i}'] = copy.deepcopy(agent_obs)

        logging.debug(f"self.state updated:\n{self.state}")
        # calculate rewards (may need to be changed to list of rewards in the future)
        self.sw_increase = sw_increase

        logging.debug("tasks (relative):")
        logging.debug(self.df_tasks_relative.head())
        logging.debug(f"next task ID = {self.current_task_id}")
        logging.debug("The info of the next task")
        logging.debug(task_info)
        logging.debug("social welfare increase")
        logging.debug(self.sw_increase)
        logging.debug("next global observation")
        logging.debug(self.obs)
        logging.debug(f"rewards for agents = {self.rewards}")
        logging.debug(f"Is the episode over? {dones}")
        logging.debug("\n\n")

        # update the total social welfare
        self.total_social_welfare += sw_increase
        if sw_increase > 0:
            self.total_allocated_tasks_num += 1

        # if this is the end of an episode, run all the benchmarks
        if dones['__all__']:
            logging.debug(
                f"The number of expensive allocations = {self.n_tasks_expensive}")
            logging.debug(f"The total social welfare = {self.total_social_welfare}")
            # run online myopic algo.
            social_welfare_bench, number_of_allocated_tasks_bench, allocation_scheme_bench = \
                online_myopic(self.df_tasks_bench, self.df_nodes_bench, self.n_time_bench,
                              self.n_tasks_bench, self.n_nodes_bench)
            self.social_welfare_online_myopic = social_welfare_bench
            logging.debug(f"social welfare (online myopic): {social_welfare_bench}")
            logging.debug(
                f"number of allocated tasks (online myopic): {number_of_allocated_tasks_bench}")
            # run random allocation algo.
            social_welfare_random_allocation, number_of_allocated_tasks_random_allocation, allocation_scheme_random_allocation = \
                random_allocation(self.df_tasks_bench, self.df_nodes_bench,
                                  self.n_time_bench,
                                  self.n_tasks_bench, self.n_nodes_bench)
            self.social_welfare_random_allocation = social_welfare_random_allocation
            logging.debug(
                f"social welfare (random_allocation): {social_welfare_random_allocation}")
            logging.debug(
                f"number of allocated tasks (random_allocation): {number_of_allocated_tasks_random_allocation}")

            # run the all bidding zero auction
            social_welfare_bidding_zero, number_of_allocated_tasks_bidding_zero, allocation_scheme_bidding_zero = \
                all_bidding_zero(self.df_tasks_bench, self.df_nodes_bench,
                                 self.n_time_bench,
                                 self.n_tasks_bench, self.n_nodes_bench)
            self.social_welfare_bidding_zero = social_welfare_bidding_zero
            logging.debug(
                f"social welfare (bidding_zero): {social_welfare_bidding_zero}")
            logging.debug(
                f"number of allocated tasks (bidding_zero): {number_of_allocated_tasks_bidding_zero}")

        # infos = {'node_0': f'social welfare increase = {sw_increase}'}
        infos = {}
        # info part is None for now
        # logging.debug(f"observation after step() = {self.obs}")
        return self.obs, self.rewards, dones, infos

    def get_total_sw_online_myopic(self):
        return self.social_welfare_online_myopic

    def get_total_sw_random_allocation(self):
        return self.social_welfare_random_allocation

    def get_total_sw_bidding_zero(self):
        return self.social_welfare_bidding_zero


class GlobalObsEdgeCloudEnv(MultiAgentEnv):
    # example source (https://github.com/ray-project/ray/blob/ced062319dca261b72b42d78048a167818c1f729/rllib/examples/centralized_critic_2.py#L73)

    action_space = Discrete(2)
    observation_space = Dict({
        "own_obs": spaces.Box(low=0, high=1, shape=(37,)),
        "opponent_obs": spaces.Box(low=0, high=1, shape=(37,)),
        "opponent_action": Discrete(2),
        })

    def __init__(self, env_config):
        self.env = EdgeCloudEnv(env_config)

    def reset(self):
        obs_dict = self.env.reset()
        return self.to_global_obs(obs_dict)

    def step(self, action_dict):
        obs_dict, rewards, dones, infos = self.env.step(action_dict)
        return self.to_global_obs(obs_dict), rewards, dones, infos

    def _to_global_obs(self, obs_dict):
        """helper function to return the global observation
        """
        return {
            self.env.agent_1: {
                "own_obs": obs_dict[self.env.agent_1],
                "opponent_obs": obs_dict[self.env.agent_2],
                "opponent_action": 0,  # populated by fill_in_actions
                },
            self.env.agent_2: {
                "own_obs": obs_dict[self.env.agent_2],
                "opponent_obs": obs_dict[self.env.agent_1],
                "opponent_action": 0,  # populated by fill_in_actions
                },
            }
