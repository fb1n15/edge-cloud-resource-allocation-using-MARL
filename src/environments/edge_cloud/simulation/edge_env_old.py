import copy

import gym
import numpy as np
import pandas as pd
from gym import spaces

from scripts.generate_simulation_data import generate_synthetic_data_edge_cloud
from scripts.online_myopic_m import online_myopic


class EdgeEnv(gym.Env):
    """Edge cloud resource allocation environment follows gym interfact"""

    def __init__(self, avg_resource_capacity, avg_unit_cost,
            seed=0, n_nodes=6, n_timesteps=20, n_tasks=40,
            max_steps=20, n_actions=1,
            p_high_value_tasks=0.1, high_value_slackness=0,
            low_value_slackness=0, resource_ratio=3, valuation_ratio=10,
            resource_coefficient=0.2,
            forgiveness_factor=30,
            allow_negative_reward=False,
            alpha=1.0, lam=1e2, verbose=True):
        """
        Initialization function for the environment.
        Args:

            seed: seed for generating simulation data
            n_timesteps The number of timestamps.
            allow_negative_reward: Flag for allowing negative rewards for the bad allocation of tasks.
            forgiveness_factor: Tolerance to sequential bad allocation of tasks.
            alpha: Percentage of the total rewards influenced by the prioritisation of high valuation tasks.
            lam: Speed of increase in the rewards generated from the prioritisation of high valuation tasks.
            not_verbose: A boolean as a flag to print information about the node allocation.

        """

        # Set the class variables
        super().__init__()

        self.avg_resource_capacity = avg_resource_capacity
        self.avg_unit_cost = avg_unit_cost
        self.max_steps = max_steps

        resource_coefficient = (
                resource_coefficient * n_tasks / n_timesteps)
        self.n_tasks = n_tasks
        self.n_timesteps = n_timesteps
        self.seed = seed
        self.n_nodes = n_nodes
        self.n_actions = n_actions
        self.p_high_value_tasks = p_high_value_tasks
        self.high_value_slackness = high_value_slackness
        self.low_value_slackness = low_value_slackness
        self.resource_ratio = resource_ratio
        self.valuation_ratio = valuation_ratio
        self.resource_coefficient = resource_coefficient
        self.auction_type = 'first-price'
        self.verbose = verbose
        # record the allocation scheme (task_id: [node_id, start_time, end_time])
        self.allocation_scheme = pd.DataFrame(
            columns=['node_id', 'start_time', 'end_time'])

        (df_tasks, df_nodes, n_time, n_tasks,
        n_nodes) = self.data_for_next_episode()

        self.allocated_tasks = []  # [(task_info, action)]

        self.lam = lam
        self.alpha = alpha
        self.allow_negative_reward = allow_negative_reward
        self.n_timesteps = n_timesteps
        self.n_resource_type = 3

        # initialise the ndarray of idle resources
        self.full_resource_capacities = np.empty(
            [self.n_nodes, self.n_resource_type, self.n_timesteps])

        for node in df_nodes.iterrows():
            self.full_resource_capacities[node[0]] = [
                [df_nodes.loc[node[0], 'CPU'] for _ in range(n_timesteps)],
                [df_nodes.loc[node[0], 'RAM'] for _ in range(n_timesteps)],
                [df_nodes.loc[node[0], 'storage'] for _ in range(n_timesteps)]]

        self.df_tasks = df_tasks
        self.df_nodes = df_nodes

        self.current_task_id = 0
        self.failed = 0

        self.forgiveness_factor = forgiveness_factor

        self.total_social_welfare = 0

        self.processed_tasks = 0
        self.action_history = []

        self.allocation_map = dict((node, []) for node in range(
            self.n_nodes))  # A dict with the shape {node: [(task, start_index, stop_index),]}

        self.idle_resource_capacities = copy.deepcopy(
            self.full_resource_capacities)

        # 10 discrete actions for each agent
        self.action_space = [spaces.Discrete(n_actions) for _ in
            range(self.n_nodes)]

        # observation is the occupancy of future 10 time steps (because the relative deadline of all tasks <= 10)
        # this is the global observation, will modify to local observations in the future
        self.observation_space = [spaces.Box(low=0, high=1,
            shape=(37,), dtype=np.float16) for _ in range(self.n_nodes)]

        self._episode_ended = False
        # an upper bound of the social welfare
        self.total_possible_reward = sum(
            df_tasks.valuation_coefficient * df_tasks.usage_time)

    def data_for_next_episode(self):
        df_tasks, df_nodes, n_time, n_tasks, n_nodes = \
            generate_synthetic_data_edge_cloud(self.avg_resource_capacity,
                self.avg_unit_cost, n_tasks=self.n_tasks,
                n_time=self.n_timesteps,
                seed=self.seed, n_nodes=self.n_nodes,
                p_high_value_tasks=self.p_high_value_tasks,
                high_value_slackness_lower_limit=self.high_value_slackness,
                high_value_slackness_upper_limit=self.high_value_slackness,
                low_value_slackness_lower_limit=self.low_value_slackness,
                low_value_slackness_upper_limit=self.low_value_slackness,
                resource_demand_high=self.resource_ratio,
                vc_ratio=self.valuation_ratio,
                k_resource=self.resource_coefficient)
        return df_tasks, df_nodes, n_time, n_tasks, n_nodes

    def reset(self):
        """Reset the tasks and nodes

        Returns the initial global observation
        """
        # occupancy is zero initially
        self.future_occup = np.zeros((self.n_nodes, 10, 3))
        self._episode_ended = False
        self.current_task_id = 0
        self.processed_tasks = 0
        self.failed = 0
        self.total_social_welfare = 0

        # generate new tasks for next episode
        (df_tasks, df_nodes, n_time, n_tasks,
        n_nodes) = self.data_for_next_episode()
        self.df_tasks = df_tasks
        self.df_nodes = df_nodes
        self.current_task = df_tasks.iloc[0]
        # make time constraints relative
        self.df_tasks_relative = self.df_tasks.iloc[0:self.max_steps].copy()
        self.df_tasks_relative["relative_start_time"] = (
                self.df_tasks_relative['start_time'] -
                self.df_tasks_relative['arrive_time'].astype(int))
        self.df_tasks_relative["relative_deadline"] = (
                self.df_tasks_relative["deadline"] -
                self.df_tasks_relative["start_time"] + 1)
        self.df_tasks_relative.drop('start_time', inplace=True, axis=1)
        self.df_tasks_relative.drop('deadline', inplace=True, axis=1)
        self.df_tasks_relative.drop('arrive_time', inplace=True, axis=1)
        # "0 max" normalisation
        self.df_tasks_normalised = (self.df_tasks_relative - 0) / (
                self.df_tasks_relative.max() - 0)

        self.seed += 1  # may need to make this random in training

        # current timeslot is where the start time of current task in
        self.current_time_slot = int(self.df_tasks.loc[0, "arrive_time"])
        # generate a dict of resource capacity of future 10 time slots
        self.resource_capacity_dict = {}
        for node_id, info in df_nodes.iterrows():
            a = [info.get('CPU'), info.get('RAM'), info.get('storage')]
            self.resource_capacity_dict[node_id] = np.array(a * 10)

        # the state is a vector of current task information and the future occupancy
        # const = np.array([1])
        task_info = self.df_tasks_normalised.iloc[0].to_numpy()
        future_occup = self.future_occup.flatten(order="F")

        # reset the idle resource capacities for all nodes
        for node in df_nodes.iterrows():
            self.full_resource_capacities[node[0]] = [
                [df_nodes.loc[node[0], 'CPU'] for _ in range(self.n_timesteps)],
                [df_nodes.loc[node[0], 'RAM'] for _ in range(self.n_timesteps)],
                [df_nodes.loc[node[0], 'storage'] for _ in range(
                    self.n_timesteps)]]

        self.idle_resource_capacities = copy.deepcopy(
            self.full_resource_capacities)

        self.state = []
        for i in range(self.n_nodes):
            future_occup = self.future_occup[i].flatten(order="F")
            agent_state = np.concatenate((task_info, future_occup),
                axis=None)

            self.state.append(agent_state)

        # get the social welfare of Online Myopic
        if self.verbose:
            print("running Online Myopic")
        df_tasks = df_tasks[:20]
        n_tasks = 20
        if self.verbose:
            print(df_tasks)
        social_welfare, number_of_allocated_tasks, allocation_scheme = \
            online_myopic(df_tasks, df_nodes, n_time, n_tasks, n_nodes)
        if self.verbose:
            print("social welfare:", social_welfare)
            print("number of allocated tasks:", number_of_allocated_tasks)
            print(f"allocation_scheme:")
            print(allocation_scheme)

        return self.state, social_welfare

    def step(self, actions):
        """
        Step function for the environment.
        Args:
            actions: a list of bid actions from all agents (e.g., 0.5 means bid 0.5 VC of the current task)

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
        if self.verbose:
            print(f"current time slot = {self.current_time_slot}")
            print(f"next task's time slot = {self.next_time_slot}")

        bids_list = []  # bid price for one time step
        max_usage_time_list = []  # maximum usage time a fog node can offer
        start_time_list = []  # start time according to the planned allocation
        relative_start_time_list = []  # relative start time according to the current task
        # calculate the maximum usage time and earliest start time for each agent
        for node_id, action in enumerate(actions):
            max_usage_time, relative_start_time = self.find_max_usage_time(
                node_id)
            # if node_id==0:
            #     print(f"current task VC = {self.current_task['valuation_coefficient']}")
            #     print(f"usage time of node 0 = {max_usage_time}")
            start_time = int(
                self.df_tasks.loc[
                    self.current_task_id, 'arrive_time'] + relative_start_time + 1)
            # action is in {1,2,...,9,10}
            bids_list.append(action * self.df_tasks.loc[
                self.current_task_id, 'valuation_coefficient'])
            max_usage_time_list.append(max_usage_time)
            start_time_list.append(start_time)
            relative_start_time_list.append(relative_start_time)

        if self.verbose:
            print("bid prices:")
            print(bids_list)
            print("max usage times:")
            print(max_usage_time_list)
            print("start times:")
            print(start_time_list)
            print("relative start times:")
            print(relative_start_time_list)

        # find the winner
        (winner_index, winner_usage_time, winner_utility, max_utility,
        sw_increase) = self.reverse_auction(bids_list, max_usage_time_list,
            start_time_list,
            verbose=self.verbose, auction_type=self.auction_type)

        self.winner_id = winner_index

        if self.verbose:
            print(f"winner ID = {self.winner_id}")

        if winner_index is not None:
            # modify the allocation scheme
            winner_start_time = start_time_list[winner_index]
            winner_relative_start_time = relative_start_time_list[winner_index]
            winner_finish_time = (winner_start_time + winner_usage_time - 1)
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

            if self.verbose:
                print(f"allocation scheme:")
                print(f"{self.allocation_scheme.loc[self.current_task_id]}")
                print("idle resource capacities of winner node:")
                print(self.idle_resource_capacities[self.winner_id][:, 0:5])
                # print("occupancy of future 10 time steps of the winner node")
                # print(self.future_occup[self.winner_id])

        # # update the occupancy of resource (10 future time steps)
        # future_idle = np.divide(self.idle_resource_capacities[:, :,
        # self.next_time_slot + 1: (self.next_time_slot + 11)],
        #     self.full_resource_capacities[:, :,
        #     self.next_time_slot + 1: (self.next_time_slot + 11)])
        # print(self.idle_resource_capacities[5])
        # print(self.full_resource_capacities[5])
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
            #         print(f"Eco = {self.future_occup[i][j]}")
            #         self.future_occup[i][j] = self.future_occup[i][j].resize(10)
            #         self.future_occup[i][j][future_occup_len - 10:] = 1

        if self.verbose:
            print("occupancy of future 10 time steps:")
            # print(self.future_occup.shape)
            print(self.future_occup)

        # calcuate the total value of the current task
        self.current_task_value = self.current_task['valuation_coefficient'] * \
                                  self.current_task['usage_time']
        # print the updated allocation scheme_nd occupancy of resources
        if self.verbose:
            print(f"current task ID = {self.current_task_id}")
            print(f"current task's value = {self.current_task_value}")
            print("current task info:")
            print(self.current_task)

        # update the global observation (state)
        self.state = []  # observation is a list of ndarrays
        # a list in case different nodes have different rewards
        self.rewards = []
        self.current_task_id += 1
        # reward is the penalty of the value lost
        self.rewards.append(sw_increase - self.current_task_value)
        # find if this is the last task of the episode
        if self.current_task_id >= self.max_steps:
            done = True
            # no new observation at the end of the the episode
            task_info = None
            self.state = None
        else:  # not the last step of the episode
            done = False
            # const = np.array([1])
            task_info = self.df_tasks_normalised.iloc[
                self.current_task_id].to_numpy()
            for i in range(self.n_nodes):
                future_occup = self.future_occup[i].flatten(order="F")
                agent_state = np.concatenate((task_info, future_occup),
                    axis=None)

                self.state.append(agent_state)

            # calculate rewards (may need to be changed to list of rewards in the future)
        self.sw_increase = sw_increase

        if self.verbose:
            print("tasks normalised:")
            print(self.df_tasks_normalised.head())
            print(f"next task ID = {self.current_task_id}")
            print("The info of the next task")
            print(task_info)
            print("social welfare increase")
            print(self.sw_increase)
            print("next global observation")
            print(self.state)
            print(f"rewards for agents = {self.rewards}")
            print(f"Is the episode over? {done}")
            print("\n\n")

        # update the total social welfare
        self.total_social_welfare += sw_increase
        # info part is None for now
        return self.state, self.rewards, done, sw_increase

    def render(self):
        print(f"current time slot = {self.current_time_slot}")
        print(f"next task ID = {self.current_task_id}")
        print(f"winner of last reverse_auction = {self.winner_id}")
        print(
            f"social welfare increase of last reverse_auction = {self.sw_increase}")
        print(f"rewards for all agents = {self.rewards}")
        print(f"Total social welfare = {self.total_social_welfare}")
        print(f"Next global observation = {self.state}")
        print()

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
                    start_time: the start time of the task according to its allocation scheme
                """
        # calculate idle resource capacity of future 10 time steps
        resource_capacity = self.resource_capacity_dict[node_id]
        # ‘F’ means to flatten in column-major (Fortran- style) order.
        occup_resource_future = resource_capacity * self.future_occup[
            node_id].flatten(order="F")
        idle_resource_future = resource_capacity - occup_resource_future
        # print(f"node {node_id}'s idle resource:")
        # print(idle_resource_future)

        max_time_length = 0

        # Try time length from the requested number of time steps till it is possible to allocate
        for time_length in reversed(
                range(1, int(self.current_task["usage_time"] + 1))):
            for start_time in range(int(self.current_task['start_time'] -
                                        self.current_task['arrive_time']),
                    int(self.current_task['deadline'] -
                        self.current_task['arrive_time'] -
                        time_length + 2)):

                # print(f"time length: {n_time}, start time: {start_time}")
                # print(f"remaining resource: {remaining_resource}")
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
        :param verbose: whether print the procedure
        :param bid_start_time_arr: array of start times
        :param bid_price_arr: array of bids
        :param bid_usage_time_arr: array of maximum usage times
        :param auction_type: "second-price" or "first-price" reverse reverse_auction
        """

        # find the winner of the reverse_auction
        valuation_coefficient = self.df_tasks.loc[
            self.current_task_id, "valuation_coefficient"]
        utility_arr = np.multiply(bid_usage_time_arr,
            (valuation_coefficient - bid_price_arr))
        max_utility = np.amax(utility_arr)  # maximum utility for this task
        if max_utility <= 0:  # if bidding prices are all too high
            winner_index = None
            winner_usage_time = None
            winner_revenue = None
            sw_increase = 0
        else:
            # which FN wins this task
            winner_index = self.argmax_earliest(utility_arr=utility_arr,
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
            # print(f"cost of the winner is {winner_cost}")

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
                        i, 1] > 0:  # if the price is greater than winner's and the usage time is not zero
                        second_price = df_bids.iloc[i, 0]
                        break
                winner_revenue = (
                        winner_usage_time * second_price - winner_cost)
                if verbose:
                    print("Dataframe of the bids:")
                    print(df_bids)
                    print(f"second-price={second_price}")

            else:
                raise ValueError("unrecognised reverse_auction type")
            # print(f"ID of the winner = {winner_index}")
            # print(f"winner's revenue = {winner_revenue}")

            # update social welfare
            sw_increase = valuation_coefficient * winner_usage_time - winner_cost
        return winner_index, winner_usage_time, winner_revenue, max_utility, sw_increase

    def argmax_earliest(self, utility_arr, bid_start_time_arr):
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
