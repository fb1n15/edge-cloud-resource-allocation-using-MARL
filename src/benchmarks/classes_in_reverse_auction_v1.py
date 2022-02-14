"""
Reverse reverse_auction where each agent bids a unit price and sends the maximum usage
time it can offer and the earliest start time.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque

from numpy.core._multiarray_umath import ndarray
from tqdm import tqdm

from agents import BaseAgent
from generate_simulation_data import *


def argmax(values):
    """
    Takes in a list of values and returns the index of the item 
    with the highest value. Breaks ties randomly.
    (Because np.argmax always return the first index of highest value)

    Args:
        values: a list of values

    Returns:
        int - the index of the highest value in values
    """
    top_value = float("-inf")
    ties = []

    for i in range(len(values)):
        # if a value in values is greater than 
        # ~the highest value update top and reset ties to zero
        if values[i] > top_value:
            top_value = values[i]
            ties = []
        # if a value is equal to top value add the index to ties
        if values[i] == top_value:
            ties.append(i)
    # return a random selection from ties.
    return np.random.choice(ties)


def argmax_earliest(arr_utility, arr_start_time):
    """
    Takes in a list of values and returns the index of the item
    with the highest value (with the smallest start time).  Breaks ties randomly.
    (Because np.argmax always return the first index of highest value)

    Args:
        arr_utility: an array of utilities
        arr_start_time: an array of start times

    Returns:
        int - the index of the highest value in values
    """
    top_value = float("-inf")
    earliest_time = float('inf')
    ties = []
    ties2 = []

    # find the nodes with highest utility to the task
    for i in range(len(arr_utility)):
        # if a value in values is greater than
        # ~the highest value update top and reset ties to zero
        if arr_utility[i] > top_value:
            top_value = arr_utility[i]
            ties = []
        # if a value is equal to top value add the index to ties
        if arr_utility[i] == top_value:
            ties.append(i)

    # find in these nodes the nodes with the earliest start time
    for i in ties:
        if arr_start_time[i] < earliest_time:
            earliest_time = arr_start_time[i]
            ties2 = []
        if arr_start_time[i] == earliest_time:
            ties2.append(i)

    # return a random selection from ties2.
    return np.random.choice(ties2)


class ReverseAuctionMDP:
    """
    States -- biddings (prices and num of time steps) from FNs, valuation vector of the task
    Actions -- winner FN, num of time steps to the winner, payment to the winner
    Rewards -- the payment to each FN
    Social Welfare -- the socialwelfare so far
    """

    def __init__(self, df_tasks, df_nodes, num_nodes=6, num_actions=4,
            longest_usage_time=4):
        """Reverse reverse_auction.

        This version each node bids for its maximum number of time steps for the task

        Args:
            df_tasks: tasks
            df_nodes: fog nodes
            num_nodes: number of fog nodes
        """
        self.arr_bidding_price = np.zeros(
            num_nodes)  # array of bids (price per time step)
        self.arr_bidding_num_time = np.zeros(num_nodes, dtype=int)
        self.arr_valuation_vector = np.zeros(
            longest_usage_time)  # the valuation vector of the task
        self.df_tasks = df_tasks  # the dataframe of the tasks
        self.df_nodes = df_nodes  # the dataframe of the fog nodes
        self.task_id = 0  # index of the current task
        self.social_welfare = 0  # record the social welfare
        self.num_fog_nodes = num_nodes  # number of FNs
        self.num_actions = num_actions  # the number of actions

    # return the list of possible actions
    def get_possible_actions(self):
        # allocate 0, 1, 2, 3, 4 time steps to this task
        return [i for i in range(self.num_actions)]

    # initialise the valuation vector of the first task
    def first_step(self):
        for i in range(self.df_tasks.loc[0, "usage_time"]):
            self.arr_valuation_vector[i] = self.df_tasks.loc[
                                               0, 'valuation_coefficient'] * (
                                                   i + 1)
        self.task_id = 0

    def step(self, arr_bidding_price, arr_bidding_num_time, arr_start_time,
            verbose=False,
            auction_type="second-price"):
        """Decides the reverse_auction result of this task

        Args:
        :param verbose: whether print the procedure
        :param arr_start_time: array of start times
        :param arr_bidding_price: array of bids
        :param arr_bidding_num_time: array of maximum usage times
        :param auction_type: "second-price" or "first-price" reverse reverse_auction
        """

        # initial step
        self.arr_bidding_price = arr_bidding_price
        self.arr_bidding_num_time = arr_bidding_num_time

        # find the winner of the reverse_auction
        valuation_coefficient = self.df_tasks.loc[
            self.task_id, "valuation_coefficient"]
        utility_arr = np.multiply(self.arr_bidding_num_time,
            (
                    valuation_coefficient - self.arr_bidding_price))
        max_utility = np.amax(utility_arr)  # maximum utility for this task
        if max_utility < 0:  # if bidding prices are all too high
            winner_index = None
            winner_num_time = None
            winner_revenue = None
        else:
            # which FN wins this task
            winner_index = argmax_earliest(utility_arr, arr_start_time)
            winner_num_time = self.arr_bidding_num_time[
                winner_index]  # no. of time steps for this task
            winner_cost = \
                (self.df_nodes.loc[winner_index, 'CPU_cost'] *
                 self.df_tasks.loc[
                     self.task_id, 'CPU']
                 + self.df_nodes.loc[winner_index, 'RAM_cost'] *
                 self.df_tasks.loc[self.task_id, "RAM"]
                 + self.df_nodes.loc[winner_index, 'storage_cost'] *
                 self.df_tasks.loc[self.task_id, "storage"]) * winner_num_time
            # print(f"cost of the winner is {winner_cost}")

            # get winner's rewards
            if auction_type == "first-price":
                winner_revenue = (winner_num_time * arr_bidding_price[
                    winner_index] - winner_cost)
            elif auction_type == "second-price":
                bids = {'prices': arr_bidding_price,
                    'times': arr_bidding_num_time}
                df_bids = pd.DataFrame(bids, columns=['prices', 'times'])
                df_bids = df_bids.sort_values('prices')
                second_price = arr_bidding_price[winner_index]
                for i in range(self.num_fog_nodes):
                    if df_bids.iloc[i, 0] > second_price and df_bids.iloc[
                        i, 1] > 0:  # if the price is greater than winner's and the usage time is not zero
                        second_price = df_bids.iloc[i, 0]
                        break
                winner_revenue = (winner_num_time * second_price - winner_cost)
                if verbose:
                    print("Dataframe of the bids:")
                    print(df_bids)
                    print(f"second-price={second_price}")

            else:
                raise ValueError("unrecognised reverse_auction type")
            # print(f"ID of the winner = {winner_index}")
            # print(f"winner's revenue = {winner_revenue}")

            # update social welfare
            self.social_welfare += valuation_coefficient * winner_num_time - winner_cost

        # next task
        self.task_id += 1
        # # update the valuation vector for the next task
        # for i in range(self.df_tasks.loc[self.task_id, "usage_time"]):
        #     self.arr_valuation_vector[i] = self.df_tasks.loc[self.task_id,
        #                                                      'valuation_coefficient'] * (i + 1)

        return winner_index, winner_num_time, winner_revenue, max_utility


# class for Fog Node agents
class FogNodeAgent(BaseAgent):
    def __init__(self, n_steps, beta, fog_index, df_tasks, df_nodes,
            num_actions, trained_agent=None, **kwargs):
        """

        Args:
            n_steps: number of steps to run
            beta: the step size for updates to the avg rewards
            fog_index: the index of the fog node
            df_tasks: the dataframe of tasks
            df_tasks_normalised: the dataframe of taske (normalised)
            df_nodes: the dataframe of fog nodes:
            num_actions: how many actions the agent can choose
            run_episode_fn: the sarsa algorithm
            **kwargs: parameters such as ɑ and ε
        """
        np.random.seed(fog_index)  # seed the generator
        self.task_index = 0
        self.index = fog_index
        self.df_tasks = df_tasks
        self.df_nodes = df_nodes
        self.num_updates = 0
        self.avg_reward = 0
        self.current_task = self.df_tasks.loc[self.task_index, :]
        # normalize the state (df_tasks)
        df_tasks_normalised = df_tasks.copy()
        # make the deadlines relative
        df_tasks_normalised['deadline'] = (df_tasks_normalised['deadline'] -
                                           df_tasks_normalised[
                                               'start_time'] + 1)
        df_tasks_normalised = df_tasks_normalised / (df_tasks_normalised.max())
        # print("normalised df_tasks")
        # print(df_tasks_normalised)
        self.df_tasks_normalised = df_tasks_normalised
        self.next_time_slot = 1  # the next time slot of current time
        self.start_time_current_task = 0  # decided start time of the current task
        self.passed_time = 0  # difference of start times between current and previous task
        self.num_actions = num_actions
        self.n_steps = n_steps
        self.beta = beta
        self.state_length = 1 + 4 + 1 + 3 + 3 * 10
        # state = [(0): constant (1~4): value vector (5): deadline (6~8): resource demand (9~...): resource occupency in future 10 time steps]
        self.state = np.zeros(self.state_length)
        self.arr_valuation_vector = np.zeros(
            4)  # the maximum length of valuation vectors is four
        # the occupency of each resource in the future 10 time slots
        self.occup_future = np.zeros(
            30)  # the resource occupency (ratio) of future 10 time steps
        self.list_avg_reward = []
        self.list_next_action = []
        self.total_reward = 0  # total rewards got by this fog node
        self.reward_list = []
        # the resource capacity in the future 10 time slots
        a = [df_nodes.loc[self.index, 'CPU'], df_nodes.loc[self.index, 'RAM'],
            df_nodes.loc[self.index, 'storage']]
        self.resource_future = np.array(a * 10)
        if trained_agent is None:
            self.w = np.random.uniform(0.0, 0.01, (
                self.num_actions, self.state_length))  # weights
        else:
            self.w = trained_agent.w
        super().__init__(**kwargs)
        # initialise the state
        for i in range(int(self.df_tasks.loc[self.task_index, "usage_time"])):
            self.arr_valuation_vector[i] = self.df_tasks_normalised.loc[
                                               self.task_index, 'valuation_coefficient'] * (
                                                   i + 1)
        # relative deadline in respect to the start time of the task
        relative_deadline = np.array(
            [self.df_tasks_normalised.loc[self.task_index, "deadline"]])
        resource_demand = np.array(
            [self.df_tasks_normalised.loc[self.task_index, "CPU"],
                self.df_tasks_normalised.loc[self.task_index, "RAM"],
                self.df_tasks_normalised.loc[self.task_index, "storage"]])

        # occupency ratio of resources of future 10 time slots
        self.state = np.concatenate(
            ([1], self.arr_valuation_vector, relative_deadline,
            resource_demand, self.occup_future))
        self.action = self.get_action(self.state)  # last action of this agent

    # reset the state of agent for excution after training
    def reset_state(self, df_tasks):
        # reset df_tasks, df_tasks_normalised for execution
        self.df_tasks = df_tasks
        # normalize the state (df_tasks)
        df_tasks_normalised = df_tasks.copy()
        # make the deadlines relative
        df_tasks_normalised['deadline'] = (df_tasks_normalised['deadline'] -
                                           df_tasks_normalised[
                                               'start_time'] + 1)
        df_tasks_normalised = df_tasks_normalised / (df_tasks_normalised.max())
        self.df_tasks_normalised = df_tasks_normalised
        self.task_index = 0
        self.current_task = self.df_tasks.loc[self.task_index, :]
        self.state = np.zeros(self.state_length)
        self.arr_valuation_vector = np.zeros(
            4)  # the maximum length of valuation vectors is four
        # the occupency of each resource in the future 10 time slots
        self.occup_future = np.zeros(
            30)  # the resource occupency (ratio) of future 10 time steps
        # initialise the state
        for i in range(int(self.df_tasks.loc[self.task_index, "usage_time"])):
            self.arr_valuation_vector[i] = self.df_tasks_normalised.loc[
                                               self.task_index,
                                               'valuation_coefficient'] * (
                                                   i + 1)
        # relative deadline in respect to the start time of the task
        relative_deadline = np.array(
            [self.df_tasks_normalised.loc[self.task_index, "deadline"]])
        resource_demand = np.array(
            [self.df_tasks_normalised.loc[self.task_index, "CPU"],
                self.df_tasks_normalised.loc[self.task_index, "RAM"],
                self.df_tasks_normalised.loc[self.task_index, "storage"]])

        # occupency ratio of resources of future 10 time slots
        self.state = np.concatenate(
            ([1], self.arr_valuation_vector, relative_deadline,
            resource_demand, self.occup_future))
        self.action = self.get_action(self.state)  # last action of this agent

    def get_operational_cost(self, verbose=False):
        """return the operational cost of the current task

        Returns:
            operational_cost: the operational cost of the current task
        """

        operational_cost = (self.df_tasks.loc[self.task_index, 'CPU'] *
                            self.df_nodes.loc[self.index, 'CPU_cost'] +
                            self.df_tasks.loc[self.task_index, 'RAM'] *
                            self.df_nodes.loc[self.index, 'RAM_cost'] +
                            self.df_tasks.loc[self.task_index, 'storage'] *
                            self.df_nodes.loc[self.index, 'storage_cost'])

        return operational_cost

    def differential_sarsa_decide_action(self, verbose=False):
        """ execute the differential semi-gradient Sarsa algorithm

        Returns:
           bidding_price: bidding price per time step
           max_time_length: offered usage time for the task
           start_time: planned start time for the task
        """

        # initialize state S and action A
        state = self.state
        action = self.action

        # loop for each step:
        if verbose:
            print(f"number of weights updates = {self.num_updates}")
            print(f"task index = {self.task_index}")
            print(f"fog node {self.index} chooses action {action}")
            # print(f"fog node {self.index}'s state is {self.state}")
        # take action A, observe R, S'
        if action == 0:  # action = 0 means that rejecting this task
            max_time_length = 0
            start_time = self.start_time_current_task
        else:
            max_time_length, start_time = self.find_max_time_length()
        if verbose:
            print(f"can offers {max_time_length} time steps at most")
            print(f"start time is {start_time}")

        if action != 0:
            # based on the value of the task
            bidding_price = (self.df_tasks.loc[
                                 self.task_index, 'valuation_coefficient'] * action / (
                                 self.num_actions))

            # # based on the operational cost of the task
            # operational_cost = self.get_operational_cost()
            # vc = self.df_tasks.loc[self.task_index, 'valuation_coefficient']
            # bidding_price = ((action - 1) / 8 * (
            #         vc - operational_cost) + operational_cost)

            # # make it a little random
            # bidding_price = random.uniform(bidding_price * 0.75,
            #     bidding_price * 1.25)
        else:
            bidding_price = 1e8
        self.action = action

        return bidding_price, max_time_length, start_time, action

    def differential_sarsa_update_weights(self, winner_bool, max_time_length,
            start_time, winner_revenue, bool_update_weights=True,
            verbose=False):
        if not winner_bool:
            reward = 0
        else:
            reward = winner_revenue
        if verbose:
            print(f"rewards is {reward}")
        self.reward_list.append(reward)

        # differential sarsa update
        self.update_resource_occupency(winner_bool, max_time_length, start_time)
        next_state = self.update_current_task()

        # choose A' as a function of q(S', ., w)
        next_action = self.get_action(next_state)
        self.list_next_action.append(next_action)
        self.list_avg_reward.append(self.avg_reward)
        if bool_update_weights:
            self.update(self.state, self.action, reward, next_state,
                next_action, verbose=verbose)
        else:
            if verbose:
                print("Do not update weights")

        self.total_reward += reward  # update the total rewards of this fog node
        if verbose:
            # print("updating weights")
            # print(f"task index: {self.task_index}")
            # print(f"action: {action}")
            # print(f"state: {self.state}")
            # print(f"next action: {next_action}")
            # print(f"next state: {next_state}")
            # print(f"task ID: {state[-1]}")
            print(f"max_time_length: {max_time_length}")
            print(f"rewards : {reward}")
            print(f"estimated average rewards: {self.avg_reward}")
        # if winner_bool:
        #     print(f"reward_list = {self.reward_list}")
        self.state = next_state
        self.action = next_action

        # print(f"actions: {agent.list_next_action}")
        # print(f"average rewards: {agent.list_avg_reward}")
        # np.savetxt("actions_and_avg_reward.csv", np.column_stack((agent.list_next_action,
        #                 agent.list_avg_reward)), delimiter=",", fmt='%s')

    def get_q_value(self, state, action):
        """get q value by linear approximation

        Args:
            state: current state
            action: action choosen

        Returns:
            q_value: the q value of this action (Q stands for quality.
            It represents how useful an action is in gaining future rewards)

        """
        q_value: ndarray = np.dot(state, self.w[action])
        return q_value

    def update(self, state, action, reward, next_state, next_action,
            bool_update_weight=True, verbose=False):
        """Updates weights and the estimated average rewards.

        Args:
            state: current state
            action: action taken
            reward: utility get by this agent
            next_state: next state
            next_action: next action
            bool_update_weight: wether update weights
            verbose: whether print the details
        """
        delta = (reward - self.avg_reward + self.get_q_value(next_state,
            next_action) - self.get_q_value(state, action))
        if verbose:
            print(f"delta = {delta}")
            print(
                f"estimated average rewards (before updating) = {self.avg_reward}")
        # update average rewards
        self.avg_reward += self.beta * delta
        if verbose:
            print(f"beta = {self.beta}")
            print(
                f"estimated average rewards (after updating) = {self.avg_reward}")

        # update weights

        # if verbose:
        # print(f"action# = {action}")
        # print(f"weights (before updating) = {self.w}")
        self.w[action] += self.alpha * delta * state
        # if verbose:
        #     print(f"alpha = {self.alpha}")
        #     print(f"weights (after updating) = {self.w}")
        # print(f"weights: ")
        # if rewards > 0:
        # print(self.w)

        # update the number of stpes (updates)
        self.num_updates += 1
        if verbose:
            print(f"num of updates = {self.num_updates}")

    # def get_policy(self):
    #     arr_q = np.dot(self.w, self.state)
    #     return np.argmax(arr_q)
    #
    # def get_value_fu(self):
    #     arr_q = np.dot(self.w, self.state)
    #     return np.max(arr_q)

    def find_max_time_length(self):
        """find the largest number of time steps this FN can offer

        Returns:
            max_time_length: maximum time steps this task can run on this agent
            start_time: the start time of the task according to its allocation scheme
        """
        # remaining resource in future 10 time slots
        remaining_resource = self.resource_future - self.occup_future * self.resource_future
        self.current_task = self.df_tasks.loc[self.task_index, :]
        max_time_length = 0

        # print(f"current task deadline: {self.current_task_id['deadline']}")
        # print(f"current task start time: {self.current_task_id['start_time']}")

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
                    if remaining_resource[(start_time + i) * 3] < \
                            self.current_task['CPU']:
                        can_allocate = False
                    if remaining_resource[(start_time + i) * 3 + 1] < \
                            self.current_task['RAM']:
                        can_allocate = False
                    if remaining_resource[(start_time + i) * 3 + 2] < \
                            self.current_task['storage']:
                        can_allocate = False
                if can_allocate:
                    max_time_length = time_length
                    self.start_time_current_task = start_time
                    break
            else:
                continue
            break

        return max_time_length, start_time

    def update_current_task(self):
        """Update the state.

        Update the state considering the allocation of current task and the type of the next task.

        Returns:
            next_state: the updated state (a numpy array)
        """
        # update the current start time to the next task
        self.task_index += 1  # task ID of the next task
        # can be processed from the next time slot after arrival
        new_start_time = int(
            self.df_tasks.loc[self.task_index, "arrive_time"] + 1)
        # how many time steps have passed since last task arrives
        self.passed_time = new_start_time - self.next_time_slot
        self.next_time_slot = new_start_time
        self.current_task = self.df_tasks.loc[self.task_index, :]
        # update the state
        # update the valuation vector
        self.arr_valuation_vector = np.zeros(4)
        for i in range(int(self.df_tasks.loc[self.task_index, "usage_time"])):
            self.arr_valuation_vector[i] = (self.df_tasks_normalised.
                                            loc[
                                                self.task_index, 'valuation_coefficient'] * (
                                                    i + 1))
        relative_deadline = np.array(
            [self.df_tasks_normalised.loc[self.task_index, "deadline"]])
        resource_demand = np.array(
            [self.df_tasks_normalised.loc[self.task_index, "CPU"],
                self.df_tasks_normalised.loc[self.task_index, "RAM"],
                self.df_tasks_normalised.loc[self.task_index, "storage"]])

        # update resource occupency of resources of future 10 time slots
        next_state = np.concatenate(
            ([1], self.arr_valuation_vector, relative_deadline,
            resource_demand, self.occup_future))
        return next_state

    def update_resource_occupency(self, bool_winner, num_time_step, start_time):
        """update the occupency of different resources in the future 10 time slots

        Args:
            bool_winner: whether this agent wins the reverse_auction
            num_time_step: how many number of time steps are allocated to this tasks
            start_time: start time of the allocation scheme

        """
        # if wins, update resource occupency
        start_time = start_time
        if bool_winner:  # whether it wins the reverse_auction
            for i in range(num_time_step):
                self.occup_future[(start_time + i) * 3] += (
                        self.df_tasks.loc[self.task_index,
                        'CPU'] /
                        self.df_nodes.loc[self.index, 'CPU'])
                self.occup_future[(start_time + i) * 3 + 1] += (
                        self.df_tasks.loc[self.task_index,
                        'RAM'] /
                        self.df_nodes.loc[
                            self.index, 'RAM'])
                self.occup_future[(start_time + i) * 3 + 2] += (
                        self.df_tasks.loc[self.task_index,
                        'storage'] /
                        self.df_nodes.loc[
                            self.index, 'storage'])

        # convert occupency according to the current time
        # update resource for the next task
        passed_time = int(
            int(self.df_tasks.loc[self.task_index + 1, "arrive_time"]) -
            int(self.df_tasks.loc[self.task_index, "arrive_time"]))
        # if the passed time is >= 10, occupency of resource are all zeros
        new_occup_future = np.zeros(
            self.occup_future.shape)  # initialise with zeros
        # if the passed time is < 10, update the resource occupancy
        if passed_time < 10:
            # print(f"new_occup_future{new_occup_future}")
            # print(f"occup_future{self.occup_future}")
            # print(f"passed_time: {passed_time}")
            # update the occupency according to the time passed before next task arrives
            new_occup_future[0: (30 - 3 * passed_time)] = self.occup_future[
            3 * passed_time:]
        # update the occupency of resource in future 10 time slots
        self.occup_future = new_occup_future.copy()
        # print(f"The occupency of resource in future 10 time slots: {self.occup_future}")
