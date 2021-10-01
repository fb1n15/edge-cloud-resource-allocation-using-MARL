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


class ReverseAuctionMDP:
    """
    States -- biddings (prices and num of time steps) from FNs, valuation vector of the task
    Actions -- winner FN, num of time steps to the winner, payment to the winner
    Rewards -- the payment to each FN
    Social Welfare -- the socialwelfare so far
    """

    def __init__(self, df_tasks, df_nodes, num_nodes=6, longest_usage_time=4, num_actions=6):
        """Reverse reverse_auction.

        This version each node bids for its maximum number of time steps for the task

        Args:
            df_tasks: tasks
            df_nodes: fog nodes
            num_nodes: number of fog nodes
        """
        self.arr_bidding_price = np.zeros(num_nodes)  # array of bids (price per time step)
        self.arr_bidding_num_time = np.zeros(num_nodes, dtype=int)
        self.arr_valuation_vector = np.zeros(longest_usage_time)  # the valuation vector of the task
        self.df_tasks = df_tasks  # the dataframe of the tasks
        self.df_nodes = df_nodes  # the dataframe of the fog nodes
        self.num_tasks = 0  # index of the current task
        self.social_welfare = 0  # record the social welfare
        self.num_fog_nodes = num_nodes  # number of FNs
        self.num_actions = num_actions

    # return the list of possible actions
    def get_possible_actions(self):
        # allocate 0, 1, 2, 3, 4 time steps to this task
        return [i for i in range(self.num_actions)]

    # initialise the valuation vector of the first task
    def first_step(self):
        for i in range(self.df_tasks.loc[0, "usage_time"]):
            self.arr_valuation_vector[i] = self.df_tasks.loc[0, 'valuation_coefficient'] * (i + 1)
        self.num_tasks = 0

    # decides the reverse_auction result of this task
    def step(self, arr_bidding_price, arr_bidding_num_time):
        # initial step
        self.arr_bidding_price = arr_bidding_price
        self.arr_bidding_num_time = arr_bidding_num_time
        max_utility = {}  # max utility for one fog node
        winner_num_time = {}  # num usage time for one fog node
        overall_max_utility = 0  # max utility among all fog nodes
        winner_index = 0  # agent 0 is the winner by default
        overall_winner_num_time = 0  # default usage time is zero

        # find the winner of the reverse_auction
        valuation_coefficient = self.df_tasks.loc[self.num_tasks, "valuation_coefficient"]
        # find the bext utility for one fog node
        i = 0
        for price_list in arr_bidding_price:
            if not price_list:  # if max usage time == 0
                max_utility[i] = 0
                winner_num_time[i] = 0
            else:
                utility_arr = np.multiply(range(1, arr_bidding_num_time[i] + 1),
                                          (valuation_coefficient - self.arr_bidding_price[i]))
                max_utility[i] = np.amax(utility_arr)
                winner_num_time[i] = argmax(utility_arr) + 1
            if max_utility[i] > overall_max_utility:
                overall_max_utility = max_utility[i]
                winner_index = i
                overall_winner_num_time = winner_num_time[i]

            i += 1
        # # find the bext utility among all fog nodes
        # overall_max_utility = np.amax(utility_arr)  # maximum utility for this task
        # winner_index = argmax(utility_arr)  # which FN wins this task
        # winner_num_time = self.bid_usage_time_arr[winner_index]  # no. of time steps for this task
        winner_cost = \
            (self.df_nodes.loc[winner_index, 'CPU_cost'] * self.df_tasks.loc[self.num_tasks, 'CPU']
             + self.df_nodes.loc[winner_index, 'RAM_cost'] *
             self.df_tasks.loc[self.num_tasks, "RAM"]
             + self.df_nodes.loc[winner_index, 'storage_cost'] *
             self.df_tasks.loc[self.num_tasks, "storage"]) * overall_winner_num_time
        # print(f"cost of the winner is {winner_cost}")
        if overall_winner_num_time <= 0:  # no revenue if usage time == 0
            winner_revenue = 0
        else:
            winner_revenue = (self.arr_bidding_price[winner_index][overall_winner_num_time - 1]
                              * overall_winner_num_time - winner_cost)

        # update social welfare
        self.social_welfare += valuation_coefficient * overall_winner_num_time - winner_cost

        # next task
        self.num_tasks += 1
        # # update the valuation vector for the next task
        # for i in range(self.df_tasks.loc[self.task_id, "usage_time"]):
        #     self.arr_valuation_vector[i] = self.df_tasks.loc[self.task_id,
        #                                                      'valuation_coefficient'] * (i + 1)

        return winner_index, overall_winner_num_time, winner_revenue, overall_max_utility


# agents that give price for all possible number of time steps
class MenuFogNodeAgent(BaseAgent):
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
                                           df_tasks_normalised['start_time'] + 1)
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
        self.state_length = 3 + 3 + 3 * 10
        self.state = np.zeros(self.state_length)
        self.vc = 0  # the valuation coefficient
        # the occupency of each resource in the future 10 time slots
        self.occup_future = np.zeros(30)  # the resource occupency (ratio) of future 10 time steps
        self.list_avg_reward = []
        self.list_next_action = []
        self.total_reward = 0  # total rewards got by this fog node
        self.reward_list = []
        # the resource capacity in the future 10 time slots
        a = [df_nodes.loc[self.index, 'CPU'], df_nodes.loc[self.index, 'RAM'],
             df_nodes.loc[self.index, 'storage']]
        self.resource_future = np.array(a * 10)
        # weights (every combination of actions has a weight)
        if trained_agent is None:
            self.w = np.random.uniform(0.0, 0.01,
                                       (self.num_actions, self.num_actions, self.num_actions,
                                        self.num_actions, self.state_length))
        else:
            self.w = trained_agent.w
        super().__init__(**kwargs)
        # initialise the state
        self.vc = self.df_tasks_normalised.loc[self.task_index, 'valuation_coefficient']
        self.state = None
        self.action = None  # actions corresponding to different number of time steps

    # reset the state of the agent for execution after training
    def reset_state(self):
        self.task_index = 0
        self.vc = self.df_tasks_normalised.loc[self.task_index, 'valuation_coefficient']
        self.occup_future = np.zeros(30)

    def get_action(self, state, max_usage_time):
        """ e-greedy policy """
        rand = np.random.rand()
        possible_actions = self.mdp.get_possible_actions()
        action = [-1, -1, -1, -1]  # initialise the combination of actions
        if rand < self.epsilon:
            for t in range(max_usage_time):
                action[t] = possible_actions[np.random.choice(len(possible_actions))]
            print(f"pick a random action: action{action}")
            return action
        else:
            action = self.compute_best_action(state, max_usage_time)
            print(f'choose the "best action": action{action}')
            return action

    def compute_best_action(self, state, max_usage_time):
        # several actions may have the 'best' q_value; choose among them randomly
        legal_actions = self.mdp.get_possible_actions()
        if legal_actions[0] is None:
            return None
        best_action = [0, 0, 0, 0]
        best_q_value = float('-inf')
        for a in legal_actions:
            for b in legal_actions:
                for c in legal_actions:
                    for d in legal_actions:
                        if max_usage_time == 0:
                            action = [-1, -1, -1, -1]
                        elif max_usage_time == 1:
                            action = [a, -1, -1, -1]
                        elif max_usage_time == 2:
                            action = [a, b, -1, -1]
                        elif max_usage_time == 3:
                            action = [a, b, c, -1]
                        elif max_usage_time == 4:
                            action = [a, b, c, d]
                        else:
                            raise ValueError("max_usage_time > 4, impossible")
                        q_value = self.get_q_value(state, action)
                        # print(f"q value of action {action}: {q_value}")
                        if q_value > best_q_value:
                            best_action = action
                            best_q_value = q_value
        return best_action

    def differential_sarsa_decide_action(self):
        """ execute the differential semi-gradient Sarsa algorithm

        Returns:
           prices_list: unit price for each number of time steps
           max_time_length: offered usage time for the task
           start_time: planned start time for the task
        """

        # relative deadline in respect to the start time of the task
        relative_deadline = np.array([self.df_tasks_normalised.loc[self.task_index, "deadline"]])
        resource_demand = np.array([self.df_tasks_normalised.loc[self.task_index, "CPU"],
                                    self.df_tasks_normalised.loc[self.task_index, "RAM"],
                                    self.df_tasks_normalised.loc[self.task_index, "storage"]])

        # maximum number of time steps this agent can offer
        max_time_length, start_time = self.find_max_time_length()
        prices_list = []
        self.state = np.concatenate(([1], [self.vc], relative_deadline,
                                     resource_demand, self.occup_future))
        self.action = [-1, -1, -1, -1]  # initial with all num time steps impossible
        self.action = self.get_action(self.state, max_usage_time=max_time_length)
        for i in self.action:
            if i != -1:
                if i == 0:  # price is 1 million if action == 0
                    price = 1e6
                else:
                    # lowest price is 0 and highest unit price is the valuation_coefficient
                    price = (self.df_tasks.loc[self.task_index, 'valuation_coefficient']
                             * (i + 1) / self.num_actions)
                prices_list.append(price)
            else:
                print("reach the max usage time")

        # loop for each step:
        print(f"task {self.task_index}")
        print(f"fog node {self.index}'s menu of prices is {prices_list}")

        return prices_list, max_time_length, start_time

    def differential_sarsa_update_weights(self, winner_index, max_time_length, winner_num_time
                                          , start_time, bool_update_weights=True):
        # for i in range(1, max_time_length + 1):
        """update weights using differential semi-gradient sarsar

        only update the state corresponding to the winner's usage time
        """
        if self.index != winner_index:
            reward = 0
            winner_bool = 0
        else:
            winner_bool = 1
            # compute the operational cost
            cpu_cost = (self.df_tasks.loc[self.task_index, 'CPU'] *
                        self.df_nodes.loc[self.index, 'CPU_cost'] * winner_num_time)
            ram_cost = (self.df_tasks.loc[self.task_index, 'RAM'] *
                        self.df_nodes.loc[self.index, 'RAM_cost'] * winner_num_time)
            storage_cost = (self.df_tasks.loc[self.task_index, 'storage'] *
                            self.df_nodes.loc[self.index, 'storage_cost'] * winner_num_time)
            operational_cost = cpu_cost + ram_cost + storage_cost
            # compute the utility as the rewards
            print(f"winner_num_time = {winner_num_time}")
            reward = ((self.df_tasks.loc[self.task_index, 'valuation_coefficient'] *
                       self.action[winner_num_time - 1] / (
                               self.num_actions - 1)) * winner_num_time) - operational_cost
        self.reward_list.append(reward)

        # differential sarsa update
        self.update_resource_occupency(winner_bool, winner_num_time, start_time)
        next_state = self.update_current_task(winner_num_time)

        # find max time for next task
        max_time, start_time = self.find_max_time_length()
        # choose A' as a function of q(S', ., w)
        next_action = self.get_action(next_state, max_time)
        self.list_next_action.append(next_action)
        self.list_avg_reward.append(self.avg_reward)
        if bool_update_weights:
            self.update(self.state, self.action, reward,
                    next_state, next_action)
        else:
            print("Do not update weights")

        # print("updating weights")
        # print(f"task index: {self.task_index}")
        # print(f"action: {action}")
        print(f"state: {self.state}")
        # print(f"next action: {next_action}")
        print(f"next state: {next_state}")
        # print(f"task ID: {state[-1]}")
        print(f"max_time_length: {max_time_length}")
        print(f"rewards : {reward}")
        self.total_reward += reward  # update the total rewards of this fog node
        print(f"estimated average rewards: {self.avg_reward}")

        self.state = next_state
        self.action = next_action

        # print(f"actions: {agent.list_next_action}")
        # print(f"average rewards: {agent.list_avg_reward}")
        # np.savetxt("actions_and_avg_reward.csv", np.column_stack((agent.list_next_action,
        #                 agent.list_avg_reward)), delimiter=",", fmt='%s')

    def get_q_value(self, state, actions_list):
        """get q value by linear approximation

        Args:
            state: current state
            actions_list: a combination of actions

        Returns:
            q_value: the q value of this action (Q stands for quality.
            It represents how useful an action is in gaining future rewards)

        """
        q_value: ndarray = np.dot(state, self.w[actions_list[0]][actions_list[1]][actions_list[2]]
        [actions_list[3]])
        return q_value

    def update(self, state, action, reward, next_state, next_action):
        """Updates weights and the estimated average rewards.

        Args:
            state: current state
            action: the combination of actions taken
            reward: utility get by this agent
            next_state: next state
            next_action: next action

        """
        delta = (reward - self.avg_reward + self.get_q_value(next_state, next_action) -
                 self.get_q_value(state, action))
        print(f"delta = {delta}")
        # update average rewards
        # print(f"beta = {self.beta}")
        self.avg_reward += self.beta * delta

        # update weights
        # print(f"alpha = {self.alpha}")
        self.w[action[0]][action[1]][action[2]][action[3]] += self.alpha * delta * state
        # print(f"weights: ")
        # print(self.w)

        # update the number of stpes (updates)
        self.num_updates += 1

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
        for time_length in reversed(range(1, int(self.current_task["usage_time"] + 1))):
            for start_time in range(int(self.current_task['start_time'] -
                                        self.current_task['arrive_time']),
                                    int(self.current_task['deadline'] -
                                        self.current_task['arrive_time'] -
                                        time_length + 2)):

                # print(f"time length: {n_time}, start time: {start_time}")
                # print(f"remaining resource: {remaining_resource}")
                can_allocate = True  # can allocate in this case?
                for i in range(0, time_length):
                    if remaining_resource[(start_time + i) * 3] < self.current_task['CPU']:
                        can_allocate = False
                    if remaining_resource[(start_time + i) * 3 + 1] < self.current_task['RAM']:
                        can_allocate = False
                    if remaining_resource[(start_time + i) * 3 + 2] < self.current_task['storage']:
                        can_allocate = False
                if can_allocate:
                    max_time_length = time_length
                    self.start_time_current_task = start_time
                    break
            else:
                continue
            break

        return max_time_length, start_time

    def update_current_task(self, usage_time):
        """Update the state.

        Update the state considering the allocation of current task and the type of the next task.

        Args:
            usage_time: the usage time for this task

        Returns:
            next_state: the updated state (a numpy array)
        """
        # update the current start time to the next task
        self.task_index += 1  # task ID of the next task
        # can be processed from the next time slot after arrival
        new_start_time = int(self.df_tasks.loc[self.task_index, "arrive_time"] + 1)
        # how many time steps have passed since last task arrives
        self.passed_time = new_start_time - self.next_time_slot
        self.next_time_slot = new_start_time
        self.current_task = self.df_tasks.loc[self.task_index, :]
        # update the state
        # update the valuation vector
        self.vc = self.df_tasks_normalised.loc[self.task_index, 'valuation_coefficient']
        relative_deadline = np.array([self.df_tasks_normalised.loc[self.task_index, "deadline"]])
        resource_demand = np.array([self.df_tasks_normalised.loc[self.task_index, "CPU"],
                                    self.df_tasks_normalised.loc[self.task_index, "RAM"],
                                    self.df_tasks_normalised.loc[self.task_index, "storage"]])

        # update resource occupency of resources of future 10 time slots
        next_state = np.concatenate(([1], [self.vc], relative_deadline,
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
        if bool_winner:  # whether it wins the reverse_auction
            for i in range(num_time_step):
                self.occup_future[(start_time + i) * 3] += (self.current_task['CPU'] /
                                                            self.df_nodes.loc[self.index, 'CPU'])
                self.occup_future[(start_time + i) * 3 + 1] += (self.current_task['RAM'] /
                                                                self.df_nodes.loc[
                                                                    self.index, 'RAM'])
                self.occup_future[(start_time + i) * 3 + 2] += (self.current_task['storage'] /
                                                                self.df_nodes.loc[
                                                                    self.index, 'storage'])

        # convert occupency according to the current time
        # update resource for the next task
        passed_time = int(int(self.df_tasks.loc[self.task_index + 1, "arrive_time"]) -
                          int(self.df_tasks.loc[self.task_index, "arrive_time"]))
        # if the passed time is >= 10, occupency of resource are all zeros
        new_occup_future = np.zeros(self.occup_future.shape)  # initialise with zeros
        # if the passed time is < 10, update the resource occupancy
        if passed_time < 10:
            # print(f"new_occup_future{new_occup_future}")
            # print(f"occup_future{self.occup_future}")
            # print(f"passed_time: {passed_time}")
            # update the occupency according to the time passed before next task arrives
            new_occup_future[0: (30 - 3 * passed_time)] = self.occup_future[3 * passed_time:]
        # update the occupency of resource in future 10 time slots
        self.occup_future = new_occup_future.copy()
        # print(f"The occupency of resource in future 10 time slots: {self.occup_future}"
