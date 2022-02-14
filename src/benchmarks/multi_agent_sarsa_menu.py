import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque
from tqdm import tqdm
import json

import sys
import os
from agents import BaseAgent
from generate_simulation_data import generate_synthetic_data_edge_cloud
from rl_agent import ReverseAuctionMDP, MenuFogNodeAgent, pd

# (dataframe) set the maximum number of rows and columns to display to unlimited
pd.set_option("display.max_rows", None, "display.max_columns", None)


# Disable
def block_print():
    sys.stdout = open(os.devnull, 'w')


# Restore
def enable_print():
    sys.stdout = sys.__stdout__


def train_multi_agent_sarsa(seed=0, total_number_of_steps=500, num_fog_nodes=6,
                            resource_coefficient_original=3, alpha=0.001, beta=0.1,
                            epsilon_tuple=(0.2, 0.1, 0.05), epsilon_steps_tuple=(500, 500),
                            plot_bool=False, num_actions=4, time_length=100,
                            high_value_proportion=0.2, high_value_slackness=0,
                            low_value_slackness=6, valuation_coefficient_ratio=10,
                            resource_ratio=1.2, trained_agents=None):
    """run multi-agent sarsa

    Args:
        :param number_of_runs: number of trials
        :param total_number_of_steps: steps of RL (allocate how many tasks)
        :param num_fog_nodes: number of fog nodes
        :param resource_coefficient_original: a coefficient for computing the resource coefficient
        :param alpha: step size of weights
        :param beta: step size of estimated average rewards
        :param epsilon_tuple: probability of exploration
        :param epsilon_steps_tuple: number of steps to run for each epsilon except for the last one
        :param plot_bool: whether plot the results
        :param num_actions: number of actions
        :param time_length: tasks arrive within this time length
        :param low_value_slackness: deadline slackness of low-value tasks
        :param resource_ratio: resource demand ratio between high-value and low-value tasks
        :param valuation_coefficient_ratio: valuation coefficient ratio between high-value
        and low-value tasks
        :param high_value_slackness: deadline slackness of high-value tasks
        :param high_value_proportion: the proportion of high-value tasks
    """

    # run the trials
    result_sarsa_list = []  # a list of the lists of average rewards for each step of sarsa
    social_welfare_list = []  # a list of total social welfare of each trial
    print(f"seed={seed}")
    block_print()
    sw_list = []  # a list of social welfare after a new task arrives
    # initialise some parameters
    n_steps = total_number_of_steps + 1
    np.random.seed(seed)
    # generate the two types of tasks
    number_of_tasks = total_number_of_steps
    V = {}  # value function: state -> value
    pi = {}  # policy: (state+price) -> probability
    actions = list(range(num_actions))  # bid [reject, 1/3, 2/3, 1] the value of the task

    # generate a seqence of analytics tasks
    resource_coefficient = (
            resource_coefficient_original * number_of_tasks / time_length)  # compute the resource coefficient
    # generate the synthetic data for simulations
    df_tasks, df_nodes, n_time, n_tasks, num_fog_nodes = \
        generate_synthetic_data_edge_cloud(n_tasks=total_number_of_steps, n_time=time_length,
                                           seed=seed, n_nodes=num_fog_nodes,
                                           p_high_value_tasks=high_value_proportion,
                                           high_value_slackness_lower_limit=high_value_slackness,
                                           high_value_slackness_upper_limit=high_value_slackness + 2,
                                           low_value_slackness_lower_limit=low_value_slackness,
                                           low_value_slackness_upper_limit=low_value_slackness + 2,
                                           resource_demand_high=resource_ratio,
                                           vc_ratio=valuation_coefficient_ratio,
                                           k_resource=resource_coefficient)
    print("resource coefficient: ", resource_coefficient)
    print(f"low value slackness = {low_value_slackness}")
    print(f"high value slackness = {high_value_slackness}")

    average_reward_sarsa_list = []
    agents_list = []
    # with tqdm(total=100) as pbar:
    # several fog nodes
    mdp = ReverseAuctionMDP(df_tasks, df_nodes, num_nodes=num_fog_nodes,
                            num_actions=num_actions)
    # generate several agents representing several fog nodes
    if trained_agents is not None:
        for i in range(mdp.num_fog_nodes):
            agent = MenuFogNodeAgent(n_steps=n_steps - 1, alpha=alpha, beta=beta, fog_index=i,
                                     df_tasks=df_tasks,
                                     df_nodes=df_nodes, num_actions=num_actions,
                                     epsilon=epsilon_tuple[0], mdp=mdp,
                                     trained_agent=trained_agents[i])
            agents_list.append(agent)
    else:
        for i in range(mdp.num_fog_nodes):
            agent = MenuFogNodeAgent(n_steps=n_steps - 1, alpha=alpha, beta=beta, fog_index=i,
                                     df_tasks=df_tasks,
                                     df_nodes=df_nodes, num_actions=num_actions,
                                     epsilon=epsilon_tuple[0], mdp=mdp)
            agents_list.append(agent)

    # the reverse reverse_auction
    for k in range(total_number_of_steps):
        # fog nodes decide their bidding price, and allocation scheme for the current task
        print()
        print(f"step: {k}")

        # epsilon decreases as the number of steps increases
        if k < epsilon_steps_tuple[0]:
            epsilon = epsilon_tuple[0]
        elif k < epsilon_steps_tuple[0] + epsilon_steps_tuple[1]:
            epsilon = epsilon_tuple[1]
        else:
            epsilon = epsilon_tuple[2]
        print(f'epsilon = {epsilon}')

        # change the epsilon of all agents
        for i in range(mdp.num_fog_nodes):
            agents_list[i].epsilon = epsilon

        bids_list = []  # bidding price for one time step
        max_usage_time_list = []  # maximum usage time a fog node can offer
        start_time_list = []  # start time according to the planned allocation
        for i in range(mdp.num_fog_nodes):
            (prices_list, max_usage_time, start_time) = \
                agents_list[i].differential_sarsa_decide_action()
            bids_list.append(prices_list)
            max_usage_time_list.append(max_usage_time)
            start_time_list.append(start_time)

        # find the winner
        (winner_index, winner_num_time, winner_utility, max_utility) = \
            mdp.step(bids_list, max_usage_time_list)

        print()
        print(f"winner's index = {winner_index}")
        print(f"number of usage time = {winner_num_time}")
        print(f"winner's utility = {winner_utility}")
        print(f"user's utility = {max_utility}")
        sw_list.append(mdp.social_welfare)  # a list of social welfare after a new task arrives

        if k < total_number_of_steps - 1:
            # update sarsa weights
            for i in range(mdp.num_fog_nodes):
                agents_list[i].differential_sarsa_update_weights(winner_index,
                                                                 max_usage_time_list[i],
                                                                 winner_num_time,
                                                                 start_time_list[i])
        else:
            print("This is the last task.")  # no need to update weights

    enable_print()
    print(f"social welfare = {sw_list[-1]}")
    social_welfare_list.append(sw_list[-1])

    # generate a list of average rewards
    average_reward_sarsa_list = []
    for i in range(total_number_of_steps):
        average_reward = sw_list[i] / (i + 1)
        average_reward_sarsa_list.append(average_reward)

    result_sarsa_list.append(average_reward_sarsa_list.copy())
    # print(result_sarsa_list)

    # print the total value of tasks
    total_value = 0
    for i in range(total_number_of_steps):
        total_value += (df_tasks.loc[i, "valuation_coefficient"] *
                        df_tasks.loc[i, "usage_time"])
    print(f"total value of tasks = {total_value}")
    print("df_tasks:")
    print(df_tasks.head())
    print("df_nodes:")
    print(df_nodes.head())

    if plot_bool:  # plot the result
        result_df = None
        for item in result_sarsa_list:
            result_sarsa_y = item.copy()
            x_list = range(len(result_sarsa_y))
            auction_df = pd.DataFrame({
                'algorithm': 'reverse reverse_auction',
                'steps': x_list,
                'average_social_welfare': result_sarsa_y
            })
            result_df = pd.concat([result_df, auction_df])
        # print(result_df)
        sns.lineplot(data=result_df, x="steps", y="average_social_welfare")
        plt.show()

        return sw_list, total_value, df_tasks, df_nodes, agents_list

    else:  # save the result for jupyter notebook
        return sw_list, total_value, df_tasks, df_nodes, agents_list


def execute_multi_agent_sarsa(number_of_runs=50, total_number_of_steps=500, num_fog_nodes=6,
                              resource_coefficient_original=3,
                              plot_bool=False, num_actions=4, time_length=100,
                              high_value_proportion=0.2, high_value_slackness=0,
                              low_value_slackness=6, valuation_coefficient_ratio=10,
                              resource_ratio=1.2, agents_list=None, bool_print=False,
                              bool_decay=True, training_seed=0):
    """run multi-agent sarsa

    Args:
        :param number_of_runs: number of trials
        :param total_number_of_steps: steps of RL (allocate how many tasks)
        :param num_fog_nodes: number of fog nodes
        :param resource_coefficient_original: a coefficient for computing the resource coefficient
        :param alpha: step size of weights
        :param beta: step size of estimated average rewards
        :param epsilon_tuple: probability of exploration
        :param epsilon_steps_tuple: number of steps to run for each epsilon except for the last one
        :param plot_bool: whether plot the results
        :param num_actions: number of actions
        :param time_length: tasks arrive within this time length
        :param low_value_slackness: deadline slackness of low-value tasks
        :param resource_ratio: resource demand ratio between high-value and low-value tasks
        :param valuation_coefficient_ratio: valuation coefficient ratio between high-value
        and low-value tasks
        :param high_value_slackness: deadline slackness of high-value tasks
        :param high_value_proportion: the proportion of high-value tasks
    """

    # run the trials
    result_sarsa_list = []  # a list of the lists of average rewards for each step of sarsa
    social_welfare_list = []  # a list of total social welfare of each trial
    for j in tqdm(range(number_of_runs)):  # run all the trials
        # for j in tqdm(range(number_of_runs - 1, number_of_runs)):  # just run one trial
        print(f"run ID = {j}")
        if bool_print:
            print("Allow printing")
        else:
            block_print()
        sw_list = []  # a list of social welfare after a new task arrives
        # initialise some parameters
        n_steps = total_number_of_steps + 1
        np.random.seed(j)
        # generate the two types of tasks
        number_of_tasks = total_number_of_steps
        V = {}  # value function: state -> value
        pi = {}  # policy: (state+price) -> probability
        actions = list(range(num_actions))  # bid [reject, 1/3, 2/3, 1] the value of the task

        # generate a seqence of analytics tasks
        resource_coefficient = (
                resource_coefficient_original * number_of_tasks / time_length)  # compute the resource coefficient
        # generate the synthetic data for simulations
        df_tasks, df_nodes, n_time, n_tasks, num_fog_nodes = \
            generate_synthetic_data_edge_cloud(n_tasks=total_number_of_steps, n_time=time_length,
                                               seed=j, n_nodes=num_fog_nodes,
                                               p_high_value_tasks=high_value_proportion,
                                               high_value_slackness_lower_limit=high_value_slackness,
                                               high_value_slackness_upper_limit=high_value_slackness + 2,
                                               low_value_slackness_lower_limit=low_value_slackness,
                                               low_value_slackness_upper_limit=low_value_slackness + 2,
                                               resource_demand_high=resource_ratio,
                                               vc_ratio=valuation_coefficient_ratio,
                                               k_resource=resource_coefficient)
        print("resource coefficient: ", resource_coefficient)
        print(f"low value slackness = {low_value_slackness}")
        print(f"high value slackness = {high_value_slackness}")

        mdp = ReverseAuctionMDP(df_tasks, df_nodes, num_nodes=num_fog_nodes,
                                num_actions=num_actions)
        print("number of fog nodes", mdp.num_fog_nodes)
        # change the epsilon of all agents to 0
        for i in range(mdp.num_fog_nodes):
            agents_list[i].reset_state()

        # the reverse reverse_auction
        for k in range(total_number_of_steps):
            # fog nodes decide their bidding price, and allocation scheme for the current task
            print()
            print(f"step: {k}")

            bids_list = []  # bidding price for one time step
            max_usage_time_list = []  # maximum usage time a fog node can offer
            start_time_list = []  # start time according to the planned allocation
            for i in range(mdp.num_fog_nodes):
                (prices_list, max_usage_time, start_time) = \
                    agents_list[i].differential_sarsa_decide_action()
                bids_list.append(prices_list)
                max_usage_time_list.append(max_usage_time)
                start_time_list.append(start_time)

            # find the winner
            (winner_index, winner_num_time, winner_utility, max_utility) = \
                mdp.step(bids_list, max_usage_time_list)

            print()
            print(f"winner's index = {winner_index}")
            print(f"number of usage time = {winner_num_time}")
            print(f"winner's utility = {winner_utility}")
            print(f"user's utility = {max_utility}")
            sw_list.append(mdp.social_welfare)  # a list of social welfare after a new task arrives

            if k < total_number_of_steps - 1:
                # update sarsa weights
                for i in range(mdp.num_fog_nodes):
                    agents_list[i].differential_sarsa_update_weights(winner_index,
                                                                     max_usage_time_list[i],
                                                                     winner_num_time,
                                                                     start_time_list[i],
                                                                     bool_update_weights=False)
            else:
                print("This is the last task.")  # no need to update weights

        enable_print()
        print(f"social welfare = {sw_list[-1]}")
        social_welfare_list.append(sw_list[-1])

        # generate a list of average rewards
        average_reward_sarsa_list = []
        for i in range(total_number_of_steps):
            average_reward = sw_list[i] / (i + 1)
            average_reward_sarsa_list.append(average_reward)

        result_sarsa_list.append(average_reward_sarsa_list.copy())
        # print(result_sarsa_list)

        # print the total value of tasks
        total_value = 0
        for i in range(total_number_of_steps):
            total_value += (df_tasks.loc[i, "valuation_coefficient"] *
                            df_tasks.loc[i, "usage_time"])
        print(f"total value of tasks = {total_value}")
        print("df_tasks:")
        print(df_tasks.head())
        print("df_nodes:")
        print(df_nodes.head())

    if plot_bool:  # plot the result
        result_df = None
        for item in result_sarsa_list:
            result_sarsa_y = item.copy()
            x_list = range(len(result_sarsa_y))
            auction_df = pd.DataFrame({
                'algorithm': 'reverse reverse_auction',
                'steps': x_list,
                'average_social_welfare': result_sarsa_y
            })
            result_df = pd.concat([result_df, auction_df])
        # print(result_df)
        sns.lineplot(data=result_df, x="steps", y="average_social_welfare")
        plt.show()

        return sw_list, total_value, df_tasks, df_nodes, agents_list


    else:  # save the result for jupyter notebook

        if bool_decay:

            with open(

                    f'../simulation_results/auction_v2_{j + 1}trials'

                    f'_rc={resource_coefficient_original}_seed={training_seed}_decay.txt',

                    'w') as f:

                f.write(json.dumps(social_welfare_list))

        else:

            with open(

                    f'../simulation_results/auction_v2_{j + 1}trials'

                    f'_rc={resource_coefficient_original}_seed={training_seed}.txt',

                    'w') as f:

                f.write(json.dumps(social_welfare_list))


if __name__ == "__main__":
    # code for running simulations
    number_of_steps = 10000
    time_length = int(number_of_steps / 4)
    num_trials = 30
    num_actions = 4
    epsilon_steps_tuple = (3000, 1000)

    # for resource_coefficient in [2]:
    #     for alpha in [0.02]:
    #         for beta in [0.01]:
    #             for epsilons_tuple in [(0.2, 0.1, 0.05)]:
    #                 # run multiple times and save the results
    #                 train_multi_agent_sarsa(alpha=alpha, beta=beta, epsilon_tuple=epsilons_tuple,
    #                                       num_actions=num_actions, n_time=n_time,
    #                                       epsilon_steps_tuple=epsilon_steps_tuple,
    #                                       total_number_of_steps=number_of_steps, num_fog_nodes=6,
    #                                       resource_coefficient_original=resource_coefficient,
    #                                       number_of_runs=num_trials, plot_bool=False)

    # run once and plot the average rewards
    epsilons_tuple = (0.2, 0.1, 0.05)
    epsilon_steps_tuple = (3000, 1000)
    resource_coefficient_original = 3
    train_multi_agent_sarsa(alpha=0.02, beta=0.01, epsilon_tuple=epsilons_tuple,
                            time_length=time_length,
                            epsilon_steps_tuple=epsilon_steps_tuple, num_actions=num_actions,
                            total_number_of_steps=number_of_steps, num_fog_nodes=6,
                            resource_coefficient_original=resource_coefficient_original,
                            number_of_runs=1, plot_bool=True)
