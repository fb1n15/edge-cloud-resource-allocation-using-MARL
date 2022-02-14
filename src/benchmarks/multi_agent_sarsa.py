import json

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
# from tqdm import tqdm
from tqdm import tqdm_notebook as tqdm  # for use in Jupyter Lab

from classes_in_reverse_auction_v1 import FogNodeAgent
from classes_in_reverse_auction_v1 import ReverseAuctionMDP
from classes_in_reverse_auction_v1 import pd
from generate_simulation_data import generate_synthetic_data_edge_cloud

# (dataframe) set the maximum number of rows and columns to display to unlimited
pd.set_option("display.max_rows", None, "display.max_columns", None)


def train_multi_agent_sarsa(avg_resource_capacity, avg_unit_cost, seed=0,
        total_number_of_steps=500, num_fog_nodes=6,
        resource_coefficient_original=3, alpha=0.001,
        beta=0.1,
        epsilon_tuple=(0.2, 0.1, 0.05),
        epsilon_steps_tuple=(500, 500, 100),
        plot_bool=False, num_actions=4, time_length=100,
        high_value_proportion=0.2, high_value_slackness=0,
        low_value_slackness=6,
        valuation_coefficient_ratio=10,
        resource_ratio=1.2, trained_agents=None,
        verbose=False, auction_type="second-price"):
    """run multi-agent sarsa

    Args:
        :param auction_type: the type of the reverse reverse_auction
        :param verbose: whether print the details of the execution
        :param number_of_runs: number of trials
        :param total_number_of_steps: steps of RL (allocate how many tasks)
        :param num_fog_nodes: number of fog nodes
        :param resource_coefficient_original: a coefficient for computing the resource coefficient
        :param alpha: step size of weights
        :param beta: step size of estimated average rewards
        :param epsilon_tuple: probability of exploration
        :param epsilon_steps_tuple: number of steps to run for each epsilon
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

    # record the allocation scheme
    allocation_scheme = pd.DataFrame(
        columns=['node_id', 'start_time', 'end_time'])
    # run the trials
    result_sarsa_list = []  # a list of the lists of average rewards for each step of sarsa
    social_welfare_list = []  # a list of total social welfare of each trial
    if verbose:
        print(f"seed={seed}")
    sw_list = []  # a list of social welfare after a new task arrives
    # initialise some parameters
    n_steps = total_number_of_steps + 1
    np.random.seed(seed)
    # generate the two types of tasks
    number_of_tasks = total_number_of_steps
    V = {}  # value function: state -> value
    pi = {}  # policy: (state+price) -> probability
    actions = list(
        range(num_actions))  # bid [reject, 1/3, 2/3, 1] the value of the task

    # generate a seqence of analytics tasks
    # compute the resource coefficient
    resource_coefficient = (
            resource_coefficient_original * number_of_tasks / time_length)
    # generate the synthetic data for simulations
    df_tasks, df_nodes, n_time, n_tasks, num_fog_nodes = \
        generate_synthetic_data_edge_cloud(avg_resource_capacity, avg_unit_cost,
            n_tasks=total_number_of_steps,
            n_time=time_length,
            seed=seed, n_nodes=num_fog_nodes,
            p_high_value_tasks=high_value_proportion,
            high_value_slackness_lower_limit=high_value_slackness,
            high_value_slackness_upper_limit=high_value_slackness + 2,
            low_value_slackness_lower_limit=low_value_slackness,
            low_value_slackness_upper_limit=low_value_slackness + 2,
            resource_demand_high=resource_ratio,
            vc_ratio=valuation_coefficient_ratio,
            k_resource=resource_coefficient)
    if verbose:
        print("resource coefficient: ", resource_coefficient)
        print(f"low value slackness = {low_value_slackness}")
        print(f"high value slackness = {high_value_slackness}")
    print('df_tasks:')
    print(df_tasks.head(10))
    print('df_nodes:')
    print(df_nodes.head())

    average_reward_sarsa_list = []
    agents_list = []
    # with tqdm(total=100) as pbar:
    mdp = ReverseAuctionMDP(df_tasks, df_nodes,
        num_nodes=num_fog_nodes, num_actions=num_actions)  # several fog nodes
    # generate several agents representing several fog nodes
    if trained_agents is not None:
        for i in range(mdp.num_fog_nodes):
            agent = FogNodeAgent(n_steps=n_steps - 1, alpha=alpha, beta=beta,
                fog_index=i,
                df_tasks=df_tasks,
                df_nodes=df_nodes, num_actions=num_actions,
                epsilon=epsilon_tuple[0], mdp=mdp,
                trained_agent=trained_agents[i])
            agents_list.append(agent)
    else:
        for i in range(mdp.num_fog_nodes):
            agent = FogNodeAgent(n_steps=n_steps - 1, alpha=alpha, beta=beta,
                fog_index=i,
                df_tasks=df_tasks,
                df_nodes=df_nodes, num_actions=num_actions,
                epsilon=epsilon_tuple[0], mdp=mdp)
            agents_list.append(agent)

    # actions taken by each node
    actions = {i: [] for i in range(mdp.num_fog_nodes)}

    # the reverse reverse_auction
    for k in tqdm(range(total_number_of_steps)):
        # fog nodes decide their bidding price, and allocation scheme for the current task
        if verbose:
            print()
            print(f"step: {k}")
        # epsilon decreases as the number of steps increases
        if k < epsilon_steps_tuple[0]:
            epsilon = epsilon_tuple[0]
        elif k < epsilon_steps_tuple[0] + epsilon_steps_tuple[1]:
            epsilon = epsilon_tuple[1]
        elif k < (epsilon_steps_tuple[0] + epsilon_steps_tuple[1] +
                  epsilon_steps_tuple[2]):
            epsilon = epsilon_tuple[2]
        else:
            epsilon = 0
        if verbose:
            print(f'epsilon = {epsilon}')

        # change the epsilon of all agents
        for i in range(mdp.num_fog_nodes):
            agents_list[i].epsilon = epsilon

        bids_list = []  # bidding price for one time step
        max_usage_time_list = []  # maximum usage time a fog node can offer
        start_time_list = []  # start time according to the planned allocation
        relative_start_time_list = []  # relative start time according to the current task
        for i in range(mdp.num_fog_nodes):
            (bidding_price, max_usage_time, relative_start_time, action) = \
                agents_list[i].differential_sarsa_decide_action(verbose=verbose)
            # tranfer relative start_time to absolute start_time
            start_time = int(
                df_tasks.loc[k, 'arrive_time'] + relative_start_time + 1)
            bids_list.append(bidding_price)
            max_usage_time_list.append(max_usage_time)
            start_time_list.append(start_time)
            relative_start_time_list.append(relative_start_time)
            actions[i].append(action)

        # find the winner
        (winner_index, winner_num_time, winner_utility, max_utility) = \
            mdp.step(bids_list, max_usage_time_list, start_time_list,
                verbose=verbose,
                auction_type=auction_type)
        sw_list.append(
            mdp.social_welfare)  # a list of social welfare after a new task arrives

        # modify the allocation scheme
        if winner_num_time is not None and winner_num_time > 0:
            allocation_scheme.loc[k] = [winner_index,
                start_time_list[winner_index],
                start_time_list[winner_index] + max_usage_time_list[
                    winner_index] - 1]
        else:  # the task is rejected
            allocation_scheme.loc[k] = [None, None, None]
        if verbose:
            print()
            print(f"nodes' bids = {bids_list}")
            print(f"nodes' usage times = {max_usage_time_list}")
            print(f"nodes' start times = {start_time_list}")
            print(f"winner's index = {winner_index}")
            print(f"number of usage time = {winner_num_time}")
            print(f"winner's utility = {winner_utility}")
            print(f"user's utility = {max_utility}")
            # print(f"social_welfare_list={sw_list}")

        if k < total_number_of_steps - 1:
            # update sarsa weights
            for i in range(mdp.num_fog_nodes):
                if verbose:
                    print(f"updating weights of node{i}:")
                if i == winner_index:  # if fog node i wins this task
                    agents_list[i].differential_sarsa_update_weights(1,
                        max_usage_time_list[i], relative_start_time_list[i],
                        winner_revenue=winner_utility,
                        verbose=verbose)
                else:  # if fog node i lose the reverse_auction
                    agents_list[i].differential_sarsa_update_weights(0,
                        max_usage_time_list[i],
                        relative_start_time_list[i],
                        winner_revenue=winner_utility,
                        verbose=verbose)
        else:
            if verbose:
                print("This is the last task.")  # no need to update weights

    if verbose:
        print(f"social welfare = {sw_list[-1]}")
    social_welfare_list.append(sw_list[-1])

    # generate a list of average rewards of recent 100 tasks
    average_reward_sarsa_list = []
    for i in range(total_number_of_steps):
        if i < 100:
            average_reward = sw_list[i] / (i + 1)
            average_reward_sarsa_list.append(average_reward)
        else:
            average_reward = (sw_list[i] - sw_list[i - 100]) / 100
            average_reward_sarsa_list.append(average_reward)

    result_sarsa_list.append(average_reward_sarsa_list.copy())
    # print(result_sarsa_list)

    # print the total value of tasks
    total_value = 0
    for i in range(total_number_of_steps):
        total_value += (df_tasks.loc[i, "valuation_coefficient"] *
                        df_tasks.loc[i, "usage_time"])
    if verbose:
        print(f"total value of tasks = {total_value}")
        print("df_tasks:")
        print(df_tasks.head())
        print("df_nodes:")
        print(df_nodes.head())

    if plot_bool:  # plot the result
        fig, axes = plt.subplots(1 + 2 * num_fog_nodes, 1, figsize=(12, 30))
        fig.suptitle('Figures')
        # plot the social welfare
        result_df = None
        for item in result_sarsa_list:
            result_sarsa_y = item.copy()
            x_list = range(len(result_sarsa_y))
            auction_df = pd.DataFrame({
                'algorithm': 'reverse reverse_auction',
                'steps': x_list,
                'average_social_welfare (recent 100 tasks)': result_sarsa_y
            })
            result_df = pd.concat([result_df, auction_df])
        # print(result_df)
        sns.lineplot(ax=axes[0], data=result_df, x="steps",
            y="average_social_welfare (recent 100 tasks)")
        # plt.show()

        # plot the learned average rewards of each node
        for i in range(num_fog_nodes):
            avg_reward = agents_list[i].list_avg_reward
            x_list = range(len(avg_reward))
            avg_reward_df = pd.DataFrame({
                'steps': x_list,
                'average rewards of node 1': avg_reward
            })
            sns.lineplot(ax=axes[i + 1], data=avg_reward_df, x="steps",
                y="average rewards of node 1")

        # plot the actions taken by each node
        for i in range(num_fog_nodes):
            actions_of_i = actions[i]
            x_list = range(len(actions_of_i))
            actions_of_i_df = pd.DataFrame({
                'steps': x_list,
                'action options': actions_of_i
            })
            sns.lineplot(ax=axes[i + num_fog_nodes + 1], data=actions_of_i_df,
                x='steps', y='action options')

        # TODO: plot the actions taken by each node

        plt.show()
    else:
        pass

    return sw_list, total_value, df_tasks, df_nodes, agents_list, allocation_scheme


# execute sarsa (do not update weights)
def execute_multi_agent_sarsa(avg_resource_capacity, avg_unit_cost,
        number_of_runs=50, total_number_of_steps=500,
        num_fog_nodes=6,
        resource_coefficient_original=3,
        plot_bool=False, num_actions=4, time_length=100,
        high_value_proportion=0.2, high_value_slackness=0,
        low_value_slackness=6,
        valuation_coefficient_ratio=10,
        resource_ratio=1.2, agents_list=None,
        bool_decay=True, training_seed=0, verbose=False,
        auction_type="second-price"):
    """execute multi-agent sarsa

    Args:
        agents_list: a list of trained agents
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
    # record the allocation scheme
    allocation_scheme = pd.DataFrame(
        columns=['node_id', 'start_time', 'end_time'])

    for j in tqdm(range(number_of_runs)):  # run all the trials
        # for j in tqdm(range(number_of_runs - 1, number_of_runs)):  # just run one trial

        if verbose:
            print(f"run ID = {j}")

        sw_list = []  # a list of social welfare after a new task arrives
        # initialise some parameters
        n_steps = total_number_of_steps + 1
        np.random.seed(j)
        # generate the two types of tasks
        number_of_tasks = total_number_of_steps
        V = {}  # value function: state -> value
        pi = {}  # policy: (state+price) -> probability
        actions = list(range(
            num_actions))  # bid [reject, 1/3, 2/3, 1] the value of the task

        # generate a seqence of analytics tasks
        # compute the resource coefficient
        resource_coefficient = (
                resource_coefficient_original * number_of_tasks / time_length)
        # generate the synthetic data for simulations
        df_tasks, df_nodes, n_time, n_tasks, num_fog_nodes = \
            generate_synthetic_data_edge_cloud(avg_resource_capacity,
                avg_unit_cost, n_tasks=total_number_of_steps,
                n_time=time_length,
                seed=j, n_nodes=num_fog_nodes,
                p_high_value_tasks=high_value_proportion,
                high_value_slackness_lower_limit=high_value_slackness,
                high_value_slackness_upper_limit=high_value_slackness + 2,
                low_value_slackness_lower_limit=low_value_slackness,
                low_value_slackness_upper_limit=low_value_slackness + 2,
                resource_demand_high=resource_ratio,
                vc_ratio=valuation_coefficient_ratio,
                k_resource=resource_coefficient)

        if verbose:
            print("resource coefficient: ", resource_coefficient)
            print(f"low value slackness = {low_value_slackness}")
            print(f"high value slackness = {high_value_slackness}")
            print("df_tasks:")
            print(df_tasks.head())
            print("df_nodes:")
            print(df_nodes.head())

        # print the total value of tasks
        total_value = 0
        for i in range(total_number_of_steps):
            total_value += (df_tasks.loc[i, "valuation_coefficient"] *
                            df_tasks.loc[i, "usage_time"])
        if verbose:
            print(f"total_number_of_steps={total_number_of_steps}")
            print(f"total value of tasks = {total_value}")

        # with tqdm(total=100) as pbar:
        mdp = ReverseAuctionMDP(df_tasks, df_nodes,
            num_nodes=num_fog_nodes,
            num_actions=num_actions)  # several fog nodes
        # reset the states of the fog node agents
        for i in range(mdp.num_fog_nodes):
            agents_list[i].reset_state(df_tasks)

        # actions taken by each node
        actions = {i: [] for i in range(mdp.num_fog_nodes)}

        # the reverse reverse_auction
        for k in tqdm(range(total_number_of_steps)):
            # fog nodes decide their bidding price, and allocation scheme for the current task
            if verbose:
                print()
                print(f"step: {k}")

            bids_list = []  # bidding price for one time step
            max_usage_time_list = []  # maximum usage time a fog node can offer
            start_time_list = []  # start time according to the planned allocation
            relative_start_time_list = []  # relative start time according to the current task
            for i in range(mdp.num_fog_nodes):
                (bidding_price, max_usage_time, relative_start_time, action) = \
                    agents_list[i].differential_sarsa_decide_action(
                        verbose=verbose)
                # tranfer relative start_time to absolute start_time
                start_time = int(
                    df_tasks.loc[k, 'arrive_time'] + relative_start_time + 1)
                bids_list.append(bidding_price)
                max_usage_time_list.append(max_usage_time)
                start_time_list.append(start_time)
                relative_start_time_list.append(relative_start_time)
                actions[i].append(action)

            # find the winner
            (winner_index, winner_num_time, winner_utility, max_utility) = \
                mdp.step(bids_list, max_usage_time_list, start_time_list,
                    verbose=verbose,
                    auction_type=auction_type)

            if verbose:
                print()
                print(f"nodes' bids = {bids_list}")
                print(f"nodes' usage times = {max_usage_time_list}")
                print(f"nodes' start times = {start_time_list}")
                print(f"winner's index = {winner_index}")
                print(f"number of usage time = {winner_num_time}")
                print(f"winner's utility = {winner_utility}")
                print(f"user's utility = {max_utility}")
            # a list of social welfare after a new task arrives
            sw_list.append(mdp.social_welfare)

            # modify the overall allocation scheme
            if winner_num_time is not None and winner_num_time > 0:
                allocation_scheme.loc[k] = [winner_index,
                    start_time_list[winner_index],
                    start_time_list[winner_index] + winner_num_time - 1]
            else:  # the task is rejected
                allocation_scheme.loc[k] = [None, None, None]

            # Do not update weights during execution
            if k < total_number_of_steps - 1:
                # update sarsa weights
                for i in range(mdp.num_fog_nodes):
                    if i == winner_index:  # if fog node i wins this task
                        agents_list[i].differential_sarsa_update_weights(1,
                            max_usage_time_list[i],
                            relative_start_time_list[i],
                            winner_revenue=winner_utility,
                            bool_update_weights=False, verbose=verbose)
                    else:  # if fog node i lose the reverse_auction
                        agents_list[i].differential_sarsa_update_weights(0,
                            max_usage_time_list[i],
                            relative_start_time_list[i],
                            winner_revenue=winner_utility,
                            bool_update_weights=False, verbose=verbose)
            else:
                if verbose:
                    print("This is the last task.")  # no need to update weights
        # enable_print()
        if verbose:
            print(f"social welfare = {sw_list[-1]}")
        social_welfare_list.append(sw_list[-1])

        # generate a list of average rewards
        average_reward_sarsa_list = []
        for i in range(total_number_of_steps):
            average_reward = sw_list[i] / (i + 1)
            average_reward_sarsa_list.append(average_reward)

        result_sarsa_list.append(average_reward_sarsa_list.copy())
        # print(result_sarsa_list)

    if plot_bool:  # plot the result
        fig, axes = plt.subplots(1 + num_fog_nodes, 1, figsize=(12, 30))
        fig.suptitle('Figures')

        # plot the social welfare
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
        sns.lineplot(ax=axes[0], data=result_df, x="steps",
            y="average_social_welfare")

        # plot the actions taken by each node
        for i in range(num_fog_nodes):
            actions_of_i = actions[i]
            x_list = range(len(actions_of_i))
            actions_of_i_df = pd.DataFrame({
                'steps': x_list,
                'action options': actions_of_i
            })
            sns.lineplot(ax=axes[i + 1], data=actions_of_i_df,
                x='steps', y='action options')

        plt.show()

    else:  # save the result for jupyter notebook
        if bool_decay:
            with open(
                    f'../simulation_results/auction_v1_{j + 1}trials'
                    f'_rc={resource_coefficient_original}_seed={training_seed}_decay.txt',
                    'w') as f:
                f.write(json.dumps(social_welfare_list))
        else:
            with open(
                    f'../simulation_results/auction_v1_{j + 1}trials'
                    f'_rc={resource_coefficient_original}_seed={training_seed}.txt',
                    'w') as f:
                f.write(json.dumps(social_welfare_list))

    return sw_list, total_value, df_tasks, df_nodes, agents_list, allocation_scheme


if __name__ == "__main__":
    # code for running simulations
    number_of_steps = 10000
    time_length = number_of_steps / 4
    num_trials = 50
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
        epsilon_steps_tuple=epsilon_steps_tuple,
        num_actions=num_actions,
        total_number_of_steps=number_of_steps,
        num_fog_nodes=6,
        resource_coefficient_original=resource_coefficient_original,
        number_of_runs=1, plot_bool=True)
