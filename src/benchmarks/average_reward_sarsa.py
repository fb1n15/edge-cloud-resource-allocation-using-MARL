# %%

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque
from tqdm import tqdm
import json

from agents import BaseAgent
from generate_simulation_data import generate_synthetic_data_edge_cloud
from classes_in_reverse_auction_v1 import ReverseAuctionMDP, FogNodeAgent, pd

pd.set_option('display.expand_frame_repr', False)  # print the whole dataframe

def run_sarsa(number_of_runs=50, total_number_of_steps=500, alpha=0.001, beta=0.1, epsilon=0.1,
              plot_bool=False):
    # # set some parameters
    # number_of_runs = 50
    # total_number_of_steps = 500
    # alpha = 0.003
    # beta = 0.06
    # epsilon = 0.1

    # run the trials
    result_sarsa_list = []
    for j in range(number_of_runs):
        try:
            print(f"run ID = {j}")
            # initialise some parameters
            n_steps = total_number_of_steps + 1
            np.random.seed(j)
            # generate the two types of tasks
            number_of_tasks = 2
            V = {}  # value function: state -> value
            pi = {}  # policy: (state+price) -> probability
            actions = [0, 1, 2, 3]  # [reject, 0.3, 0.6, 1.0] of the value of the task
            # generate a seqence of analytics tasks
            k_resource = 7
            df_tasks, df_nodes, n_time, n_tasks, n_nodes = \
                generate_synthetic_data_edge_cloud(n_tasks=number_of_tasks, n_time=10, k_resource=k_resource,
                                                   p_high_value_tasks=0.5, resource_demand_low=5, seed=1)
            # print("df_tasks:")
            # print(df_tasks)

            # normalize the state (df_tasks)
            df_tasks_normalised = df_tasks.copy()
            df_tasks_normalised['deadline'] = df_tasks_normalised['deadline'] - \
                                              df_tasks_normalised['start_time'] + 1
            df_tasks_normalised = df_tasks_normalised / (df_tasks_normalised.max())
            # print("normalised df_tasks")
            # print(df_tasks_normalised)

            # generate one sample of tasks
            list_sample = []
            list_sample_normalised = []

            sample_sequence = np.random.randint(0, 2, n_steps)
            list_tasks_normalised = df_tasks_normalised.values.tolist()
            df1_normalised = list_tasks_normalised[0]  # type one
            df2_normalised = list_tasks_normalised[1]  # type two

            list_tasks = df_tasks.values.tolist()
            df1 = list_tasks[0]
            df2 = list_tasks[1]

            # print(df1_normalised)
            # print(df2_normalised)

            for i in sample_sequence:
                if i == 0:
                    list_sample.append(df1)
                    list_sample_normalised.append(df1_normalised)
                else:
                    list_sample.append(df2)
                    list_sample_normalised.append(df2_normalised)

            # convert to a dataframe

            # print("A sample of tasks")
            # print(list_sample)
            df_sample = pd.DataFrame(list_sample, columns=df_tasks.columns)
            df_sample_normalised = pd.DataFrame(list_sample_normalised, columns=df_tasks_normalised.columns)
            # print("the data frame of a sample of tasks")
            # print(df_sample)
            # print("the data frame of a sample of tasks (normalised)")
            # print(df_sample_normalised)

            current_time = 0
            for i in range(len(df_sample.index)):
                df_sample.loc[i, "arrive_time"] += current_time
                df_sample.loc[i, "start_time"] += current_time + int()
                df_sample.loc[i, "deadline"] += current_time
                current_time = int(df_sample.loc[i, "arrive_time"] + 1)
            # print(df_sample.head())

            average_reward_sarsa_list = []
            # with tqdm(total=100) as pbar:
            mdp = ReverseAuctionMDP(df_tasks, df_nodes, k=1)  # only one fog node
            agent = FogNodeAgent(mdp=mdp, n_steps=n_steps - 1, alpha=alpha, beta=beta, fog_index=0,
                                 df_tasks=df_sample, df_tasks_normalised=df_sample_normalised,
                                 df_nodes=df_nodes, epsilon=epsilon)

            agent.run_episode()
            total_reward = agent.total_reward
            # print(f"total rewards = {total_reward}")

            # print(f"list of rewards: {agent.reward_list}")
            # max_vc = 564.06
            # average_reward = total_reward * max_vc / (n_steps-1)
            # average_reward_sarsa_list.append(average_reward)
            # pbar.update(1)
            max_vc = 564.06
            reward_list = [i * max_vc for i in agent.reward_list]
            average_reward_sarsa_list = []
            average_reward = 0
            for i in range(n_steps - 1):
                average_reward = reward_list[i] / (i + 1) + (i) / (i + 1) * average_reward
                average_reward_sarsa_list.append(average_reward)

            result_sarsa_list.append(average_reward_sarsa_list.copy())
        except Exception:  # just continue if some runs fail
            pass

    if plot_bool == True:
        result_sarsa_y = [item for sublist in result_sarsa_list for item in sublist]
        result_sarsa_y = pd.Series(result_sarsa_y)
        print(result_sarsa_y)
        # plot the result
        sns.lineplot(data=result_sarsa_y)
        plt.show()

    else:
        # save the result for jupyter notebook
        with open(f'test{alpha}{beta}{epsilon}.txt', 'w') as f:
            f.write(json.dumps(result_sarsa_list))


for alpha in [0.02]:
    for beta in [0.01]:
        for epsilon in [0.05, 0.1, 0.2]:
            run_sarsa(alpha=alpha,beta=beta,epsilon=epsilon,total_number_of_steps=12000,
                          number_of_runs=50, plot_bool=False)
            # run_sarsa(alpha=alpha,beta=beta,epsilon=epsilon,total_number_of_steps=1000,
            #           number_of_runs=5, plot_bool=)
