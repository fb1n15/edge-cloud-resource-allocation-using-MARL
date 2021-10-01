import pickle

from multi_agent_sarsa import execute_multi_agent_sarsa

seed = 0
auction_type = "first-price"
# auction_type = "second-price"
number_of_tasks = 100
num_actions = 10
valuation_coefficient_ratio = 10
resource_ratio = 3
high_value_proportion = 0.1
time_length = int(number_of_tasks / 10)
num_trials = seed + 1
resource_coefficient_original = 0.3

filehandler = open(
    f"../trained_agents/reverse_auction_v1_seed={seed}_rc={resource_coefficient_original}_agents",
    'rb')
agents_list = pickle.load(filehandler)
sw_list, total_value, df_tasks_2, df_nodes, agents_list, allocation_scheme = \
    execute_multi_agent_sarsa(num_actions=num_actions,
        time_length=time_length,
        high_value_proportion=high_value_proportion,
        total_number_of_steps=number_of_tasks,
        num_fog_nodes=6,
        resource_coefficient_original=resource_coefficient_original,
        valuation_coefficient_ratio=valuation_coefficient_ratio,
        number_of_runs=num_trials, plot_bool=True,
        bool_decay=True,
        resource_ratio=resource_ratio,
        agents_list=agents_list, training_seed=seed,
        verbose=True, auction_type=auction_type)
print(f"total value of tasks = {total_value}")
social_welfare = sw_list[-1]
print(f"total social welfare = {social_welfare}")
