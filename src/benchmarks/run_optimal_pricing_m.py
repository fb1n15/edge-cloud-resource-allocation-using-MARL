"""
Just run optimal pricing in simple situations
"""
# import packages
from tqdm import tqdm
import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from importlib import reload
import os
import sys

from other_functions import get_social_welfare

stdout = sys.stdout

from numpy.core.multiarray import ndarray
from online_myopic_mechanism import online_myopic
import optimal_pricing_m as opm
import generate_simulation_data_m as gsdm
from generate_simulation_data import generate_synthetic_data_edge_cloud
from other_functions import display_allocation


# Disable priting
def block_print():
    sys.stdout = open(os.devnull, 'w')


# Restore
def enable_print():
    sys.stdout = stdout


# set the parameters
mipgap = 0.1
n_tasks = 300
n_time = int(n_tasks / 10)
seed = 1
n_nodes = 6
resource_coefficient_original = 0.3
resource_coefficient = (resource_coefficient_original * n_tasks / n_time)
high_value_slackness = 0
low_value_slackness = 6
valuation_ratio = 3
high_value_proportion = 0.1
# generate synthetic data for the simulation
df_tasks, df_nodes, n_time, n_tasks, n_nodes = \
    generate_synthetic_data_edge_cloud(n_tasks=n_tasks,
        n_time=n_time,
        n_nodes=n_nodes,
        p_high_value_tasks=high_value_proportion,
        high_value_slackness_lower_limit=high_value_slackness,
        high_value_slackness_upper_limit=high_value_slackness + 2,
        low_value_slackness_lower_limit=low_value_slackness,
        low_value_slackness_upper_limit=low_value_slackness + 2,
        k_resource=resource_coefficient,
        vc_ratio=valuation_ratio,
        seed=seed)
df_tasks = df_tasks.rename(columns={"storage": "DISK"})
df_nodes = df_nodes.rename(
    columns={"storage": "DISK", "storage_cost": "DISK_cost"})

print("df_tasks:")
print(df_tasks.head())
print("df_nodes:")
print(df_nodes.head())

# run Optimal Pricing algo.
# set price range
column = df_tasks['valuation_coefficient']
price_upper_value = column.max()
price_lower_value = 0
# granularity = (price_upper_value - price_lower_value) / 5
granularity = (price_upper_value - price_lower_value) / 10

# run OP algo.
list_social_welfare, list_number_of_allocated_tasks, optimal_phi, allocation_scheme = \
    opm.optimal_pricing(df_tasks, df_nodes, n_time, n_tasks, n_nodes,
        granularity=granularity,
        price_upper_value=price_upper_value,
        price_lower_value=price_lower_value, global_phi=False)
print("social welfare:", list_social_welfare)
print("number of allocated tasks:", list_number_of_allocated_tasks)
print(f"optimal_phi = {optimal_phi}")

# print the total value of tasks
total_value = 0
for i in range(n_tasks):
    total_value += (df_tasks.loc[i, "valuation_coefficient"] *
                    df_tasks.loc[i, "usage_time"])
print(f"Total value of tasks = {total_value}")
# display_allocation(allocation_scheme, number_of_FNs, N, df_tasks, T)
# print(f"allocation_scheme = {allocation_scheme}")

# the social welfare of 100 tasks in the middle
social_welfare_partial = get_social_welfare(df_tasks, df_nodes,
    allocation_scheme, start_id=99,
    end_id=199)
print(f"social_welfare_partial={social_welfare_partial}")

# # another situation
# df_tasks['DISK'] = df_tasks['DISK'].apply(lambda x: 0.1)
# print("df_tasks:")
# print(df_tasks.head())
# print("df_nodes:")
# print(df_nodes.head())
#
# # set price range
# price_upper_value = 600
# price_lower_value = 0
# granularity = (price_upper_value-price_lower_value) / 10
#
# # run online myopic algo.
# social_welfare, number_of_allocated_tasks, optimal_phi, allocation_scheme = \
#     opm.optimal_pricing(df_tasks, df_nodes, T, N, number_of_FNs,
#                         granularity=granularity,
#                         price_upper_value=price_upper_value,
#                         price_lower_value=price_lower_value, global_phi=False)
# print("social welfare:", social_welfare)
# print("number of allocated tasks:", number_of_allocated_tasks)
# # print the total value of tasks
# total_value = 0
# for i in range(number_of_tasks):
#     total_value += (df_tasks.loc[i, "valuation_coefficient"] *
#                     df_tasks.loc[i, "usage_time"])
# print(f"total value of tasks = {total_value}")
# display_allocation(allocation_scheme, number_of_FNs, N, df_tasks, T)
# print(f"optimal_phi = {optimal_phi}")
