"""
Just run optimal (different prices at different nodes) pricing in simple situations
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

stdout = sys.stdout

from numpy.core.multiarray import ndarray
from online_myopic_mechanism import online_myopic
import optimal_pricing_hill_climbing as ophc
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
n_tasks = 100
n_time = 10
seed = 3
n_nodes = 6

# generate synthetic data for the simulation
df_tasks, df_nodes, n_time, n_tasks, n_nodes = \
    generate_synthetic_data_edge_cloud(n_tasks=n_tasks,
                                       n_time=n_time,
                                       n_nodes=n_nodes,
                                       k_resource=3,
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
print(f"price_upper_value={price_upper_value}")

# run Optimal Pricing algo.
(social_welfare, number_of_allocated_tasks, optimal_solution,
 allocation_scheme) = (
    ophc.optimal_pricing(df_tasks, df_nodes, n_time, n_tasks, n_nodes,
                         n_iterations=500,
                         price_lower_value=price_lower_value,
                         price_upper_value=price_upper_value,
                         step_size_para=0.1))
print("number of allocated tasks:", number_of_allocated_tasks)
# print the total value of tasks
total_value = 0
for i in range(n_tasks):
    total_value += (df_tasks.loc[i, "valuation_coefficient"] *
                    df_tasks.loc[i, "usage_time"])
print(f"Total value of tasks = {total_value}")
print(f"Social welfare = {social_welfare}")

# some random comments
