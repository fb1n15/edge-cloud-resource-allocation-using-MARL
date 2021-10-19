import pickle
# The pickle module implements binary protocols for serializing and de-serializing a Python object structure.
from typing import List, Any
# from Graph import Graph
import numpy as np
import csv
import random
import pandas as pd
import math

# test upload

# random (normal) positive numbers
def PosNormal(mean, sigma, size, random_generator):
    x = []
    while size > 0:
        y = random_generator.normal(mean, sigma, 1)
        if y > 0:
            x.append(y[0])
            size -= 1
    return x


def generate_synthetic_data_edge_cloud(avg_resource_capacity, avg_unit_cost,
                                       n_tasks=10, n_time=10, n_nodes=6,
                                       p_high_value_tasks=0.2, seed=7,
                                       high_value_slackness_lower_limit=0,
                                       high_value_slackness_upper_limit=0,
                                       low_value_slackness_lower_limit=0,
                                       low_value_slackness_upper_limit=6,
                                       resource_demand_high=1, resource_demand_low=1, vc_ratio=10,
                                       k_resource=1, usage_time_ub=3):
    """generate synthetic data for simulations

    Args:
        avg_resource_capacity: a dict of average resource capacity for each node
        avg_unit_cost: a dict of average unit operational cost for each node
        usage_time_ub: upper bound of usage times

    """
    T = n_time  # 12 discrete time steps
    random_generator = np.random.RandomState(seed)

    # arrival time follows a uniform distribution
    arrive_time = random_generator.uniform(0, n_time - 3, n_tasks)
    arrive_time.sort()

    end_time = n_time - 1

    # user's valuation and time constraint
    # The valuation b_prime is uniformly randomly chosen within the interval decided by the constant F
    v_prime = []
    start_times = []
    usage_times = []
    finish_times = []
    demand_CPU = []
    demand_RAM = []
    demand_disk = []
    # generate the indices of high value tasks
    high_value_tasks = random_generator.choice(range(n_tasks),
        int(n_tasks * p_high_value_tasks),
        replace=False)

    # generate valuation coefficients and time constraints
    for i in range(0, n_tasks):
        if i in high_value_tasks:  # high_value tasks
            # valuation coefficient ratio
            v_prime.append(
                random_generator.uniform(50 * vc_ratio, 100 * vc_ratio))
            # task can start from the next time step of its arrival time
            start_time = int(arrive_time[i]) + 1
            start_times.append(start_time)
            usage_time = random_generator.randint(usage_time_ub, usage_time_ub + 1)
            # j: earliest finish time
            j = start_time + usage_time - 1
            # generate finish_time
            k = j + random_generator.randint(high_value_slackness_lower_limit,
                high_value_slackness_upper_limit + 1)
            # finish time cannot be greater than the end time
            if k > end_time:
                finish_time = end_time
            else:
                finish_time = k
            finish_times.append(finish_time)
            #
            usage_time = min(usage_time, finish_time-start_time+1)
            usage_times.append(usage_time)
            # resource constraints
            demand_CPU.append(random_generator.uniform(3, 5))
            demand_RAM.append(random_generator.uniform(3, 5))
            demand_disk.append(random_generator.uniform(3, 5))
        else:
            v_prime.append(random_generator.uniform(50 * 1, 100 * 1))
            # task can start from the next time step of its arrival time
            start_time = int(arrive_time[i]) + 1
            start_times.append(start_time)
            usage_time = random_generator.randint(usage_time_ub, usage_time_ub + 1)
            # j: earliest finish time
            j = start_time + usage_time - 1
            # generate finish_time
            k = j + random_generator.randint(low_value_slackness_lower_limit,
                low_value_slackness_upper_limit + 1)
            # finish time cannot be greater than the end time
            if k > end_time:
                finish_time = end_time
            else:
                finish_time = k
            finish_times.append(finish_time)
            usage_time = min(usage_time, finish_time-start_time+1)
            usage_times.append(usage_time)
            demand_CPU.append(random_generator.uniform(2, 3))
            demand_RAM.append(random_generator.uniform(2, 3))
            demand_disk.append(random_generator.uniform(2, 3))

    # dataframe of the tasks
    df_tasks = pd.DataFrame(
        {'valuation_coefficient': v_prime, 'arrive_time': arrive_time,
            'start_time': start_times, 'deadline': finish_times,
            'usage_time': usage_times, 'CPU': demand_CPU,
            'RAM': demand_RAM, 'storage': demand_disk})

    # generate resource capacities of fog nodes
    # initialise with zeros
    CPU_capacity = np.zeros(n_nodes)
    RAM_capacity = np.zeros(n_nodes)
    storage_capacity = np.zeros(n_nodes)
    for node in avg_resource_capacity.keys():
        CPU_capacity[node] = avg_resource_capacity[node][0]
        RAM_capacity[node] = avg_resource_capacity[node][1]
        storage_capacity[node] = avg_resource_capacity[node][2]

    # generate unit operational costs of fog nodes
    # initialise with zeros
    CPU_unit_cost = np.zeros(n_nodes)
    RAM_unit_cost = np.zeros(n_nodes)
    storage_unit_cost = np.zeros(n_nodes)
    for node in avg_unit_cost.keys():
        CPU_unit_cost[node] = avg_unit_cost[node][0]
        RAM_unit_cost[node] = avg_unit_cost[node][1]
        storage_unit_cost[node] = avg_unit_cost[node][2]

    # aggregate to a dataframe
    df_nodes = pd.DataFrame({'CPU': CPU_capacity, 'RAM': RAM_capacity,
        'storage': storage_capacity, 'CPU_cost': CPU_unit_cost,
        'RAM_cost': RAM_unit_cost,
        'storage_cost': storage_unit_cost})

    # print('New simulation data generated!')
    return df_tasks, df_nodes, n_time, n_tasks, n_nodes
