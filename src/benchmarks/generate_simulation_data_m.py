import pickle
# The pickle module implements binary protocols for serializing and de-serializing a Python object structure.
from typing import List, Any
# from Graph import Graph
import numpy as np
import csv
import random
import pandas as pd
import math


# random (normal) positive numbers
def PosNormal(mean, sigma, size, random_generator):

    x = []
    while size > 0:
        y = random_generator.normal(mean, sigma, 1)
        if y > 0:
            x.append(y[0])
            size -= 1
    return x


def generate_synthetic_data_edge_cloud(nr_tasks=10, time_length=10, n_nodes=6, p_high_value_task=0.2, seed=7,
                                       high_value_slackness_lower_limit=0, high_value_slackness_upper_limit=0,
                                       low_value_slackness_lower_limit=0, low_value_slackness_upper_limit=6,
                                       resource_demand_mean_high_value=1, resource_demand_mean_low_value=0.5, resource_demand_sigma=1.0, vc_ratio=10,
                                       k_resource=1, global_resources=True, ub_usage_time=4, node_upper_resource_limit=1.0, node_lower_resource_limit=0.5,
                                       node_cpu_lb_cost=0.5, node_cpu_ub_cost=1.0, node_ram_lb_cost=0.5, node_ram_ub_cost=1.0, node_disk_lb_cost=0.5, node_disk_ub_cost=1.0,
                                       wait_time_range=(1,2)):
    """Generate a synthetic list of tasks based on the normal distribution.

    Args:
        :param nr_tasks (int): Number of tasks. Default is 10.
        :param n_time (int): Number of timestamps. Default is 10.
        :param n_nodes (int): Number of fog nodes. Default is 6.
        :param p_high_value_tasks (double): Percentage of high values tasks. Default is 0.2
        :param seed (int): The seed for the random number generator. Default is 7.
        :param high_value_slackness_lower_limit (int): Lower value for deadlines for high value tasks. Default is 0.
        :param high_value_slackness_upper_limit (int): Upper value for deadlines for high value tasks. Default is 0.
        :param low_value_slackness_lower_limit (int): Lower value for deadlines for low value tasks. Default is 0.
        :param low_value_slackness_upper_limit (int): Upper value for deadlines for low value tasks. Default is 6.
        :param resource_demand_mean_high_value (int): The mean for demand for high value tasks. Default is 1.
        :param resource_demand_mean_low_value (int): The the mean for the demand for low value tasks. Default is 0.5.
        :param vc_ratio (int): High value resource range. (50 * param - 100 * param). Default is 10.
        :param resource_demand_sigma (double): Distance from the mean for demand for tasks. Default is 1.0.
        :param k_resource (int): Number of resources per fog node. Default is 1.
        :param global_resources (boolean): Set to True for all nodes having the same amount of resources. Default is True
        :param node_cpu_lb_cost (int): The value for the cpu cost lower bound for a node. Default is 0.5.
        :param node_cpu_ub_cost (int): The value for the cpu cost upper bound for a node. Default is 1.0.
        :param node_ram_lb_cost (int): The value for the ram cost lower bound for a node. Default is 0.5.
        :param node_ram_ub_cost (int): The value for the ram cost upper bound for a node. Default is 1.0.
        :param node_disk_lb_cost (int): The value for the disk cost lower bound for a node. Default is 0.5.
        :param node_disk_ub_cost (int): The value for the disk cost upper bound for a node. Default is 1.0.
        :param ub_usage_time (int): Maximum number of timestamps for the usage value of a task. Default is 4.
        :param node_upper_resource_limit (double): The upper limit for the amount of resources per node. Default is 1.0.
        :param node_lower_resource_limit (double): The lower limit for the amount of resources per node. Default is 0.5.
        :param wait_time_range (int, int): The rage for the wait time for tasks after they arrive until they can start. Default is (1,2) i.e tasks start on the next timestamp.

    Returns:
        [dataframe, dataframe, int, int, int]: [Dataframe of tasks, Dataframe of fog nodes, number of timestamps, number of tasks, number of fog nodes]
    """

    if wait_time_range[0] < 1 or wait_time_range[1] > time_length -1:
        print("Invalid values for waiting time range. Use values in the range 1 <-> n_timesteps - 1.\nUsing (1,2).")
        wait_time_range = (1,2)


    nr_timestamps = time_length  # 12 discrete time steps
    random_generator = np.random.RandomState(seed)

    # arrival time follows a uniform distribution
    arrive_time = random_generator.uniform(0, time_length - 3, nr_tasks)
    arrive_time.sort()

    end_time = time_length - 1

    # user's valuation and time constraint
    # The valuation b_prime is uniformly randomly chosen within the interval decided by the constant F
    v_prime = []
    start_time = []
    usage_time = []
    finish_time = []
    demand_CPU = []
    demand_RAM = []
    demand_disk = []

    # generate the indices of high value tasks
    high_value_tasks = random_generator.choice(
        range(nr_tasks), int(nr_tasks * p_high_value_task), replace=False)

    # generate valuation coefficients and time constraints
    for i in range(0, nr_tasks):
        if i in high_value_tasks:  # high_value tasks
            # valuation coefficient ratio
            v_prime.append(random_generator.uniform(
                50 * vc_ratio, 100 * vc_ratio))

            # Generate a random start time for tasks based on the waiting time range parameters
            start_time.append(int(arrive_time[i]) + random_generator.randint(wait_time_range[0], wait_time_range[1]))
            usage_time.append(random_generator.randint(1, ub_usage_time+1))

            # j: earliest finish time
            j = start_time[i] + usage_time[i] - 1

            # generate finish_time
            k = j + random_generator.randint(high_value_slackness_lower_limit,
                                             high_value_slackness_upper_limit+1)

            # finish time cannot be greater than the end time
            if k > end_time:
                x = end_time
            else:
                x = k

            finish_time.append(x)
            # resource constraints
            demand_CPU.append(
                PosNormal(resource_demand_mean_high_value, resource_demand_sigma, 1, random_generator)[0])
            demand_RAM.append(
                PosNormal(resource_demand_mean_high_value, resource_demand_sigma, 1, random_generator)[0])
            demand_disk.append(
                PosNormal(resource_demand_mean_high_value, resource_demand_sigma, 1, random_generator)[0])
        else:
            v_prime.append(random_generator.uniform(50 * 1, 100 * 1))
            # task can start from the next time step of its arrival time
            start_time.append(int(arrive_time[i]) + 1)
            usage_time.append(random_generator.randint(1, 3))
            # j: earliest finish time
            j = start_time[i] + usage_time[i] - 1
            # generate finish_time
            k = j + random_generator.randint(low_value_slackness_lower_limit,
                                             low_value_slackness_upper_limit+1)
            # finish time cannot be greater than the end time
            if k > end_time:
                x = end_time
            else:
                x = k
            finish_time.append(x)
            demand_CPU.append(
                PosNormal(resource_demand_mean_low_value, resource_demand_sigma, 1, random_generator)[0])
            demand_RAM.append(
                PosNormal(resource_demand_mean_low_value, resource_demand_sigma, 1, random_generator)[0])
            demand_disk.append(
                PosNormal(resource_demand_mean_low_value, resource_demand_sigma, 1, random_generator)[0])

    # dataframe of the tasks
    df_tasks = pd.DataFrame({'valuation_coefficient': v_prime, 'arrive_time': arrive_time,
                             'start_time': start_time, 'deadline': finish_time,
                             'usage_time': usage_time, 'CPU': demand_CPU,
                             'RAM': demand_RAM, 'DISK': demand_disk
                             })

    # GENERATE RESOURCE CAPACITIES OF FOG NODES
    if global_resources:

        CPU_capacity = np.full(n_nodes, k_resource)
        RAM_capacity = np.full(n_nodes, k_resource)
        DISK_capacity = np.full(n_nodes, k_resource)

    else:

        CPU_capacity = random_generator.uniform(
            node_lower_resource_limit, node_upper_resource_limit, n_nodes)
        RAM_capacity = random_generator.uniform(
            node_lower_resource_limit, node_upper_resource_limit, n_nodes)
        DISK_capacity = random_generator.uniform(
            node_lower_resource_limit, node_upper_resource_limit, n_nodes)

    CPU_cost = random_generator.uniform(
        node_cpu_lb_cost, node_cpu_ub_cost, n_nodes)
    RAM_cost = random_generator.uniform(
        node_ram_lb_cost, node_ram_ub_cost, n_nodes)
    DISK_cost = random_generator.uniform(
        node_disk_lb_cost, node_disk_ub_cost, n_nodes)

    df_nodes = pd.DataFrame({'CPU': CPU_capacity, 'RAM': RAM_capacity,
                             'DISK': DISK_capacity, 'CPU_cost': CPU_cost,
                             'RAM_cost': RAM_cost, 'DISK_cost': DISK_cost,
                             })

    # print('New simulation data generated!')
    return df_tasks, df_nodes, nr_timestamps, nr_tasks, n_nodes
