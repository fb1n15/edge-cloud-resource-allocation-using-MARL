"""
------------------------
OPTIMAL PRCING BENCHMARK:
-------------------------

It uses a pricing mechanism to generate a bid for the reverse_auction stage of the allocation mechanism. This price is the same for every unit of resource and for every node.

The algorithm can be run with a fixed price per unique unit resource (i.e. one price for CPU, one price for RAM and one prince for DISK) or a fixed price per unit resource (i.e. the same price for CPU, RAM and DISK).

The optimal price is found by doing a bruteforce search in the specified price range. The algorithms then assigns the tasks.

In order to run the benchmark algorithm, call the optimal_pricing() function with the required parameters.
"""

from functools import partial
from numpy.core.fromnumeric import argmax
from numpy.lib.function_base import average
import pandas as pd
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt

import math

pd.set_option('display.max_rows', 50000)
pd.set_option('display.max_columns', 200000)
pd.set_option('display.width', 400000)
pd.set_option('max_colwidth', 2000000)


def compute_bid(task, fog_node, phi_CPU=1.0, phi_RAM=1.0, phi_DISK=1.0):
    """Calculate the bid for the current node for the current task.

    Args:
        :param task (dataframe): A dataframe containing information about the current task.
        :param fog_node (series): A series containing information about the current node.
        :param phi_CPU (double): A price parameter used to determin the CPU price of each fog node.
        :param phi_RAM (double): A price parameter used to determine the RAM price of each fog node.
        :param phi_DISK (double): A price parameter used to determine the DISK price of each fog node.

    Returns:
        [int]: The bid of the current fog node.
    """

    CPU_valuation = phi_CPU * task['CPU']
    RAM_valuation = phi_RAM * task['RAM']
    DISK_valuation = phi_DISK * task['DISK']

    task_fog_perspective_valuation = (CPU_valuation + RAM_valuation + DISK_valuation)

    ###############################
    # GENERATE TASK VALUATION HERE#
    #
    # 1. Occupancy rate:
    #       Multiply phi with the occupancy ratio of the node for that task: phi * task_base_valuation_value / (fog_node.iloc['CPU'] + fog_node.iloc['RAM'] + fog_node.iloc['DISK'])
    #
    # 2.
    ###############################

    # task_fog_perspective_valuation = task_base_valuation_value / \
    #     (fog_node['CPU'] + fog_node['RAM'] + fog_node['DISK'])

    return task['valuation_coefficient'] - task_fog_perspective_valuation


def preauction_filters(participants, task, indivisible_tasks=False):
    """Apply some preauction filters to the participants list in order to remove any invalid candidates.

    Args:
        participants (list): A list of participating nodes and their info.
        indivisible_tasks (bool): Only allow nodes that offer full usage time.

    Returns:
        [list]: A list of participants.
    """

    if indivisible_tasks:
        participants = [
            x for x in participants if x['usage_time'] >= task.loc['usage_time']]

    return participants


# TODO: Implement any post reverse_auction filters


def postauction_filters(participants, task):
    return participants


def reverse_auction(task, participants, apply_preauction_filters=True,
                    apply_postauction_filters=True, verbose=True, phi_CPU=1.0, phi_DISK=1.0,
                    phi_RAM=1.0):
    """A function that computes the reverse reverse_auction for a set of fog nodes given a task.

    Args:
        task (series): A series containing the information about the current task.
        participants (dict): A dictionary with information about the participating nodes.
        apply_preauction_filters (bool): Apply any preauction filters.
        apply_postauction_filters (bool): Apply any postauction filters.

    Returns:
        [int]: The index of the winning node.
    """

    # Apply any specified filters
    if apply_preauction_filters:
        participants = preauction_filters(
            participants=participants,
            task=task,
            indivisible_tasks=False  # can execute part of the task
        )

    if apply_postauction_filters:
        participants = postauction_filters(
            participants=participants,
            task=task,
        )
    if verbose:
        print("\n\nBidding nodes are:\n\n", participants)

    best_bid = -math.inf

    # If there are no nodes left after filters, return no winner
    if not participants:
        return None, -1

    winner = participants[0]

    for node in participants:

        current_bid = compute_bid(
            task=task,
            fog_node=node['node_info'],
            phi_CPU=phi_CPU,
            phi_RAM=phi_RAM,
            phi_DISK=phi_DISK
        )

        if best_bid < current_bid:
            best_bid = current_bid
            winner = node

    return winner, best_bid


def format_allocation(nr_nodes, nr_timestamps, allocation_matrix, start_matrix):
    """Generate the formatted dict of the allocation of tasks.

    Args:
        nr_nodes (int): Number of fog nodes.
        nr_timestamps (int): Number of timestamps.
        allocation_matrix (matrix): Matrix showing the time allocated for each task.
        start_matrix (matrix): Matrix showing the start time of each task.

    Returns:
        [dict] : A formated dict of the allocation scheme.
    """

    # Prepare variables
    nr_tasks = len(allocation_matrix)
    output = {}
    for i in range(nr_nodes):
        output["node_" + str(i)] = {}

    # Define the fill string
    fill_string = ''.join(['.' for i in range(len(str(nr_tasks)))])

    # Add the occupancy rate for each node at each timestamp
    for i in range(len(allocation_matrix)):

        for j in range(len(allocation_matrix[0])):

            if allocation_matrix[i][j] != 0:
                output["node_" + str(j)][i] = [
                    fill_string[len(str(i)):] + str(i) if start_matrix[i][j] <= x < start_matrix[i]
                    [j] + allocation_matrix[i][j] else fill_string for x in range(nr_timestamps)]

    return output


def attempt_allocation(df_tasks, df_nodes, nr_timestamps, task_number, fn_nr, phi_CPU=1.0,
                       phi_RAM=1.0, phi_DISK=1.0, verbose=True):
    """The optimal pricing benchmark algorithm. I computes a fixed unit price for every resource and then every fn bids for the task on a reverse reverse_auction manner.

    Args:
        :param df_tasks (Dataframe): A dataframe containing the incoming tasks.
        :param df_nodes (Dataframe): A dataframe containing the information about the fog nodes.
        :param n_timesteps (int): Number of timestamps for the benchmark.
        :param task_number (int): Total number of tasks.
        :param fn_nr (int): Total number of fog nodes.
        :param phi (double): Optimal cost parameter.

    Returns:
        [int, int, dataframe, dataframe]: The socail welfare, the number of allocated tasks, the allocation matrix, the start time of each task. 
    """

    # Initialize metrics
    social_welfare = 0
    tasks_allocated = 0

    mat_time_temp = np.zeros([task_number, fn_nr], dtype=int)
    mat_start_time = np.zeros([task_number, fn_nr], dtype=int)
    mat_time_allocated = np.zeros([task_number, fn_nr], dtype=int)

    mat_CPU = []
    mat_RAM = []
    mat_DISK = []

    for i in range(fn_nr):
        mat_CPU.append(
            np.full(nr_timestamps, df_nodes.loc[i, "CPU"], dtype=float))
        mat_RAM.append(
            np.full(nr_timestamps, df_nodes.loc[i, "RAM"], dtype=float))
        mat_DISK.append(
            np.full(nr_timestamps, df_nodes.loc[i, "DISK"], dtype=float))

    mat_CPU = np.array(mat_CPU)
    mat_RAM = np.array(mat_RAM)
    mat_DISK = np.array(mat_DISK)

    # Process each task in the tasklist
    for task_index in range(task_number):

        current_task = df_tasks.iloc[task_index]

        if verbose:
            print("\n\nCurrent_task:\n\n", current_task, '\n')
        bidding_nodes = []

        # Compute which nodes can enter the reverse_auction i.e. they have enough resources and enough free timeslots
        for node_index in range(fn_nr):

            # Check if there are enough timestamps for this node
            for usage_time in reversed(range(1, df_tasks.loc[task_index, 'usage_time'] + 1)):
                is_allocated = False

                for time_slot in range(df_tasks.loc[task_index, 'start_time'],
                                       df_tasks.loc[task_index, 'deadline'] + 1):
                    not_enough_resources = False

                    # If task usage time does not fit inside its start-deadline window then break
                    if time_slot + usage_time > df_tasks.loc[task_index, 'deadline'] + 1:
                        break

                    # Check if there is enough resources for current time slot
                    for usage_time_slot in range(time_slot, time_slot + usage_time):

                        # Compute the resources requirements for the current timestamp
                        if df_tasks.loc[task_index, 'CPU'] > mat_CPU[node_index, usage_time_slot] or \
                                df_tasks.loc[task_index, 'RAM'] > mat_RAM[
                            node_index, usage_time_slot] or \
                                df_tasks.loc[task_index, 'DISK'] > mat_DISK[
                            node_index, usage_time_slot]:
                            not_enough_resources = True

                        # If there is not enough resources try the next time slot for that node
                        if not_enough_resources:
                            continue

                        # TODO: Alternative to this ^: try to find the best node for the current time slot so there are no gaps in the usage time.

                        if verbose:
                            print("Adding node", node_index,
                                  "to the bidding list...")

                        # Add the current node to the bidding nodes and continue to next node
                        bidding_nodes.append({
                            'node_info': df_nodes.iloc[node_index],
                            'node_index': node_index,
                            'timestamp': usage_time_slot,
                            'usage_time': usage_time,
                            'start_time': time_slot,
                            'task_index': task_index
                        })

                        # Only select the first available timeslot for each node
                        break

                    is_allocated = True
                    break

                if is_allocated:
                    break

        # If there are no nodes biding advance to next task
        if not bidding_nodes:
            continue

        # Start reverse_auction between eligible nodes
        winner, bid = reverse_auction(
            task=current_task,
            participants=bidding_nodes,
            phi_CPU=phi_CPU,
            phi_RAM=phi_RAM,
            phi_DISK=phi_DISK,
            verbose=False
        )

        if bid >= 0 and winner is not None:

            if verbose:
                print("\n\n############\nWinner is node \n",
                      winner, '\nwith a bid of', bid, "\n#############\n\n")

            # Update the allocation total social welfare, the allocation matrix and the number of allocated tasks
            mat_time_temp[winner['task_index'],
                          winner['node_index']] = winner['usage_time']
            mat_start_time[winner['task_index'],
                           winner['node_index']] = winner['start_time']

            mat_time_allocated[winner['task_index'], winner['node_index']
            ] = mat_time_temp[winner['task_index'], winner['node_index']]
            if verbose:
                print("\nAllocation matrix:\n", mat_time_allocated)

            # update the resource capacity of FNs
            for t in range(mat_start_time[winner['task_index'], winner['node_index']],
                           mat_start_time[winner['task_index'], winner['node_index']]
                           + mat_time_allocated[winner['task_index'], winner['node_index']]):
                mat_CPU[winner['node_index'],
                        t] -= df_tasks.loc[winner['task_index'], 'CPU']
                mat_RAM[winner['node_index'],
                        t] -= df_tasks.loc[winner['task_index'], 'RAM']
                mat_DISK[winner['node_index'],
                         t] -= df_tasks.loc[winner['task_index'], 'DISK']

            if verbose:
                print("Remaining CPU capacity:\n", mat_CPU)
            if verbose:
                print("Remaining RAM capacity:\n", mat_RAM)
            if verbose:
                print("Remaining storage capacity:\n", mat_DISK)

            # update the social welfare
            # social_welfare = social_welfare + current_task_id.loc['valuation_coefficient'] - winner['usage_time'] * \
            #     (winner['node_info'].loc['CPU_cost'] * current_task_id.loc['CPU'] +
            #      winner['node_info'].loc['RAM_cost'] * current_task_id.loc['RAM'] +
            #      winner['node_info'].loc['DISK_cost'] * current_task_id.loc['DISK'])

            social_welfare = social_welfare + winner['usage_time'] * (
                    current_task.loc['valuation_coefficient'] -
                    (winner['node_info'].loc['CPU_cost'] * current_task.loc['CPU'] +
                     winner['node_info'].loc['RAM_cost'] * current_task.loc['RAM'] +
                     winner['node_info'].loc['DISK_cost'] * current_task.loc['DISK']))

            tasks_allocated += 1

            if verbose:
                print('\nSocial welfare: ', social_welfare, '\n')
        else:
            if verbose:
                print("No winner this timestamp!")

    if verbose:
        print("Allocation Scheme:\n", mat_time_allocated, '\n\n\n')

    if verbose:
        print("TOTAL SOCIAL WELFARE: ", social_welfare, '\n')

    return social_welfare, tasks_allocated, mat_time_allocated, mat_start_time


# TODO: Optimise run time using binary search
def optimal_pricing(df_tasks, df_nodes, nr_timestamps, task_number, fn_nr, granularity=0.1,
                    price_lower_value=0, price_upper_value=10, global_phi=False):
    """The driver function for the optimal pricing.

    Args:
        df_tasks (dataframe): A dataframe with task information
        df_nodes (dataframe): A dataframe with fog node information.
        nr_timestamps (int): Number of timestamps
        task_number (int): Number of tasks.
        fn_nr (int): Number of fog nodes.
        granularity (double, optional): The granularity for the optimal price search. Defaults to 0.1.
        price_lower_value (int, optional): The lower limit for the optimal price value. Defaults to 0.
        price_upper_value (int, optional): The upper limit for the optimal price value. Defaults to 10.
        global_phi (bool, optional): Toggle for searching for the optimal price per resource or not. Defaults to False.

    Returns:
        [int, int, (double, double, double), dict]: [The maximum possible walfare, the number of allocated tasks,(optimal CPU phi, optimal RAM phi, optimal DISK phi), ]
    """

    # Set granularity for the search and also range
    # granularity = (price_upper_value-price_lower_value) / 100  # make  granularity relative
    granularity = np.arange(price_lower_value, price_upper_value, granularity)

    # Prepare variables
    social_walfares = []
    allocations = []
    allocated_task_number_list = []
    phi_list = []

    # Search for one value of the optimal price for all resources
    if global_phi:
        # Check every price in the granularity range and save results
        for phi in tqdm(granularity):
            social_welfare, tasks_allocated, mat_time_allocated, mat_start_time = attempt_allocation(
                df_tasks, df_nodes, nr_timestamps, task_number, fn_nr, phi_CPU=phi, phi_RAM=phi,
                phi_DISK=phi, verbose=False)

            # Append results to the specific list
            social_walfares.append(social_welfare)
            allocations.append((mat_time_allocated, mat_start_time))
            phi_list.append((phi, phi, phi))
            allocated_task_number_list.append(tasks_allocated)

    # Search for a value for EACH resource
    else:
        for cpu_phi in tqdm(granularity):
            for ram_phi in granularity:
                for disk_phi in granularity:
                    social_welfare, tasks_allocated, mat_time_allocated, mat_start_time = attempt_allocation(
                        df_tasks, df_nodes, nr_timestamps, task_number, fn_nr, phi_CPU=cpu_phi,
                        phi_RAM=ram_phi, phi_DISK=disk_phi, verbose=False)

                    # Append results to the specific list
                    social_walfares.append(social_welfare)
                    allocations.append((mat_time_allocated, mat_start_time))
                    phi_list.append((cpu_phi, ram_phi, disk_phi))
                    allocated_task_number_list.append(tasks_allocated)

    output = format_allocation(fn_nr, nr_timestamps,
                               allocations[social_walfares.index(max(social_walfares))][0],
                               allocations[social_walfares.index(max(social_walfares))][1])

    # Plot the evolution of the social walfare
    # plt.plot(social_walfares)
    # plt.savefig(fname="welfare_evolution.png")

    return max(social_walfares), allocated_task_number_list[
        social_walfares.index(max(social_walfares))], phi_list[
               social_walfares.index(max(social_walfares))], output
    # return max(social_walfares), allocated_task_number_list[social_walfares.index(max(social_walfares))],  allocations[social_walfares.index(max(social_walfares))][0], allocations[social_walfares.index(max(social_walfares))][1], output
