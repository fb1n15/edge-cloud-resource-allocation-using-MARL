import numpy as np
import pandas as pd
from tqdm import tqdm


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
                    fill_string[len(str(i)):] + str(i) if start_matrix[i][
                                                              j] <= x <
                                                          start_matrix[i]
                                                          [j] +
                                                          allocation_matrix[i][
                                                              j] else fill_string
                    for x in range(nr_timestamps)]

    return output


def online_myopic(df_tasks, df_nodes, n_time, n_tasks, n_nodes, verbose=False):
    """
    online myopic mechanism based on the FCFS principle
    :param df_tasks: a dataframe of the types of tasks
    :param df_nodes: a dataframe of the information of fog nodes
    :param n_time: range of the time interval
    :param n_tasks: number of tasks
    :param n_nodes: number of fog nodes
    :param verbose: display the performance logs

    :return social_welfare: the overall social welfare
    :return number_of_allocated_tasks: the number of tasks that get allocated
    :return mat_time_allocated: allocation scheme for all tasks
    """

    # initialise some variables
    number_of_allocated_tasks = 0
    social_welfare = 0
    # remaining capacity for each time slot
    mat_CPU = []
    mat_RAM = []
    mat_storage = []
    for i in range(n_nodes):
        mat_CPU.append(np.full(n_time, df_nodes.loc[i, "CPU"], dtype=float))
        mat_RAM.append(np.full(n_time, df_nodes.loc[i, "RAM"], dtype=float))
        mat_storage.append(np.full(n_time, df_nodes.loc[i, "storage"], dtype=float))
    mat_CPU = np.array(mat_CPU)
    mat_RAM = np.array(mat_RAM)
    mat_storage = np.array(mat_storage)

    if verbose:
        print("remining capacyfyy for each time slot", "CPU:",
              mat_CPU, "RAM:", mat_RAM, "storage:", mat_storage, sep='\n')

    # number of time steps can be provided by each FN
    mat_time_temp = np.zeros([n_tasks, n_nodes], dtype=int)
    # proposed start time for each task from each fog node
    mat_start_time = np.zeros([n_tasks, n_nodes], dtype=int)
    mat_utility = np.zeros([n_tasks, n_nodes])
    # number of time steps that is finally allocated
    mat_time_allocated = np.zeros([n_tasks, n_nodes], dtype=int)
    if verbose:
        print("nodes information:")
    if verbose:
        print(df_nodes)

    for n in tqdm(range(n_tasks), disable=True):

        if verbose:
            print('\n\nCurrent timestampL: ', n, '\n\n',
                  'Alocating task:', n, '\n ', df_tasks.loc[n], '\n\n')

        # Compute the bid of each node
        for fn in range(n_nodes):

            # Check if the task has value for this node
            cost_unit_time = df_tasks.loc[n, 'CPU'] * df_nodes.loc[
                fn, 'CPU_cost'] + \
                             df_tasks.loc[n, 'RAM'] * df_nodes.loc[
                                 fn, 'RAM_cost'] + \
                             df_tasks.loc[n, 'storage'] * df_nodes.loc[
                                 fn, 'storage_cost']

            if verbose:
                print("cost unit for node", fn, cost_unit_time)

            if df_tasks.loc[n, 'valuation_coefficient'] <= cost_unit_time:
                continue
            else:
                if verbose:
                    print("the utility is positive")

                # Check how many timestamps you can alocate to that node
                # ( try from max needed timesstamps to min possible)
                for usage_time in reversed(
                        range(df_tasks.loc[n, 'usage_time'] + 1)):
                    allocated = False

                    # Check if at timestamp t + usage time there is enough
                    # free time for that node
                    for t in range(df_tasks.loc[n, 'start_time'],
                                   df_tasks.loc[n, 'deadline'] + 1):
                        not_enough_resource = False

                        if t + usage_time > df_tasks.loc[n, 'deadline'] + 1:
                            continue

                        # If reached here -> check if there is enough resources
                        # for every timestamp
                        for t1 in range(t, t + usage_time):

                            # if resource capacity is not enough in current
                            # time slot
                            if (df_tasks.loc[n, 'CPU'] > mat_CPU[fn, t1] or
                                    df_tasks.loc[n, 'RAM'] > mat_RAM[fn, t1] or
                                    df_tasks.loc[n, 'storage'] > mat_storage[fn, t1]):
                                not_enough_resource = True

                        if not_enough_resource:
                            continue

                        # If enough resources and enough free time
                        # save the usage time and start time of task n
                        mat_time_temp[n, fn] = usage_time
                        mat_start_time[n, fn] = t
                        allocated = True

                        break
                    if allocated:
                        break
            # find the utility of task n in different fn
            # mat_utility[n, fn] = mat_time_temp[n, fn] * (
            # df_tasks.loc[n, 'valuation_coefficient'] /
            # df_tasks.loc[n, 'usage_time'] - cost_unit_time)

            mat_utility[n, fn] = mat_time_temp[n, fn] * (
                    df_tasks.loc[n, 'valuation_coefficient'] - cost_unit_time)
        # find which fn gives task n the maximum utility
        max_utility = np.amax(mat_utility[n])

        if verbose:
            print('array of utilites:', mat_utility[n])
        if verbose:
            print("max_utility:", max_utility)

        max_fn = np.where(mat_utility[n] == max_utility)[0][0]

        if verbose:
            print("winner fn:", max_fn)

        # update the number of allocated tasks
        if max_utility > 0:
            number_of_allocated_tasks += 1

        # save it to the final allocation matrix
        mat_time_allocated[n, max_fn] = mat_time_temp[n, max_fn]
        if verbose:
            print("\nAllocation matrix:\n", mat_time_allocated)

        # update the resource capacity of FNs
        for t in range(mat_start_time[n, max_fn], mat_start_time[n, max_fn]
                                                  + mat_time_allocated[
                                                      n, max_fn]):
            mat_CPU[max_fn, t] -= df_tasks.loc[n, 'CPU']
            mat_RAM[max_fn, t] -= df_tasks.loc[n, 'RAM']
            mat_storage[max_fn, t] -= df_tasks.loc[n, 'storage']

        if verbose:
            print("remaining CPU capacity:\n", mat_CPU)
            print("remaining RAM capacity:\n", mat_RAM)
            print("remaining storage capacity:\n", mat_storage)

        # update the social welfare
        social_welfare += max_utility

        if verbose:
            print('\nSocial welfare: ', social_welfare, '\n')

    if verbose:
        print("Allocation Scheme:\n", mat_time_allocated)

    output = format_allocation(nr_nodes=n_nodes, nr_timestamps=n_time,
                               allocation_matrix=mat_time_allocated,
                               start_matrix=mat_start_time)

    return social_welfare, number_of_allocated_tasks, output
