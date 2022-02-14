import numpy as np
import pandas as pd


def online_myopic(df_tasks, df_nodes, n_time, n_tasks, n_nodes):
    """
    online myopic mechanism based on the FCFS principle
    :param df_tasks: a dataframe of the types of tasks
    :param df_nodes: a dataframe of the information of fog nodes
    :param T: range of the time interval
    :param N: number of tasks
    :param number_of_FNs: number of fog nodes

    :return social_welfare: the overall social welfare
    :return number_of_allocated_tasks: the number of tasks that get allocated
    :return mat_time_allocated: allocation scheme for all tasks
    """
    # initialise some variables
    number_of_allocated_tasks = 0
    social_welfare = 0
    sw_list = []  # a list of social welfare after a new task arrives
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
    # number of time steps can be provided by each FN
    mat_time_temp = np.zeros([N, n_nodes], dtype=int)
    # proposed start time for each task from each fog node
    mat_start_time = np.zeros([N, n_nodes], dtype=int)
    mat_utility = np.zeros([N, n_nodes])
    # number of time steps that is finally allocated
    mat_time_allocated = np.zeros([N, n_nodes], dtype=int)
    # print("nodes information:")
    # print(df_nodes)

    for n in range(N):
        for fn in range(n_nodes):
            cost_unit_time = df_tasks.loc[n, 'CPU'] * df_nodes.loc[fn, 'CPU_cost'] + \
                             df_tasks.loc[n, 'RAM'] * df_nodes.loc[fn, 'RAM_cost'] + \
                             df_tasks.loc[n, 'storage'] * df_nodes.loc[fn, 'storage_cost']

            if df_tasks.loc[n, 'valuation_coefficient'] <= cost_unit_time:
                continue
            else:
                # print("the utility is positive")
                for usage_time in reversed(range(df_tasks.loc[n, 'usage_time'] + 1)):
                    allocated = False
                    for t in range(df_tasks.loc[n, 'start_time'], df_tasks.loc[n, 'deadline'] + 1):
                        not_enough_resource = False
                        if t + usage_time > df_tasks.loc[n, 'deadline'] + 1:
                            continue
                        for t1 in range(t, t + usage_time):
                            # if resource capacity is not enough in current time slot
                            if (df_tasks.loc[n, 'CPU'] > mat_CPU[fn, t1] or
                                    df_tasks.loc[n, 'RAM'] > mat_RAM[fn, t1] or
                                    df_tasks.loc[n, 'storage'] > mat_storage[fn, t1]):
                                not_enough_resource = True
                        if not_enough_resource:
                            continue
                        # save the usage time and start time of task n
                        mat_time_temp[n, fn] = usage_time
                        mat_start_time[n, fn] = t
                        allocated = True
                        break
                    if allocated:
                        break
            # find the utility of task n in different fn
            mat_utility[n, fn] = mat_time_temp[n, fn] * (df_tasks.loc[n, 'valuation_coefficient']
                                                         - cost_unit_time)
        # find which fn gives task n the maximum utility
        max_utility = np.amax(mat_utility[n])
        # print('array of utilites:', mat_utility[n])
        # print("max_utility:", max_utility)
        max_fn = np.where(mat_utility[n] == max_utility)[0][0]
        # print("winner fn:", max_fn)
        # update the number of allocated tasks
        if max_utility > 0:
            number_of_allocated_tasks += 1
        # save it to the final allocation matrix
        mat_time_allocated[n, max_fn] = mat_time_temp[n, max_fn]

        # update the resource capacity of FNs
        for t in range(mat_start_time[n, max_fn], mat_start_time[n, max_fn]
                                                  + mat_time_allocated[n, max_fn]):
            mat_CPU[max_fn, t] -= df_tasks.loc[n, 'CPU']
            mat_RAM[max_fn, t] -= df_tasks.loc[n, 'RAM']
            mat_storage[max_fn, t] -= df_tasks.loc[n, 'storage']

        # print("remaining CPU capacity:\n", mat_CPU)
        # print("remaining RAM capacity:\n", mat_RAM)
        # print("remaining storage capacity:\n", mat_storage)

        # update the social welfare
        social_welfare += max_utility
        sw_list.append(social_welfare)

    # print("Allocation Scheme:\n", mat_time_allocated)
    return social_welfare, number_of_allocated_tasks, mat_time_allocated, mat_start_time, sw_list
