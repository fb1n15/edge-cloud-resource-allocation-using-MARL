'''
--------------------------
ONLINE OPTIMAL BENCHMARK:
---------------------------

It uses IBM CPLEX to maximise the social walfare of a current network structure and task list by solving the current environment given the usual problem restrictions.

In order to use the benchmark, call the offline_optimal() function with the required parameters.
'''

import pandas as pd
import numpy as np
from tqdm import tqdm
from docplex.mp.model import Model

pd.set_option('display.max_rows', 50000)
pd.set_option('display.max_columns', 200000)
pd.set_option('display.width', 400000)
pd.set_option('max_colwidth', 2000000)


def as_df(cplex_solution, name_key='name', value_key='value'):
    """ Converts the solution to a pandas dataframe with two columns: variable name and values

    :param name_key: column name for variable names. Default is 'name'
    :param value_key: cilumn name for values., Default is 'value'.

    :return: a pandas dataframe, if pandas is present.
    """
    assert name_key
    assert value_key
    assert name_key != value_key
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            'Cannot convert solution to pandas.DataFrame if pandas is not available')

    names = []
    values = []
    for dv, dvv in cplex_solution.iter_var_values():
        names.append(dv.to_string())
        values.append(dvv)
    name_value_dict = {name_key: names, value_key: values}
    return pd.DataFrame(name_value_dict)


def format_allocation(nr_nodes, nr_timestamps, max_str_length_task,
        cplex_solution):
    """Visualize the allocation of the tasks in cli.

    Args:
        nr_nodes (int): Number of fog nodes.
        nr_timestamps (int): Number of timestamps.
        max_str_length_task (int): The maximum length of the name of a task.
        cplex_solution (cplex.Solution): The solution of the cplex optimizer.

    Returns:
        [output]: [A formated dict of the allocation scheme]
    """

    # Convert to dataframe
    solution_df = as_df(cplex_solution)

    # Populate the output dict with placeholders
    output = {}
    for i in range(nr_nodes):
        output["node_" + str(i)] = {}

    # Define the fill string
    fill_string = ''.join(['.' for i in range(max_str_length_task)])

    task_dict = {}
    task_distribution = {}

    # For every variable in the solution update the output
    for i in solution_df['name']:

        # Only process the allocation variables
        if len(i.split('_')) != 6:
            break

        # Only process solutions of the model aka the ones with value == 1.0
        if solution_df[solution_df['name'] == i].iloc[0]['value'] != 1.0:
            continue

        # Extract the information from the current processed variables
        task = int(i.split('_')[3])
        node = int(i.split('_')[4])
        timestamp = int(i.split('_')[5])

        # If its the first time the node has seen this task add a new entry to its position
        if task not in output['node_' + str(node)]:
            output['node_' + str(node)][task] = task_dict[task] = [
                fill_string for y in range(nr_timestamps)]

        # Update the allocated timeslot for that node
        output['node_' +
               str(node)][task][timestamp] = fill_string[len(str(task)):] + str(
            task)

    return output


def calculate_social_walfare(df_allocation, df_tasks, df_nodes):
    """Compute social walfare for a specifc allocation of tasks.

    Args:
        df_allocation (dataframe): A dataframe with the current allocation.
        df_tasks (dataframe): A dataframe with the task information.
        df_nodes (dataframe): A dataframe with the node information.

    Returns:
        [int, int]: [The social walfare, The number of allocated tasks]
    """

    social_walfare = 0
    allocated_tasks = 0

    for index, alloc_info in df_allocation.iterrows():
        task_info = df_tasks.iloc[alloc_info.loc['task']]
        node_info = df_nodes.iloc[alloc_info.loc['node']]
        usage_time = alloc_info.loc['nr_timestamps_allocated']

        # social_walfare = social_walfare + usage_time * (
        #     task_info.loc['valuation_coefficient'] / df_tasks.loc[alloc_info.loc['task'], 'usage_time'] - (
        #         task_info.loc['CPU'] * node_info.loc['CPU_cost'] +
        #         task_info.loc['RAM'] * node_info.loc['RAM_cost'] +
        #         task_info.loc['DISK'] * node_info.loc['DISK_cost']
        #     ))

        social_walfare = social_walfare + usage_time * (
                task_info.loc['valuation_coefficient'] - (
                task_info.loc['CPU'] * node_info.loc['CPU_cost'] +
                task_info.loc['RAM'] * node_info.loc['RAM_cost'] +
                task_info.loc['DISK'] * node_info.loc['DISK_cost']
        ))

        allocated_tasks += 1

    return social_walfare, allocated_tasks


def parse_solution_to_df(model_solution, df_tasks, df_nodes, current_timestamp):
    """Parse a model solution to a allocation dataframe.

    Args:
        model_solution (model.solution): The solution to a CPLEX model.
        df_tasks (dataframe): A dataframe with the task information.
        df_nodes (dataframe): A dataframe with the node information
        current_timestamp (int): The current timestamp when the function was called.

    Returns:
        [dataframe]: [A dataframe of the allocation]
    """

    solution_df = as_df(model_solution)

    d = {}

    time_d = {}

    # For every variable in the solution update the output
    for i in solution_df['name']:

        # Only process the allocation variables
        if len(i.split('_')) != 6:
            break

        # Only process solutions of the model aka the ones with value == 1.0
        if solution_df[solution_df['name'] == i].iloc[0]['value'] != 1.0:
            continue

        # Extract the information from the current processed variables
        task = int(i.split('_')[3])
        node = int(i.split('_')[4])
        timestamp = int(i.split('_')[5])
        # print(i.split('_'))

        if timestamp <= current_timestamp:

            d[task] = [node]

            if task not in time_d:
                time_d[task] = [timestamp]
            else:
                time_d[task].append(timestamp)

    task_list = []
    node_list = []
    timestamp_list = []
    usage_time_list = []
    for task in d:
        task_list.append(task)
        node_list.append(d[task][0])
        # timestamp_list.append(d[task][1])
        timestamp_list.append(min(time_d[task]))
        # usage_time_list.append(d[task][2])
        usage_time_list.append(len(time_d[task]))

    return pd.DataFrame({
        'task': task_list,
        'node': node_list,
        'allocated_time': timestamp_list,
        'nr_timestamps_allocated': usage_time_list,
    })


def allocate_waiting_tasks(df_optimal_allocation, df_tasks_waiting,
        df_tasks_stoppable, df_nodes, df_tasks,
        current_timestamp, nr_timestamps, nr_nodes,
        nr_tasks, mipgap=0.1, verbose=False):
    """Allocate the waiting tasks using CPLEX.

    Args:
        df_optimal_allocation (dataframe): A dataframe with the allocation of tasks (that have started already).
        df_tasks_waiting (dataframe): A dataframe with the tasks that have been allocated and are waiting to start.
        df_tasks_stoppable (dataframe): A dataframe with the tasks that can be stopped.
        df_nodes (dataframe): A dataframe with node information.
        df_tasks (dataframe): A dataframe with task information.
        current_timestamp (int): The timestamp when the optimization was called.
        nr_timestamps (int): Total number of timestamps.
        nr_nodes (int): Number of fog nodes in the network.
        nr_tasks (int): Number of tasks in the network.

    Returns:
        [model.solution, dataframe, int, int]: [The solution for the model,  The allocation up to the current timestamp as a dataframe, The social walfare, The largest string length from the tasks]
    """

    # print("ALLOC:\n", df_optimal_allocation, "\n\nWAITING:\n",
    #     df_tasks_waiting, '\n\nSTOPPAbLE:\n', df_tasks_stoppable, '\n\n')

    # Merge stoppable tasks with waiting tasks
    if len(df_tasks_stoppable.index) > 0:
        df_tasks_waiting = pd.concat([df_tasks_stoppable, df_tasks_waiting])

    # print("TO BE OPTIMZED:\n", df_tasks_waiting)

    mdl = Model(name='Maximise social welfare')
    # set the tolerance to 1%
    mdl.parameters.mip.tolerances.mipgap = mipgap
    # auxiliary variable representing if a task is allocated to a fognode n at time slot t
    z = mdl.binary_var_dict(
        ((task, fog_node, timestamp) for task in range(nr_tasks)
            for fog_node in range(nr_nodes) for timestamp in
        range(nr_timestamps)),
        name="task_node_time")

    # variable that is one if z change from 0 to 1
    d = mdl.binary_var_dict(((task, timestamp) for task in range(nr_tasks)
        for timestamp in range(nr_timestamps - 1)),
        name='task_timestamp')

    x = mdl.binary_var_dict(((task, node) for task in range(
        nr_tasks) for node in range(nr_nodes)), name="task_node")

    # Add tasks already allocated
    # You cannot change history
    for index, i in df_optimal_allocation.iterrows():

        if i.loc['nr_timestamps_allocated'] > 0 and i.loc['allocated_time'] + \
                i.loc['nr_timestamps_allocated'] <= current_timestamp:
            for timestamp in range(i.loc['allocated_time'],
                    i.loc['allocated_time'] + i.loc[
                        'nr_timestamps_allocated']):
                # print("ADDING:", [i.loc['task'], i.loc['node'], timestamp], "=1\n")

                mdl.add_constraint(
                    z[i.loc['task'], i.loc['node'], timestamp] == 1)
            mdl.add_constraint(x[i.loc['task'], i.loc['node']] == 1)

            # else:
            #     for timestamp in range(i.loc['allocated_time'], current_timestamp):

            #         print("Setting ",[i.loc['task'], i.loc['node'], timestamp],  "to 1\n\n")

            #         mdl.add_constraint(z[i.loc['task'], i.loc['node'], timestamp] == 1)
            #         mdl.add_constraint(x[i.loc['task'],  i.loc['node']] == 1)

    # Add constraint for stoppable tasks
    for index, i in df_tasks_stoppable.iterrows():
        for node in range(nr_nodes):
            # cannot migrate running tasks
            if node != df_optimal_allocation.loc[
                df_optimal_allocation['task'] == index, 'node'].item():
                mdl.add_constraint(x[index, node] == 0)
                for timestamp in range(nr_timestamps):
                    mdl.add_constraint(z[index, node, timestamp] == 0)
                    mdl.add_constraint(x[index, node] == 0)
            else:
                if verbose:
                    print("ALLOWING JUST NODE", node, "FOR TASK", index, "\n")
                else:
                    pass

    # One task can only run in one node
    for task, info in df_tasks_waiting.iterrows():
        mdl.add_constraint(mdl.sum(x[task, node]
            for node in range(nr_nodes)) <= 1)

    # time constraints
    for task, i in df_tasks_waiting.iterrows():
        # allocated time (starts from now) <= required time
        mdl.add_constraint(
            (mdl.sum(z[task, fog_node, timestamp] for timestamp in
                range(current_timestamp,
                    nr_timestamps) for fog_node in range(nr_nodes)) <=
             df_tasks_waiting.loc[task, 'usage_time']))

        for fog_node in range(nr_nodes):

            # # no usage time before the start time
            # for timestamp in range(
            #         int(df_tasks_waiting.loc[task, 'start_time'])):
            #     # print((task, fog_node, timestamp), '=0')
            #     mdl.add_constraint(z[task, fog_node, timestamp] == 0)

            # no usage time after the deadline
            for timestamp in range(
                    int(df_tasks_waiting.loc[task, 'deadline'] + 1),
                    nr_timestamps):
                # print((task, fog_node, timestamp), '=0')
                mdl.add_constraint(z[task, fog_node, timestamp] == 0)

            # no usage time before the start time, if the task is not running
            if task not in df_optimal_allocation['task'].values or (
                    df_optimal_allocation.loc[df_optimal_allocation['task']
                                              == task, 'nr_timestamps_allocated'].item() == 0):

                for timestamp in range(
                        int(df_tasks_waiting.loc[task, 'start_time'])):
                    # print((task, fog_node, timestamp),'=0')
                    mdl.add_constraint(z[task, fog_node, timestamp] == 0)
            # no usage time before the real start running time, if the task is running
            elif task in df_optimal_allocation['task'].values and \
                    df_optimal_allocation.loc[df_optimal_allocation[
                                                  'task'] == task, 'nr_timestamps_allocated'].item() >= 0:

                for timestamp in range(df_optimal_allocation.loc[
                    df_optimal_allocation[
                        'task'] == task, 'allocated_time'].item()):
                    mdl.add_constraint(z[task, fog_node, timestamp] == 0)

    # cannot allocate before current timestamp
    for timestamp in range(current_timestamp):
        for task, i in df_tasks_waiting.iterrows():
            for node in range(nr_nodes):

                if task not in df_optimal_allocation['task'].values or (
                        task in df_optimal_allocation['task'].values and
                        df_optimal_allocation.loc[df_optimal_allocation[
                                                      'task'] == task, 'nr_timestamps_allocated'].item() == 0):
                    mdl.add_constraint(z[task, node, timestamp] == 0)

    # resource constraints
    for timestamp in range(nr_timestamps):

        for fog_node in range(nr_nodes):
            mdl.add_constraint(mdl.sum(
                z[task, fog_node, timestamp] * df_tasks_waiting.loc[task, 'CPU']
                    for task, i in df_tasks_waiting.iterrows())
                               <= df_nodes.loc[fog_node, 'CPU'])

            mdl.add_constraint(mdl.sum(
                z[task, fog_node, timestamp] * df_tasks_waiting.loc[task, 'RAM']
                    for task, i in df_tasks_waiting.iterrows())
                               <= df_nodes.loc[fog_node, 'RAM'])

            mdl.add_constraint(mdl.sum(
                z[task, fog_node, timestamp] * df_tasks_waiting.loc[
                    task, 'DISK']
                    for task, i in df_tasks_waiting.iterrows()) <= df_nodes.loc[
                                   fog_node, 'DISK'])

    # one tasktimestamp is only processed in one fog node
    for timestamp in range(nr_timestamps):
        for task, i in df_tasks_waiting.iterrows():
            mdl.add_constraint(mdl.sum(z[task, fog_node, timestamp]
                for fog_node in range(nr_nodes)) <= 1)

    # tasks are non-preemptive for timestamps
    # d is 1 if z change from 0 to 1
    # and sum(d) shoud <= 1
    for task, i in df_tasks_waiting.iterrows():
        for timestamp in range(nr_timestamps - 1):
            mdl.add_constraint(d[task, timestamp] == (mdl.sum(
                z[task, fog_node, timestamp + 1] for fog_node in
                    range(nr_nodes)) - 1 >= mdl.sum(
                z[task, fog_node, timestamp] for fog_node in
                    range(nr_nodes))))

        # sum(d) inspect of time is less or equal to one
        mdl.add_constraint(mdl.sum(d[task, timestamp]
            for timestamp in range(current_timestamp,
            nr_timestamps - 1)) <= 1)

    # tasks are non-preemptive for fog nodes
    for fog_node in range(nr_nodes):
        for task, i in df_tasks_waiting.iterrows():

            if task in df_tasks_stoppable.index:
                # print(df_optimal_allocation.loc[df_optimal_allocation['task'] == task, 'allocated_time'])
                start_time = int(
                    df_optimal_allocation.loc[df_optimal_allocation[
                                                  'task'] == task, 'allocated_time'])
            else:
                start_time = int(df_tasks_waiting.loc[task, 'start_time'])

            deadline = int(df_tasks_waiting.loc[task, "deadline"])

            for t1 in range(start_time + 1, deadline):
                for t2 in range(t1, deadline):
                    mdl.add_constraint(
                        z[task, fog_node, t1] - z[task, fog_node, t1 - 1] + z[
                            task, fog_node, t2] - z[task, fog_node, t2 + 1]
                        >= -1)

    # value_of_tasks = mdl.sum(df_tasks.loc[task, 'valuation_coefficient'] / df_tasks.loc[task, 'usage_time'] * z[task, fog_node, timestamp] * x[task, fog_node]
    #                          for task, i in df_tasks_waiting.iterrows() for fog_node in range(n_nodes) for timestamp in range(n_timesteps))

    value_of_tasks = mdl.sum(df_tasks.loc[task, 'valuation_coefficient'] * z[
        task, fog_node, timestamp] * x[task, fog_node]
        for task, i in df_tasks_waiting.iterrows() for
        fog_node in range(nr_nodes) for timestamp in
        range(nr_timestamps))

    CPU_cost = mdl.sum(
        df_tasks.loc[task, 'CPU'] * df_nodes.loc[fog_node, 'CPU_cost'] * z[
            task, fog_node, timestamp]
            for task, i in df_tasks_waiting.iterrows() for fog_node in
            range(nr_nodes) for timestamp in range(nr_timestamps))
    RAM_cost = mdl.sum(
        df_tasks.loc[task, 'RAM'] * df_nodes.loc[fog_node, 'RAM_cost'] * z[
            task, fog_node, timestamp]
            for task, i in df_tasks_waiting.iterrows() for fog_node in
            range(nr_nodes) for timestamp in range(nr_timestamps))
    DISK_cost = mdl.sum(
        df_tasks.loc[task, 'DISK'] * df_nodes.loc[fog_node, 'DISK_cost'] * z[
            task, fog_node, timestamp]
            for task, i in df_tasks_waiting.iterrows() for fog_node in
            range(nr_nodes) for timestamp in range(nr_timestamps))
    # make the solver perfers early time slots.
    time_cost = mdl.sum(1e-5 * timestamp * z[task, fog_node, timestamp]
        for task, i in df_tasks_waiting.iterrows() for fog_node in
        range(nr_nodes) for timestamp in range(nr_timestamps))

    social_welfare = (value_of_tasks - CPU_cost - RAM_cost - DISK_cost -
                      time_cost)
    # social_welfare = (value_of_tasks - CPU_cost - RAM_cost - DISK_cost)

    # the objective is to maximise the social welfare
    mdl.maximize(social_welfare)

    mdl.solve()

    # Initialize report variables
    number_of_allocated_tasks = 0

    # Variable for the visualization
    max_string_length_task = 0

    if verbose:
        mdl.print_solution()

    for task, i in df_tasks_waiting.iterrows():
        x = 0
        # for t in range(n_timesteps):
        #     for p in range(n_nodes):
        #         if z[(task, p, t)].solution_value != 0:
        #             x = 1
        #             break
        #     if x == 1:
        #         break

        max_string_length_task = max(max_string_length_task, len(str(task)))
        number_of_allocated_tasks += x

    return mdl.solution, parse_solution_to_df(mdl.solution, df_tasks_waiting,
        df_nodes,
        current_timestamp), social_welfare.solution_value, max_string_length_task


def display_allocation(output, nr_nodes, nr_tasks, df_tasks,
        max_str_length_task, nr_timestamps):
    """Visualize the allocation of the tasks in CLI.

    Args:
        output (dict): A formated dict with the allocation scheme.
        nr_nodes (int): Number of fog nodes.
        nr_tasks (int): Number of tasks.
        df_tasks (dataframe): A dataframe with the task information.
        max_str_length_task (int): The length of the string name of the largest task in the allocation to be displayed.
        nr_timestamps (int): The number of timestamps in the allocation.
    """

    task_dict = {}

    # Define the fill string
    fill_string = ''.join(['.' for i in range(max_str_length_task)])

    for node in output:
        for task, alloc in output[node].items():
            task_dict[task] = sum(
                [1 if x != fill_string else 0 for x in alloc])

    for task in task_dict:
        print("Task", task, ": ", task_dict[task],
            "/", df_tasks.loc[task, 'usage_time'])

    for (k, v) in output.items():

        # If there are no tasks for current fog node, print a placeholder
        if not v:
            space = ''.join(
                [' ' for x in
                    range(len('node_' + str(nr_nodes - 1)) - len(str(k)))])
            print(k, space + ':', [fill_string for i in range(nr_timestamps)])
        # If there are tasks for the current fog node print them sequentially and allign space to the right
        else:
            index = 0
            for (kt, vt) in v.items():

                # Print the first array of timestamps with no indent cause of the name of the node
                if index == 0:

                    space = ''.join(
                        [' ' for x in
                            range(len('node_' + str(nr_nodes - 1)) - len(
                                str(k)))])
                    print(k, space + ':', vt)

                # Print the other timestamp series with indent to account for the length of the name of the current node
                else:
                    space_before = ''.join(
                        [' ' for i in range(len('node_' + str(k)) + (
                                len(str(nr_nodes - 1)) - len(str(k))))])
                    print(space_before, ':', vt)

                index += 1


def check_if_task_is_stopped(model_solution, current_timestamp, task):
    """Check if a task in the allocation is stopped or not.

    Args:
        model_solution (model.solution): A solution of a CPLEX model.
        current_timestamp (int): The timestamp for the call of this function.
        task (int): The task to be checked.

    Returns:
        [bool]: [True if the tasks has been stopped]
    """

    # Convert to dataframe
    solution_df = as_df(model_solution)

    max_found_timestamp = -1

    # For every variable in the solution update the output
    for i in solution_df['name']:

        # Only process the allocation variables
        if len(i.split('_')) != 6:
            break

        # Only process solutions of the model aka the ones with value == 1.0
        if solution_df[solution_df['name'] == i].iloc[0]['value'] != 1.0:
            continue

        # Extract the information from the current processed variables
        parsed_task = int(i.split('_')[3])
        parsed_node = int(i.split('_')[4])
        parsed_timestamp = int(i.split('_')[5][0])

        if task == parsed_task:
            max_found_timestamp = max(max_found_timestamp, parsed_timestamp)

    return current_timestamp >= max_found_timestamp


def online_optimal(df_tasks, df_nodes, nr_timestamps, task_number, nr_nodes,
        mipgap=0.1, verbose=False):
    """The optimal pricing benchmark algorithm. I computes a fixed unit price for every resource and then every fn bids for the task on a reverse reverse_auction manner.

    Args:
        :param mipgap: relative gap of Cplex solution
        :param df_tasks (Dataframe): A dataframe containing the incoming tasks.
        :param df_nodes (Dataframe): A dataframe containing the information about the fog nodes.
        :param n_timesteps (int): Number of timestamps for the benchmark.
        :param task_number (int): Total number of tasks.
        :param n_nodes (int): Total number of fog nodes.
        :param verbose (bool): Check for the logging of progress. Default is False.

    Returns:
        [int, int, dataframe, dataframe]: The socail welfare,
        the number of allocated tasks, the allocation matrix,
        the start time of each task.
    """

    df_tasks_waiting = pd.DataFrame(
        {'valuation_coefficient': [], 'arrive_time': [],
            'start_time': [], 'deadline': [],
            'usage_time': [], 'CPU': [],
            'RAM': [], 'DISK': []
        })
    df_tasks_allocated = pd.DataFrame(
        {'valuation_coefficient': [], 'arrive_time': [],
            'start_time': [], 'deadline': [],
            'usage_time': [], 'CPU': [],
            'RAM': [], 'DISK': []
        })
    df_tasks_stoppable = pd.DataFrame(
        {'valuation_coefficient': [], 'arrive_time': [],
            'start_time': [], 'deadline': [],
            'usage_time': [], 'CPU': [],
            'RAM': [], 'DISK': []
        })

    df_full_allocation = pd.DataFrame({
        'task': [],
        'node': [],
        'allocated_time': [],
        'nr_timestamps_allocated': [],
    })

    # Initialize metrics
    social_welfare = 0
    allocated_tasks = 0

    max_string_length_task = 0
    task_index = 0
    last_solution = None

    # Process each task in the tasklist
    for current_timestamp in tqdm(range(nr_timestamps)):

        if verbose:
            print("Current timestamp:", current_timestamp, "\n\n")
        print("Current timestamp:", current_timestamp, "\n\n")

        print(df_full_allocation)
        # Check for allocated tasks that have to start
        for index, i in df_full_allocation.iterrows():

            if i.loc['allocated_time'] < current_timestamp and i.loc[
                'task'] in df_tasks_waiting.index:
                if verbose:
                    print("Started task ", i.loc['task'])

                print("Started task ", i.loc['task'])

                df_tasks_stoppable = pd.concat(
                    [df_tasks_stoppable,
                        df_tasks.iloc[i.loc['task']].to_frame().T])

                # Remove item from waiting tasks list
                df_tasks_waiting.drop(i.loc['task'], inplace=True)
                df_tasks_allocated = pd.concat(
                    [df_tasks_allocated,
                        df_tasks.iloc[i.loc['task']].to_frame().T])

            # #
            # if i.loc['task'] in df_tasks_stoppable.index:

            #     df_full_allocation.loc[df_full_allocation['task'] == i.loc['task'],
            #                            'nr_timestamps_allocated'] = df_full_allocation.loc[df_full_allocation['task'] == i.loc['task'], 'nr_timestamps_allocated'] + 1

        # Update the possible start time of the waiting tasks to max(current timestamp, original_start_time) and not pass the deadline
        for index, i in df_tasks_waiting.iterrows():
            df_tasks_waiting.loc[index, 'start_time'] = min(
                df_tasks_waiting.loc[index, 'deadline'], max(
                    current_timestamp,
                    df_tasks_waiting.loc[index, 'start_time']))

        # Update the possible start time of the waiting tasks to max(current timestamp, original_start_time)
        for index, i in df_tasks_stoppable.iterrows():
            if verbose:
                print(f"index={index}")
            df_tasks_stoppable.loc[index, 'start_time'] = min(
                df_tasks_stoppable.loc[index, 'deadline'], max(
                    current_timestamp,
                    df_tasks_stoppable.loc[index, 'start_time']))

        current_task = df_tasks.iloc[task_index] if task_index >= 0 else \
            df_tasks.iloc[0]

        # Check for any new incoming tasks and if there are any try to allocate using myopic
        while int(current_task.loc[
            'arrive_time']) - current_timestamp == -1 and task_index != -1:

            max_string_length_task = max(
                max_string_length_task, len(str(task_index)))
            if verbose:
                print("\n\nAllocating task", task_index, '\n')

            df_tasks_waiting = pd.concat(
                [df_tasks_waiting, current_task.to_frame().T])

            # Advance with the new task if there are any left, if not set task index to -1 to flag for last tasks
            task_index = (
                    task_index + 1) if task_index < task_number - 1 else -1
            current_task = df_tasks.iloc[task_index] if task_index >= 0 else \
                df_tasks.iloc[0]

        if verbose:
            print("Current waiting tasks:\n", df_tasks_waiting, "\n\n")

        if len(df_tasks_waiting.index) > 0:

            # Update the usage time and the start time
            for index, i in df_tasks_stoppable.iterrows():
                df_tasks_stoppable.loc[index,
                'usage_time'] = df_tasks_stoppable.loc[
                                    index, 'usage_time'] - 1

            for index, i in df_tasks_stoppable.iterrows():
                # Check if the usage time (remaining) is 0 or the deadline has been reached
                if df_tasks_stoppable.loc[index, 'usage_time'] == 0 or \
                        df_tasks_stoppable.loc[index, 'start_time'] > \
                        df_tasks_stoppable.loc[index, 'deadline']:
                    df_tasks_stoppable.drop(index, inplace=True)

            # count the number of waiting tasks
            index = df_tasks_waiting.index
            n_waiting_tasks = len(index)
            print(f"current_timestamp={current_timestamp}")
            print(f"number of waiting tasks = {n_waiting_tasks}")

            # Run optimization on started tasks + waiting tasks at the end of each timestamp
            (last_solution, df_full_allocation, last_walfare,
            max_string_length_task) = allocate_waiting_tasks(
                df_optimal_allocation=df_full_allocation,
                df_tasks_waiting=df_tasks_waiting,
                df_tasks_stoppable=df_tasks_stoppable,
                df_nodes=df_nodes,
                df_tasks=df_tasks,
                current_timestamp=current_timestamp,
                nr_timestamps=nr_timestamps,
                nr_nodes=nr_nodes,
                nr_tasks=task_number, mipgap=mipgap)

            # Update the stoppable tasks dataframe
            for index, i in df_tasks_stoppable.iterrows():

                # If task has been stopped remove from stoppable tasks
                if check_if_task_is_stopped(model_solution=last_solution,
                        current_timestamp=current_timestamp,
                        task=index):
                    if verbose:
                        print('Task', index, "has been stopped and removed\n")
                    df_tasks_stoppable.drop(index, inplace=True)

            if verbose:
                print("Allocation Past optimization:\n", df_full_allocation)

                display_allocation(
                    output=format_allocation(
                        nr_nodes=nr_nodes,
                        nr_timestamps=nr_timestamps,
                        max_str_length_task=max_string_length_task,
                        cplex_solution=last_solution,
                    ),
                    nr_nodes=nr_nodes,
                    nr_tasks=task_number,
                    df_tasks=df_tasks,
                    nr_timestamps=nr_timestamps,
                    max_str_length_task=max_string_length_task
                )

                print("ALLOCATION AFTER OPT:\n", df_full_allocation)
                print("\n\nMODEL WALFARE:", last_walfare, )

            # update the social welfare
            social_welfare, allocated_tasks = calculate_social_walfare(
                df_full_allocation, df_tasks, df_nodes)

    # Print stats
    if verbose:
        print("Allocation Scheme:\n", df_full_allocation, '\n\n\n')

    if verbose:
        print("TOTAL SOCIAL WELFARE: ", social_welfare, '\n')

    return social_welfare, allocated_tasks, format_allocation(nr_nodes=nr_nodes,
        nr_timestamps=nr_timestamps,
        max_str_length_task=max_string_length_task,
        cplex_solution=last_solution)
