"""
--------------------------
OFFLINE OPTIMAL BENCHMARK:
---------------------------

It uses IBM CPLEX to maximise the social walfare of a current network structure and task list by solving the current environment given the usual problem restrictions.

This represents the upper bound of the social walfare.

In order to use the benchmark, call the offline_optimal() function with the required parameters.
"""

from numpy.core.fromnumeric import take
import pandas as pd
import numpy as np

from tqdm import tqdm
from docplex.mp.model import Model as Model
from docplex.mp.environment import Environment


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


def compute_walfare_based_on_tasks(model_solution, df_tasks, df_nodes):
    """Calculate the upper bound based on the number of allocated tasks.
    It is computed by summing the valuation coefficients z

    Args:
        model_solution (CPLEX.mp.model.solution): A solution object from the CPLEX model.
        df_tasks (datagrame): A dataframe of tasks and their valuation.

    Returns:
        [int]: [The total socail walfare]
    """

    # Convert to dataframe
    solution_df = as_df(model_solution)
    tasks_dict = {}
    task_usage = {}

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
        timestamp = int(i.split('_')[5][0])

        tasks_dict[task] = node
        task_usage[task] = (task_usage[task] + 1) if task in task_usage else 1

    # Calculate Social walfare by summing valuation of individual timestamps
    # return sum(
    #     [(df_tasks.loc[task, 'valuation_coefficient'] / df_tasks.loc[task, 'usage_time'] -
    #       (df_tasks.loc[task, 'CPU'] * df_nodes.loc[node, 'CPU_cost'] +
    #        df_tasks.loc[task, 'RAM'] * df_nodes.loc[node, 'RAM_cost'] +
    #        df_tasks.loc[task, 'DISK'] * df_nodes.loc[node, 'DISK_cost'])) * task_usage[task]
    #      for (task, node) in tasks_dict.items()])
    return sum(
        [(df_tasks.loc[task, 'valuation_coefficient'] -
          (df_tasks.loc[task, 'CPU'] * df_nodes.loc[node, 'CPU_cost'] +
           df_tasks.loc[task, 'RAM'] * df_nodes.loc[node, 'RAM_cost'] +
           df_tasks.loc[task, 'DISK'] * df_nodes.loc[node, 'DISK_cost'])) *
         task_usage[task]
         for (task, node) in tasks_dict.items()])


def offline_optimal(df_tasks, df_nodes, timestamp_nr, task_nr, node_nr,
                    mipgap=0.1):
    """Solve the constraints allocation problem using CPLEX.

    Args:
        df_tasks (dataframe): A dataframe containing information about the tasks.
        df_nodes (dataframe): A dataframe containing information about the fog nodes.
        timestamp_nr (int): Number of timestamps.
        task_nr (int): Number of tasks.
        node_nr (int): Number of fog nodes.
        :param mipgap: relative tolerance of the Cplex solution

    Returns:
        [double, double, int, dict]: [social welfare from allocated tasks, social walfare from model, number of allocated tasks, allocation dict]
    """

    mdl = Model(name='Maximise social welfare')
    # set the tolerance to 1%
    mdl.parameters.mip.tolerances.mipgap = mipgap

    # auxiliary variable representing if a task is allocated to a fognode n at time slot t
    z = mdl.binary_var_dict(
        ((task, fog_node, timestamp) for task in range(task_nr)
         for fog_node in range(node_nr) for timestamp in range(timestamp_nr)),
        name="task_node_time")

    # variable that is one if z change from 0 to 1
    d = mdl.binary_var_dict(((task, timestamp) for task in range(task_nr)
                             for timestamp in range(timestamp_nr - 1)),
                            name='task_timestamp')

    x = mdl.binary_var_dict(((task, node) for task in range(
        task_nr) for node in range(node_nr)), name="task_node")

    for task in range(task_nr):
        mdl.add_constraint(mdl.sum(x[task, node]
                                   for node in range(node_nr)) <= 1)

    # # Flag tasks that are impossible to allocate due to time constraints
    # for index, task in df_tasks.iterrows():
    #     if(task.loc['start_time'] + task.loc['usage_time'] > timestamp_nr - 1):
    #         for node in range(node_nr):
    #             for timestamp in range(timestamp_nr):
    #                 mdl.add_constraint(z[index,node,timestamp] == 0)

    # time constraints
    for task in range(task_nr):

        mdl.add_constraint(
            (mdl.sum(z[task, fog_node, timestamp] for timestamp in range(
                timestamp_nr) for fog_node in range(node_nr)) <= df_tasks.loc[
                 task, 'usage_time']))  # allocated time <= required time

        for fog_node in range(node_nr):

            # mdl.add_constraint((mdl.sum(z[task, fog_node, timestamp] for timestamp in range(
            #     timestamp_nr)) <= df_tasks.loc[task, 'usage_tim
            # e']))  # allocated time <= required time

            # mdl.add_constraint((mdl.sum(z[task, fog_node, timestamp] for timestamp in range(
            #     timestamp_nr)) <= df_tasks.loc[task, 'usage_time']))  # allocated time <= required time

            for timestamp in range(int(df_tasks.loc[task, 'start_time'])):
                # no usage time before the start time
                mdl.add_constraint(z[task, fog_node, timestamp] == 0)

            for timestamp in range(int(df_tasks.loc[task, 'deadline'] + 1),
                                   timestamp_nr):
                # no usage time after the deadline
                mdl.add_constraint(z[task, fog_node, timestamp] == 0)

    # resource constraints
    for timestamp in range(timestamp_nr):

        for fog_node in range(node_nr):
            mdl.add_constraint(mdl.sum(
                z[task, fog_node, timestamp] * df_tasks.loc[task, 'CPU'] for
                task in range(task_nr))
                               <= df_nodes.loc[fog_node, 'CPU'])

            mdl.add_constraint(mdl.sum(
                z[task, fog_node, timestamp] * df_tasks.loc[task, 'RAM'] for
                task in range(task_nr))
                               <= df_nodes.loc[fog_node, 'RAM'])

            mdl.add_constraint(mdl.sum(
                z[task, fog_node, timestamp] * df_tasks.loc[task, 'DISK']
                for task in range(task_nr)) <= df_nodes.loc[
                                   fog_node, 'DISK'])

    # one tasktimestamp is only processed in one fog node
    for timestamp in range(timestamp_nr):
        for task in range(task_nr):
            mdl.add_constraint(mdl.sum(z[task, fog_node, timestamp]
                                       for fog_node in range(node_nr)) <= 1)

    # tasks are non-preemptive for timestamps
    # d is 1 if z change from 0 to 1
    for task in range(task_nr):
        for timestamp in range(timestamp_nr - 1):
            mdl.add_constraint(d[task, timestamp] == (
                    mdl.sum(z[task, fog_node, timestamp + 1] for fog_node in
                            range(node_nr)) - 1
                    >= mdl.sum(
                z[task, fog_node, timestamp] for fog_node in range(node_nr))))

        # sum(d) inspect of time is less or equal to one
        mdl.add_constraint(mdl.sum(d[task, timestamp]
                                   for timestamp in
                                   range(timestamp_nr - 1)) <= 1)

    # tasks are non-preemptive for fog nodes
    for fog_node in range(node_nr):
        for task in range(task_nr):

            start_time = df_tasks.loc[task, 'start_time']
            deadline = df_tasks.loc[task, "deadline"]

            for t1 in range(start_time + 1, deadline):
                for t2 in range(t1, deadline):
                    mdl.add_constraint(
                        z[task, fog_node, t1] - z[task, fog_node, t1 - 1] + z[
                            task, fog_node, t2] -
                        z[task, fog_node, t2 + 1]
                        >= -1)

    # value_of_tasks = mdl.sum(df_tasks.loc[task, 'valuation_coefficient'] / df_tasks.loc[task, 'usage_time'] * z[task, fog_node, timestamp] * x[task, fog_node]
    #                          for task in range(task_nr) for fog_node in range(node_nr) for timestamp in range(timestamp_nr))

    value_of_tasks = mdl.sum(
        df_tasks.loc[task, 'valuation_coefficient'] * z[
            task, fog_node, timestamp] * x[
            task, fog_node]
        for task in range(task_nr) for fog_node in range(node_nr) for timestamp
        in
        range(timestamp_nr))

    CPU_cost = mdl.sum(
        df_tasks.loc[task, 'CPU'] * df_nodes.loc[fog_node, 'CPU_cost'] * z[
            task, fog_node, timestamp]
        for task in range(task_nr) for fog_node in range(node_nr) for timestamp
        in
        range(timestamp_nr))
    RAM_cost = mdl.sum(
        df_tasks.loc[task, 'RAM'] * df_nodes.loc[fog_node, 'RAM_cost'] * z[
            task, fog_node, timestamp]
        for task in range(task_nr) for fog_node in range(node_nr) for timestamp
        in
        range(timestamp_nr))
    DISK_cost = mdl.sum(
        df_tasks.loc[task, 'DISK'] * df_nodes.loc[fog_node, 'DISK_cost'] * z[
            task, fog_node, timestamp]
        for task in range(task_nr) for fog_node in range(node_nr) for timestamp
        in
        range(timestamp_nr))

    social_welfare = value_of_tasks - CPU_cost - RAM_cost - DISK_cost

    # the objective is to maximise the social welfare
    mdl.maximize(social_welfare)

    mdl.solve()

    # Initialize report variables
    number_of_allocated_tasks = 0

    # Variable for the visualization
    max_string_length_task = 0

    for n in tqdm(range(0, task_nr)):
        x = 0
        for t in range(timestamp_nr):
            for p in range(node_nr):
                if z[(n, p, t)].solution_value != 0:
                    x = 1
                    break
            if x == 1:
                break

        max_string_length_task = max(max_string_length_task, len(str(n)))
        number_of_allocated_tasks += x

    social_walfare_by_task_count = compute_walfare_based_on_tasks(
        mdl.solution, df_tasks=df_tasks, df_nodes=df_nodes)

    # mdl.print_solution()

    output = format_allocation(nr_nodes=node_nr,
                               max_str_length_task=max_string_length_task,
                               nr_timestamps=timestamp_nr,
                               cplex_solution=mdl.solution)

    return social_walfare_by_task_count, social_welfare.solution_value, number_of_allocated_tasks, output

    # return social_welfare.solution_value, number_of_allocated_tasks, z
