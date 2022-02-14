def display_allocation(output, nr_nodes, nr_tasks, df_tasks, nr_timestamps):
    """Visualize the allocation of the tasks in CLI.

    Args:
        output (dict): A formated dict with the allocation scheme.
        nr_nodes (int): Number of fog nodes.
        nr_tasks (int): Number of tasks.
    """

    task_dict = {}

    # Define the fill string
    fill_string = ''.join(['.' for i in range(len(str(nr_tasks)))])

    # print(output)

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


def get_allocation_time(allocation_scheme):
    """Get allocation time from the generated allocation scheme

    :param allocation_scheme: one line of the allocation_scheme
    :return: task_id, start_time, finish_time, and usage_time of this task
    """
    item = allocation_scheme
    task_id = item[0]
    task_scheme = item[1]
    seen = []  # the list of allocated time steps
    for pos, element in enumerate(task_scheme):
        if f"{task_id}" in element:
            seen.append(pos)
    # print(f"task_id={task_id}")
    # print(f"seen={seen}")
    start_time = seen[0]
    finish_time = seen[-1]
    usage_time = finish_time - start_time + 1
    return task_id, start_time, finish_time, usage_time


def get_resource_usage(df_tasks, scheme, n_time, step, verbose=False):
    """get resource usage for this node

    :param step: this time step
    :param df_tasks:
    :param n_time:
    :param scheme: allocation scheme for this node
    :return: cpu_usage, ram_usage, disk_usage
    """

    # initialise the resource usage
    cpu_usage = 0
    ram_usage = 0
    disk_usage = 0
    for item in scheme.items():
        task_id = item[0]
        task_scheme = item[1]
        if f"{task_id}" in task_scheme[step]:
            cpu_usage += df_tasks.loc[task_id, 'CPU']
            ram_usage += df_tasks.loc[task_id, 'RAM']
            disk_usage += df_tasks.loc[task_id, 'DISK']
    if verbose:
        print(f"time step={step}")
        print(cpu_usage, ram_usage, disk_usage)
    return cpu_usage, ram_usage, disk_usage


def get_social_welfare(df_tasks, df_nodes, allocation_scheme, start_id=49,
        end_id=149):
    """Get social_welfare from the generated allocation scheme

    :param start_id: get the social welfare from this task
    :param end_id: to this task
    :param df_nodes: fog nodes information
    :param df_tasks: task information
    :param allocation_scheme: one line of the allocation_scheme
    :return: social_welfare of these tasks
    """
    social_welfare = 0  # initialise social welfare
    for node_id, scheme in enumerate(allocation_scheme.values()):
        for item in scheme.items():
            if item:  # if item is not empty
                (task_id, start_time, finish_time, usage_time
                ) = get_allocation_time(item)
                if start_id <= task_id <= end_id:
                    # add the value
                    social_welfare += df_tasks.loc[
                                          task_id, 'valuation_coefficient'] * usage_time
                    # minus the cost
                    for resource in ['CPU', 'RAM', 'DISK']:
                        social_welfare -= (df_tasks.loc[
                                               task_id, f'{resource}'] *
                                           df_nodes.loc[
                                               node_id, f'{resource}_cost']) * usage_time

    return social_welfare


def set_parameters(seed=0, n_tasks=100, auction_type='second-price'):
    """set parameters for the synthetic data"""
    mipgap = 0.1
    n_tasks = int(n_tasks)
    n_time = int(n_tasks / 40)
    n_nodes = 6
    resource_coefficient_original = 0.2
    resource_coefficient = (resource_coefficient_original * n_tasks / n_time)
    high_value_slackness = 0
    low_value_slackness = 6
    valuation_ratio = 10
    resource_ratio = 3
    p_high_value_task = 0.1
    # parameters of fog nodes
    h_r = 6
    l_r = 3
    l_c = 0.1
    n_c = 0.2
    h_c = 0.3
    avg_resource_capacity = {0: [h_r, h_r, h_r], 1: [h_r, h_r, h_r],
        2: [l_r, l_r, l_r],
        3: [l_r, l_r, l_r], 4: [l_r, l_r, l_r], 5: [l_r, l_r, l_r]}
    avg_unit_cost = {0: [l_c, l_c, l_c], 1: [n_c, n_c, n_c], 2: [n_c, n_c, n_c],
        3: [h_c, h_c, h_c], 4: [h_c, h_c, h_c], 5: [h_c, h_c, h_c]}
    auction_type = auction_type
    # stop exploration after 5000 steps
    epsilons_tuple = (0.6, 0.3, 0.1)
    epsilon_steps_tuple = (
        int(n_tasks / 4), int(n_tasks / 4),
        int(n_tasks / 4))
    return (seed, mipgap, n_tasks, n_time, n_nodes, resource_coefficient,
    high_value_slackness,
    low_value_slackness, valuation_ratio, resource_ratio, p_high_value_task,
    avg_resource_capacity, avg_unit_cost, epsilons_tuple, epsilon_steps_tuple,
    auction_type)
