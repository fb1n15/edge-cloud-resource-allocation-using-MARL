from __future__ import print_function
from datetime import date

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from generate_simulation_data_m import generate_synthetic_data_edge_cloud
from online_myopic_m import online_myopic
from optimal_pricing_m import optimal_pricing
from offline_optimal_m import offline_optimal
from online_optimal_m import online_optimal
import time

df_tasks, df_nodes, nr_timestamps, nr_tasks, nr_fog_nodes = generate_synthetic_data_edge_cloud(
    time_length=12,
    n_nodes=10,
    nr_tasks=20,
    resource_demand_mean_high_value=2.0,
    resource_demand_mean_low_value=2.0,
    p_high_value_task=0.5,
    k_resource=1,
    resource_demand_sigma=0.4,
    vc_ratio=100,
    global_resources=False,
    node_upper_resource_limit=3.8,
    node_lower_resource_limit=2.0,
    node_cpu_lb_cost=1.0,
    node_cpu_ub_cost=4.0,
    node_ram_lb_cost=1.0,
    node_ram_ub_cost=4.0,
    node_disk_lb_cost=1.0,
    node_disk_ub_cost=4.0,
    wait_time_range=(1, 4),
    seed=42
)


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
                [' ' for x in range(len('node_' + str(nr_nodes - 1)) - len(str(k)))])
            print(k, space + ':', [fill_string for i in range(nr_timestamps)])
        # If there are tasks for the current fog node print them sequentially and allign space to the right
        else:
            index = 0
            for (kt, vt) in v.items():

                # Print the first array of timestamps with no indent cause of the name of the node
                if index == 0:

                    space = ''.join(
                        [' ' for x in range(len('node_' + str(nr_nodes - 1)) - len(str(k)))])
                    print(k, space + ':', vt)

                # Print the other timestamp series with indent to account for the length of the name of the current node
                else:
                    space_before = ''.join(
                        [' ' for i in range(len('node_' + str(k)) + (len(str(nr_nodes - 1)) - len(str(k))))])
                    print(space_before, ':', vt)

                index += 1


def evaluate_task_range(x, y):

    offline_optimal_list = []
    online_optimal_list = []
    optimal_price_list = []
    online_myopic_list = []

    nr_fog_nodes = 0
    nr_tasks = 0
    nr_timestamps = 0

    time_of_run = time.time()
    
    seeds = [1, 9, 42,7]

    for val in range(x, y):

        print('\n\nITERATION:', val, '\n\n')
        
        online_t = []
        offline_t = []
        price_t = []
        myopic_t = []
        
        for s in seeds:

            df_tasks, df_nodes, nr_timestamps, nr_tasks, nr_fog_nodes = generate_synthetic_data_edge_cloud(
                time_length=10,
                n_nodes=10,
                nr_tasks=val,
                resource_demand_mean_high_value=2.0,
                resource_demand_mean_low_value=2.0,
                p_high_value_task=0.5,
                k_resource=1,
                resource_demand_sigma=0.4,
                vc_ratio=100,
                global_resources=False,
                node_upper_resource_limit=3.8,
                node_lower_resource_limit=2.0,
                node_cpu_lb_cost=1.0,
                node_cpu_ub_cost=4.0,
                node_ram_lb_cost=1.0,
                node_ram_ub_cost=4.0,
                node_disk_lb_cost=1.0,
                node_disk_ub_cost=4.0,
                wait_time_range=(1, 4),
                seed=s
            )

            social_walfare_offline, social_walfare_solution, allocated_tasks_offline, allocation_scheme_offline = offline_optimal(
                df_tasks, df_nodes, nr_timestamps, nr_tasks, nr_fog_nodes)

            offline_t.append(social_walfare_offline)

            social_walfare_price, allocated_tasks_price, optimal_phi, allocation_scheme_pricing = optimal_pricing(
                df_tasks=df_tasks, df_nodes=df_nodes, nr_timestamps=nr_timestamps, task_number=nr_tasks, fn_nr=nr_fog_nodes, global_phi=True, price_upper_value=4.0, price_lower_value=0.0)

            price_t.append(social_walfare_price)
           
            social_walfare_online, allocated_tasks_online, allocation_scheme_online = online_optimal(
                df_tasks, df_nodes, nr_timestamps, nr_tasks, nr_fog_nodes, verbose=False)

            online_t.append(social_walfare_online)

            social_walfare_myopic, allocated_tasks_myopic, allocation_scheme_myopic = online_myopic(
                df_tasks, df_nodes, nr_timestamps, nr_tasks, nr_fog_nodes, verbose=False)
            
            myopic_t.append(social_walfare_myopic)
            
        online_t = np.array(online_t)
        offline_t = np.array(offline_t)
        price_t = np.array(price_t)
        myopic_t = np.array(myopic_t)

        offline_optimal_list.append((offline_t.mean(), offline_t.std()))
        optimal_price_list.append((price_t.mean(), price_t.std()))
        online_optimal_list.append((online_t.mean(), online_t.std()))
        online_myopic_list.append((myopic_t.mean(), myopic_t.std()))

        
        print(optimal_price_list, '\n', offline_optimal_list,
            '\n', online_myopic_list, '\n', online_optimal_list)

        # plt.plot([a for a in range(x, y)], offline_optimal_list, label='offline')
        # plt.plot([a for a in range(x, y)], online_optimal_list, label='online')
        # plt.plot([a for a in range(x, y)], optimal_price_list, label='price')
        # plt.plot([a for a in range(x, y)], online_myopic_list,  label='myopic')

        # plt.title('Evolution of Benchmarks for an Incresing Number of Tasks')
        # plt.xlabel("Number of Tasks")
        # plt.ylabel("Walfare")
        # plt.legend()
        # plt.show()
        # plt.savefig("./" + str(nr_fog_nodes) + "_nodes_" + str(n_timesteps) + "_time_" +
        #             str(x) + "-" + str(y) + "_tasks_comparison_" + str(time_of_run) + ".png")


        fig, ax = plt.subplots()
        clrs = sns.color_palette("husl", 5)
        
        with sns.axes_style("darkgrid"):
            epochs = list(range(x, val+1))

            ax.plot(epochs, [a[0] for a in offline_optimal_list], label="Offline Optimal", c=clrs[0])
            ax.fill_between(epochs,[a[0]-a[1] for a in offline_optimal_list], [a[0]+a[1] for a in offline_optimal_list] ,alpha=0.3, facecolor=clrs[0])
            
            ax.plot(epochs,[a[0] for a in online_optimal_list], label="Online Optimal", c=clrs[1])
            ax.fill_between(epochs, [a[0]-a[1] for a in online_optimal_list], [a[0]+a[1] for a in online_optimal_list],alpha=0.3, facecolor=clrs[1])
            
            ax.plot(epochs, [a[0] for a in optimal_price_list], label="Optimal Price", c=clrs[2])
            ax.fill_between(epochs, [a[0]-a[1] for a in optimal_price_list], [a[0]+a[1] for a in optimal_price_list] ,alpha=0.3, facecolor=clrs[2])
            
            ax.plot(epochs,[a[0] for a in online_myopic_list], label="Online Myopic", c=clrs[3])
            ax.fill_between(epochs, [a[0]-a[1] for a in online_myopic_list], [a[0]+a[1] for a in online_myopic_list] ,alpha=0.3, facecolor=clrs[3])
                
            ax.legend()
            # ax.set_yscale('log')
            
        plt.title('Evolution of Benchmarks for an Incresing Number of Tasks')
        plt.xlabel("Number of Tasks")
        plt.ylabel("Walfare")
        
        plt.savefig("./metrics/plots/" + str(nr_fog_nodes) + "_nodes_" + str(nr_timestamps) + "_time_" +
                    str(x) + "-" + str(y) + "_tasks_comparison_" + str(time_of_run) + ".png")


evaluate_task_range(1, 80)


print("NUMBER OF TIMESTAMPS:", nr_timestamps, "\n\nGENERATED TASKS:\n",
      df_tasks, "\n\nGENERATED FOG NODES:\n", df_nodes, "\n\n")


# TODO: CENTRALIZED DEEP REINFORCEMENT LEARNING ALGORITHM

# TODO: STANDARD MULTI-AGENT REINFORCEMENT LEARNING ALGORITHMS


# print("\nRunning OFFLINE OPTIMAL benchmark...#")
# print("------------------------------------\n")

# social_walfare_offline, social_walfare_solution, allocated_tasks_offline, allocation_scheme_offline = offline_optimal(
#     df_tasks, df_nodes, n_timesteps, nr_tasks, nr_fog_nodes)

# display_allocation(allocation_scheme_offline,
#                    n_nodes=nr_fog_nodes, nr_tasks=nr_tasks, df_tasks=df_tasks, n_timesteps=n_timesteps)

# print("\n\nOFFLINE OPTIMAL RESULTS:\n------------------------\n")
# print("\nAllocated tasks:", allocated_tasks_offline, "/", nr_tasks, "\n")
# print("Social walfare upper bound [CPLEX]:", social_walfare_solution, "\n")
# print("Social walfare upper bound [ALLOCATION]:", social_walfare_offline, "\n")

# print("\nRunning ONLINE MYOPIC benchmark...#")
# print("------------------------------------\n")

# social_walfare_myopic, allocated_tasks_myopic, allocation_scheme_myopic = online_myopic(df_tasks, df_nodes, n_timesteps, nr_tasks, nr_fog_nodes, verbose=False)

# print("\n\nONLINE MYOPIC RESULTS:\n------------------------\n")

# display_allocation(allocation_scheme_myopic,
#                    n_nodes=nr_fog_nodes, nr_tasks=nr_tasks, df_tasks=df_tasks, n_timesteps=n_timesteps)

# print("\nAllocated tasks:", allocated_tasks_myopic, "/", nr_tasks, "\n")
# print("Social walfare:", social_walfare_myopic, "\n")


# print('ONLINE MYOPIC is', "{:.3f}%".format(social_walfare_myopic / social_walfare_offline * 100), 'of upper bound!\n')

# print("\nRunning OPTIMAL PRICING benchmark...")
# print("------------------------------------\n")

# social_walfare_price, allocated_tasks_price, optimal_phi, allocation_scheme_pricing = optimal_pricing(
#     df_tasks=df_tasks, df_nodes=df_nodes, n_timesteps=n_timesteps, task_number=nr_tasks, fn_nr=nr_fog_nodes, global_phi=True, price_upper_value=4.0, price_lower_value=0.0)

# display_allocation(allocation_scheme_pricing,
#                    n_nodes=nr_fog_nodes, nr_tasks=nr_tasks, df_tasks=df_tasks, n_timesteps=n_timesteps)

# print("\n\nOPTIMAL PRICING RESULTS:\n------------------------\n")
# print("\nOptimal phi:", optimal_phi, "\n")
# print("Allocated tasks:", allocated_tasks_price, "/", nr_tasks, "\n")
# print("Social walfare:", social_walfare_price, "\n")

# print('OPTIMAL PRICING is', "{:.3f}%".format(social_walfare_price / social_walfare_offline * 100), 'of upper bound!\n')


print("\nRunning ONLINE OPTIMAL benchmark...#")
print("------------------------------------\n")

social_walfare_online, allocated_tasks_online, allocation_scheme_online = online_optimal(
    df_tasks, df_nodes, nr_timestamps, nr_tasks, nr_fog_nodes, verbose=False)

display_allocation(output=allocation_scheme_online, nr_nodes=nr_fog_nodes,
                   nr_tasks=nr_tasks, df_tasks=df_tasks, nr_timestamps=nr_timestamps)

print("\n\nONLINE OPTIMAL RESULTS:\n------------------------\n")
print("\nAllocated tasks:", allocated_tasks_online, "/", nr_tasks, "\n")
print("Social walfare:", social_walfare_online, "\n")


# print('ONLINE OPTIMAL is', "{:.3f}%".format(
#     social_walfare_online / social_walfare_offline * 100), 'of upper bound!')
