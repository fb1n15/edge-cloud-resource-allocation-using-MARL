from itertools import combinations
import csv
import pandas as pd
import numpy as np
from generate_simulation_data import *
from value_iteration import *

pd.set_option('display.expand_frame_repr', False)  # print the whole dataframe

# the main program
number_of_tasks = 2
V = {}  # value function: state -> value
pi = {}  # policy: (state+price) -> probability
actions = [0, 1, 2, 3]  # [reject, 0.3, 0.6, 1.0] of the value of the task
# generate a seqence of analytics tasks
k_resource = 7
df_tasks, df_nodes, n_time, n_tasks, n_nodes = \
    generate_synthetic_data_edge_cloud(n_tasks=number_of_tasks, n_time=10, k_resource=k_resource,
                                       p_high_value_tasks=0.5, resource_demand_low=5, seed=1)
# initialise a value iteration object
vi = ValueIteration(df_tasks, df_nodes, actions, k_resource)
# algorithm parameters
gamma = 0.9
theta = 0.1
states = []  # the list of all possible states
# generate the initial state
task_no = 0
cpu_demand = df_tasks.loc[task_no, 'CPU']
ram_demand = df_tasks.loc[task_no, 'RAM']
storage_demand = df_tasks.loc[task_no, 'storage']
vc = df_tasks.loc[task_no, 'valuation_coefficient']
start_time = df_tasks.loc[task_no, 'start_time']
deadline = df_tasks.loc[task_no, 'deadline']
usage_time = df_tasks.loc[task_no, 'usage_time']
initial_state = [3.0, 1, 3, 2, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0]
# update the state
initial_state[4] = cpu_demand;
initial_state[5] = ram_demand;
initial_state[6] = storage_demand;
initial_state[0] = vc;
initial_state[1] = start_time
initial_state[2] = deadline;
initial_state[3] = usage_time;
initial_state[38] = task_no
print(f"types of tasks: ")
print(df_tasks)
print(f"initial state: {initial_state}")
states.append(initial_state)

# find all possible resouce occupencies in a time slot
list_resouce_occupency = []
for i in range(number_of_tasks):  # how many tasks run together in this time slot
    for j in [p for p in itertools.product(range(number_of_tasks), repeat=i)]:
        cpu = np.float64(0)
        ram = np.float64(0)
        storage = np.float64(0)
        for k in j:
            cpu += df_tasks.loc[k, 'CPU']
            ram += df_tasks.loc[k, 'RAM']
            storage += df_tasks.loc[k, 'storage']
        if cpu <= k_resource and ram <= k_resource and storage <= k_resource:
            list_resouce_occupency.append(["{:.6f}".format(cpu/k_resource),
                                           "{:.6f}".format(ram/k_resource),
                                           "{:.6f}".format(storage/k_resource)])

# print(f"the list of all possible occupencies: {list_resouce_occupency}")

list_occupency_ten_time_slots = list(itertools.product(list_resouce_occupency, repeat=10))
# make a flat list out of list of lists
list_occupency_all_possibility = []
for lt in list_occupency_ten_time_slots:
    lt = [item for sublist in lt for item in sublist]
    list_occupency_all_possibility.append(lt)


# print("the list of possible occupencies of 10 time slots:")
# print(list_occupency_all_possibility)

# generate all possible states
for n in range(N):
    print(f"generating states for task {n}")
    # print(initial_state)
    task_no = n
    cpu_demand = df_tasks.loc[task_no, 'CPU']
    ram_demand = df_tasks.loc[task_no, 'RAM']
    storage_demand = df_tasks.loc[task_no, 'storage']
    vc = df_tasks.loc[task_no, 'valuation_coefficient']
    start_time = df_tasks.loc[task_no, 'start_time']
    deadline = df_tasks.loc[task_no, 'deadline']
    usage_time = df_tasks.loc[task_no, 'usage_time']
    time_pass = int(df_tasks.loc[task_no, 'arrive_time'])
    # update the state
    initial_state[4] = cpu_demand;
    initial_state[5] = ram_demand;
    initial_state[6] = storage_demand;
    initial_state[0] = vc;
    initial_state[1] = start_time
    initial_state[2] = deadline;
    initial_state[3] = usage_time;
    initial_state[38] = task_no
    for occupency in list_occupency_all_possibility:
        initial_state[7:37] = occupency
        for i in range(3 * time_pass):
            initial_state.pop(7)
            initial_state.insert(36, 0)
        for a in actions:
            initial_state[37] = a
            states.append(initial_state.copy())
    print(f"the number of states = {len(states)}")

# convert a list of lists to a list of tuples
states = [tuple(l) for l in states]


np.savetxt("all_states_float64.csv", states, delimiter=", ", fmt='%s')