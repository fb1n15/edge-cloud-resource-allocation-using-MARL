import numpy as np
import pandas as pd
from docplex.mp.model import Model as Model


def offline_optimal(df_tasks, df_nodes, n_time, n_tasks, n_nodes, gap=0.1):
    mdl = Model(name='Maximise social welfare')
    # set the tolerance to 1%
    mdl.parameters.mip.tolerances.mipgap = 0.01

    # auxiliary variable representing if fog node p is allocated for user n at time slot t
    z = mdl.binary_var_dict(((n, p, t) for n in range(N)
                             for p in range(n_nodes) for t in range(T)), name="z")
    # variable that is one if z change from 0 to 1
    d = mdl.binary_var_dict(((n, t) for n in range(N)
                            for t in range(T-1)), name='d')

    value_of_tasks = mdl.sum(df_tasks.loc[n, 'valuation_coefficient'] * z[n, p, t]
                             for n in range(N) for p in range(n_nodes) for t in range(T))
    CPU_cost = mdl.sum(df_tasks.loc[n, 'CPU'] * df_nodes.loc[p, 'CPU_cost'] * z[n, p, t]
                       for n in range(N) for p in range(n_nodes) for t in range(T))
    RAM_cost = mdl.sum(df_tasks.loc[n, 'RAM'] * df_nodes.loc[p, 'RAM_cost'] * z[n, p, t]
                       for n in range(N) for p in range(n_nodes) for t in range(T))
    storage_cost = mdl.sum(df_tasks.loc[n, 'storage'] * df_nodes.loc[p, 'storage_cost'] * z[n, p, t]
                           for n in range(N) for p in range(n_nodes) for t in range(T))
    social_welfare = value_of_tasks - CPU_cost - RAM_cost - storage_cost

    # the objective is to maximise the social welfare
    mdl.maximize(social_welfare)

    # time constraints
    for n in range(N):
        mdl.add_constraint(mdl.sum(z[n, p, t] for p in range(n_nodes) for t in range(T)) <=
                           df_tasks.loc[n, 'usage_time'])  # allocated time <= required time
        for p in range(n_nodes):
            for t in range(int(df_tasks.loc[n, 'start_time'])):
                mdl.add_constraint(z[n, p, t] == 0)  # no usage time before the start time
            for t in range(int(df_tasks.loc[n, 'deadline'] + 1), T):
                mdl.add_constraint(z[n, p, t] == 0)  # no usage time after the deadline
                
    # resource constraints
    for t in range(T):
        for p in range(n_nodes):
            mdl.add_constraint(mdl.sum(z[n, p, t] * df_tasks.loc[n, 'CPU'] for n in range(N))
                               <= df_nodes.loc[p, 'CPU'])
            mdl.add_constraint(mdl.sum(z[n, p, t] * df_tasks.loc[n, 'RAM'] for n in range(N))
                               <= df_nodes.loc[p, 'RAM'])
            mdl.add_constraint(mdl.sum(z[n, p, t] * df_tasks.loc[n, 'storage'] for n in range(N))
                               <= df_nodes.loc[p, 'storage'])

    # one task is only processed in one fog node
    for n in range(N):
        for t in range(T):
            mdl.add_constraint(mdl.sum(z[n, p, t] for p in range(n_nodes)) <= 1)
    
    # tasks are non-preemptive
    # d is 1 if z change from 0 to 1
    for n in range(N):
        for t in range(T-1):
            mdl.add_constraint(d[n, t] == (mdl.sum(z[n, p, t+1] for p in range(n_nodes)) - 1
                               >= mdl.sum(z[n, p, t] for p in range(n_nodes))))
    # sum(d) inspect of time is less or equal to one
        mdl.add_constraint(mdl.sum(d[n, t] for t in range(T-1)) <= 1)

    # # tasks are non-preemptive
    # for p in range(n_nodes):
    #     for n in range(n_tasks):
    #         start_time = df_tasks.loc[n, 'start_time']
    #         deadline = df_tasks.loc[n, "deadline"]
    #         for t1 in range(start_time + 1, deadline):
    #             for t2 in range(t1, deadline):
    #                 mdl.add_constraint(z[n, p, t1] - z[n, p, t1-1] + z[n, p, t2] - z[n, p, t2+1]
    #                                    >= -1)

    mdl.solve()

    number_of_allocated_tasks = 0

    for n in range(0, N):
        x = 0
        for t in range(T):
            for p in range(n_nodes):
                if z[(n, p, t)].solution_value != 0:
                    x = 1
        number_of_allocated_tasks += x

    return social_welfare.solution_value, number_of_allocated_tasks, z


