from pprint import pprint

import numpy as np

from common.config_file_handler import load_yaml
from environments.edge_cloud.simulation.environment import EdgeCloudEnv
from environments.edge_cloud.simulation.environment import EdgeCloudEnv1
from environments.edge_cloud.simulation.environment import argmax_earliest
import pytest


def test_argmax_earliest():
    # case 1
    utilities1 = [0, 0, 1]
    start_times1 = [0, 0, 1]
    assert 2 == argmax_earliest(utilities1, start_times1)
    # case 2, earlier task shoud be prioritised
    utilities2 = [0, 0, 0]
    start_times2 = [1, 0, 1]
    assert 1 == argmax_earliest(utilities2, start_times2)
    # case 3
    utilities3 = [2, 1, 1]
    start_times3 = [1, 0, 1]
    assert 0 == argmax_earliest(utilities3, start_times3)


@pytest.fixture
def edge_env() -> EdgeCloudEnv:
    config = load_yaml(
        '/marl-edge-cloud/configs/experiments/edge_cloud/hyperparameters/cpu_ppo_fc_independent_with_history.yaml')
    env_config = config['env-config']
    edge_env = EdgeCloudEnv(env_config)
    edge_env.reset()
    return edge_env


@pytest.fixture
def edge_env1() -> EdgeCloudEnv:
    config = load_yaml(
        '/marl-edge-cloud/configs/experiments/edge_cloud/hyperparameters/cpu_ppo_fc_independent_with_history.yaml')
    env_config = config['env-config']
    edge_env = EdgeCloudEnv1(env_config)
    edge_env.reset()
    return edge_env


def test_reset(edge_env):
    initial_obs = edge_env.reset()
    first_task = edge_env.df_tasks.iloc[0]
    print(initial_obs)
    print("first task:")
    print(first_task)
    print("node info:")
    print(edge_env.df_nodes.iloc[0])

    assert first_task['valuation_coefficient'] == initial_obs['drone_0'][0]
    assert first_task['start_time'] - int(first_task['arrive_time']) - 1 == \
           initial_obs['drone_0'][5]
    assert first_task['deadline'] - int(first_task['start_time']) + 1 == \
           initial_obs['drone_0'][6]
    assert first_task['usage_time'] == initial_obs['drone_0'][1]
    assert first_task['CPU'] == initial_obs['drone_0'][2]
    assert first_task['RAM'] == initial_obs['drone_0'][3]
    assert first_task['storage'] == initial_obs['drone_0'][4]
    # print(edge_env.history_len)
    actions_history_list = [1 for _ in range(edge_env.history_len * edge_env.n_nodes)]
    assert initial_obs['drone_0'][
           -edge_env.history_len * edge_env.n_nodes:] == actions_history_list


def test_reset1(edge_env1):
    """Test the reset function of EdgeCloudEnv1 (Env with shorter observation)"""
    initial_obs = edge_env1.reset()
    first_task = edge_env1.df_tasks.iloc[0]
    print("initial_obs:")
    print(initial_obs)
    print(f"observation's number of dimension: {len(initial_obs['drone_0'])}")
    print("first task:")
    print(first_task)
    print("node info:")
    print(edge_env1.df_nodes.iloc[0])
    # task type part of the observation
    assert first_task['valuation_coefficient'] == initial_obs['drone_0'][0]
    assert first_task['start_time'] - int(first_task['arrive_time']) - 1 == \
           initial_obs['drone_0'][5]
    assert first_task['deadline'] - int(first_task['start_time']) + 1 == \
           initial_obs['drone_0'][6]
    assert first_task['usage_time'] == initial_obs['drone_0'][1]
    assert first_task['CPU'] == initial_obs['drone_0'][2]
    assert first_task['RAM'] == initial_obs['drone_0'][3]
    assert first_task['storage'] == initial_obs['drone_0'][4]
    # history part of the observation
    # print(edge_env1.history_len)
    actions_history_list = [1 for _ in range(edge_env1.history_len * edge_env1.n_nodes)]
    assert initial_obs['drone_0'][
           -edge_env1.history_len * edge_env1.n_nodes:] == actions_history_list
    # future occup part
    future_occup_list = [0 for _ in range(edge_env1.occup_len * 3)]
    assert initial_obs['drone_0'][
           -(edge_env1.history_len * edge_env1.n_nodes + edge_env1.occup_len * 3)
           : - edge_env1.history_len * edge_env1.n_nodes] == future_occup_list


def test_find_max_usage_time(edge_env):
    node_id = 0
    resource_capacity = edge_env.resource_capacity_dict[node_id]
    print(f"resource_capacity = {resource_capacity}")
    current_task = edge_env.current_task
    print("Current task type")
    print(current_task)
    print("Result:", edge_env.find_max_usage_time(0))
    # case two: not enough resource from some time
    edge_env.resource_capacity_dict[node_id][0] = 1
    print(f"resource_capacity = {resource_capacity}")
    result = edge_env.find_max_usage_time(0)
    print("Result:", edge_env.find_max_usage_time(0))
    assert 3 == result[0]
    assert 1 == result[1]
    # not enough resource
    edge_env.resource_capacity_dict[node_id][8] = 1
    edge_env.resource_capacity_dict[node_id][5] = 1
    print(f"resource_capacity = {resource_capacity}")
    result = edge_env.find_max_usage_time(0)
    print("Result:", edge_env.find_max_usage_time(0))
    assert 1 == result[0]
    assert 3 == result[1]
    # case one: not enough resource
    for resource_type in ["CPU", "RAM", "storage"]:
        resource_demand = edge_env.current_task[resource_type]
        edge_env.current_task[resource_type] = 51
        result = edge_env.find_max_usage_time(0)
        edge_env.current_task[resource_type] = resource_demand
        assert 0 == result[0]


# @pytest.mark.skip(reason="Have not yet implemented the test")
def test_step(edge_env):
    # before step
    print("POI")
    print(f"current_task_id: {edge_env.current_task_id}")

    # after step
    actions = {0: 0.3, 1: 0.2, 2: 0.4}
    obs, rewards, dones, infos = edge_env.step(actions)
    print(f"current_task_id: {edge_env.current_task_id}")
    print(f"winner_id = {edge_env.winner_id}")
    assert edge_env.winner_id == 1
    print(f"allocation_scheme:\n {edge_env.allocation_scheme}")
    print(f"sw increase = {edge_env.sw_increase}")
    winner_cost = 0
    winner_index = edge_env.winner_id
    winner_usage_time = edge_env.winner_usage_time
    for resource_type in ["CPU", "RAM", "storage"]:
        winner_cost += (
                edge_env.df_tasks.loc[edge_env.current_task_id - 1, resource_type] *
                edge_env.df_nodes.loc[
                    winner_index, f"{resource_type}_cost"] * winner_usage_time)
    expected_sw_increase = (edge_env.df_tasks.loc[
        edge_env.current_task_id - 1, "valuation_coefficient"]) * winner_usage_time - winner_cost
    assert edge_env.sw_increase == expected_sw_increase
    print(f"rewards: {rewards}")
    assert rewards[f'drone_{winner_index}'] == expected_sw_increase
    print(f"winner's future occupancy = {edge_env.future_occup[winner_index]}")
    # check resource occupancy update
    for i, resource_type in enumerate(["CPU", "RAM", "storage"]):
        for time_step in range(edge_env.winner_start_time, edge_env.winner_finish_time):
            assert (edge_env.future_occup[winner_index][i][time_step] ==
                    pytest.approx(edge_env.df_tasks.loc[
                                      edge_env.current_task_id - 1, resource_type] /
                                  edge_env.df_nodes.loc[
                                      winner_index, f"{resource_type}"]))
    print(f"next observation is {obs}")
    # TODO: task info
    assert obs['drone_0'][0] == edge_env.df_tasks.loc[
        edge_env.current_task_id, "valuation_coefficient"]
    assert obs['drone_0'][2] == edge_env.df_tasks.loc[edge_env.current_task_id, "CPU"]
    # TODO: future occupancy
    print(f"expected future occupancy:")
    print(edge_env.future_occup)
    assert np.array(obs['drone_1'][7]) == edge_env.future_occup[1][0][0]
    # TODO: action history
    assert obs['drone_0'][-1] == actions[2]
    assert obs['drone_0'][-8] == actions[1]
    assert obs['drone_0'][-15] == actions[0]


def test_step1(edge_env1):
    # before step
    print("POI")
    print(f"current_task_id: {edge_env1.current_task_id}")

    # after step
    actions = {0: 0.3, 1: 0.2, 2: 0.4}
    obs, rewards, dones, infos = edge_env1.step(actions)
    print(f"current_task_id: {edge_env1.current_task_id}")
    print(f"winner_id = {edge_env1.winner_id}")
    assert edge_env1.winner_id == 1
    print(f"allocation_scheme:\n {edge_env1.allocation_scheme}")
    print(f"sw increase = {edge_env1.sw_increase}")
    winner_cost = 0
    winner_index = edge_env1.winner_id
    winner_usage_time = edge_env1.winner_usage_time
    for resource_type in ["CPU", "RAM", "storage"]:
        winner_cost += (
                edge_env1.df_tasks.loc[edge_env1.current_task_id - 1, resource_type] *
                edge_env1.df_nodes.loc[
                    winner_index, f"{resource_type}_cost"] * winner_usage_time)
    expected_sw_increase = (edge_env1.df_tasks.loc[
        edge_env1.current_task_id - 1, "valuation_coefficient"]) * winner_usage_time - winner_cost
    assert edge_env1.sw_increase == expected_sw_increase
    print(f"rewards: {rewards}")
    assert rewards[f'drone_{winner_index}'] == expected_sw_increase
    print(f"winner's future occupancy = {edge_env1.future_occup[winner_index]}")
    # check resource occupancy update
    for i, resource_type in enumerate(["CPU", "RAM", "storage"]):
        for time_step in range(edge_env1.winner_start_time, edge_env1.winner_finish_time):
            assert (edge_env1.future_occup[winner_index][i][time_step] ==
                    pytest.approx(edge_env1.df_tasks.loc[
                                      edge_env1.current_task_id - 1, resource_type] /
                                  edge_env1.df_nodes.loc[
                                      winner_index, f"{resource_type}"]))
    print(f"next observation is {obs}")
    # TODO: task info
    assert obs['drone_0'][0] == edge_env1.df_tasks.loc[
        edge_env1.current_task_id, "valuation_coefficient"]
    assert obs['drone_0'][2] == edge_env1.df_tasks.loc[edge_env1.current_task_id, "CPU"]
    # TODO: future occupancy
    print(f"expected future occupancy:")
    print(edge_env1.future_occup)
    assert np.array(obs['drone_1'][7]) == pytest.approx(edge_env1.future_occup[1][0][0])
    # TODO: max usage time history
    max_usage_times = [1, 1, 1]
    assert obs['drone_0'][-1] == max_usage_times[2]
    assert obs['drone_0'][-5] == max_usage_times[1]
    assert obs['drone_0'][-9] == max_usage_times[0]


def test_reverse_auction(edge_env):
    # return winner_index, winner_usage_time, winner_revenue, max_utility, sw_increase
    # first-price
    # TODO same price earlier start time should win
    bid_price_arr = [0, 0, 0]
    bid_usage_time_arr = [2, 2, 2]
    bid_start_time_arr = [2, 1, 2]
    winner_index, winner_usage_time, winner_revenue, max_utility, sw_increase = edge_env.reverse_auction(
        bid_price_arr, bid_usage_time_arr, bid_start_time_arr)
    print(winner_index, winner_usage_time, winner_revenue, max_utility, sw_increase)
    assert winner_index == 1
    assert winner_usage_time == 2
    winner_income = 0
    winner_cost = 0
    for resource_type in ["CPU", "RAM", "storage"]:
        winner_cost += (edge_env.df_tasks.loc[edge_env.current_task_id, resource_type] *
                        edge_env.df_nodes.loc[
                            winner_index, f"{resource_type}_cost"] * winner_usage_time)
    expected_winner_revenue = winner_income - winner_cost
    assert winner_revenue == expected_winner_revenue
    expected_max_utility = (edge_env.df_tasks.loc[
                                edge_env.current_task_id, "valuation_coefficient"] -
                            bid_price_arr[winner_index]) * winner_usage_time
    assert max_utility == expected_max_utility
    expected_sw_increase = (edge_env.df_tasks.loc[
        edge_env.current_task_id, "valuation_coefficient"]) * winner_usage_time - winner_cost
    assert sw_increase == expected_sw_increase

    # TODO more utility should win
    bid_price_arr = [1, 1, 0]
    bid_usage_time_arr = [2, 2, 2]
    bid_start_time_arr = [2, 1, 2]
    winner_index, winner_usage_time, winner_revenue, max_utility, sw_increase = edge_env.reverse_auction(
        bid_price_arr, bid_usage_time_arr, bid_start_time_arr)
    print(winner_index, winner_usage_time, winner_revenue, max_utility, sw_increase)
    assert winner_index == 2
    assert winner_usage_time == 2

    # second-price reverse auction
    # TODO more utility should win
    bid_price_arr = [1, 2, 3]
    bid_usage_time_arr = [2, 2, 2]
    bid_start_time_arr = [2, 1, 2]
    winner_index, winner_usage_time, winner_revenue, max_utility, sw_increase = edge_env.reverse_auction(
        bid_price_arr, bid_usage_time_arr, bid_start_time_arr,
        auction_type='second-price')
    print(winner_index, winner_usage_time, winner_revenue, max_utility, sw_increase)
    assert winner_index == 0
    assert winner_usage_time == 2
    second_winner_index = 1
    winner_income = bid_price_arr[second_winner_index] * winner_usage_time
    winner_cost = 0
    for resource_type in ["CPU", "RAM", "storage"]:
        winner_cost += (edge_env.df_tasks.loc[edge_env.current_task_id, resource_type] *
                        edge_env.df_nodes.loc[
                            winner_index, f"{resource_type}_cost"] * winner_usage_time)
    expected_winner_revenue = winner_income - winner_cost
    assert winner_revenue == expected_winner_revenue
