from pprint import pprint

from common.config_file_handler import load_yaml
from environments.edge_cloud.simulation.environment import EdgeCloudEnv
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
        '/Users/fan/OneDrive - University of Southampton/My-Projects/MARL-Jack/marl-disaster-relief/configs/experiments/edge_cloud/hyperparameters/cpu_ppo_fc_independent_lr_with_history.yaml')
    env_config = config['env-config']
    edge_env = EdgeCloudEnv(env_config)
    edge_env.reset()
    return edge_env


def test_reset(edge_env):
    initial_obs = edge_env.reset()
    first_task = edge_env.df_tasks.iloc[0]
    # print(initial_obs)
    # print(first_task)
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
    # case one: not enough resource
    edge_env.current_task["CPU"] = 51
    result = edge_env.find_max_usage_time(0)
    assert 0 == result[0]
    # case one: not enough resource
    edge_env.current_task["RAM"] = 51
    result = edge_env.find_max_usage_time(0)
    assert 0 == result[0]



@pytest.mark.skip(reason="Have not yet implemented the test")
def test_step():
    assert False


@pytest.mark.skip(reason="Have not yet implemented the test")
def test_reverse_auction():
    assert False
