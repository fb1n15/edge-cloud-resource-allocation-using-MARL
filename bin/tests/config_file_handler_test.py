from ray import tune

from common.config_file_handler import do_tune_wrap


def test_config_wrapper():
    assert {} == do_tune_wrap({})
    assert {"lr": 0.2} == do_tune_wrap({"lr": 0.2})
    assert type(do_tune_wrap({"lr": {"uniform": [0, 1]}})["lr"]) == tune.sample.Float
    assert type(do_tune_wrap({"lr": {"loguniform": [0.0001, 0.01]}})["lr"]) == tune.sample.Float
    assert type(do_tune_wrap({"lr": {"choice": [0, 1]}})["lr"]) == tune.sample.Categorical
    assert type(do_tune_wrap({"mutations": {"lr": {"loguniform": [0.0001, 0.01]}}})["mutations"]["lr"]) == tune.sample.Float
