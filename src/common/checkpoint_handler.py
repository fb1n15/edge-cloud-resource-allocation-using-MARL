import os
import json

from simulation.environment import GridWorldEnv


def save_checkpoints(checkpoints):
    """Save checkpoints into file with some model parameter settings"""
    path = "GridworldCheckPoints/"
    if not os.path.exists(path):
        os.makedirs(path)
    with open(os.path.join(path, "checkpoints.txt"), "w+") as f:
        f.write(str(checkpoints))


def load_checkpoints():
    with open(os.path.join("GridworldCheckPoints", "checkpoints.txt"), "r") as f:
        return eval(f.readline())


def subdirs(path):
    """Yield directory names contained in directory"""
    for item in os.listdir(path):
        if not os.path.isfile(os.path.join(path, item)):
            yield item


def load_results(results_path):
    with open(results_path, "r") as f:
        for line in f.readlines():
            try:
                yield json.loads(line)
            except:
                pass
                #TODO remove


def load_params(params_path):
    with open(params_path, "r") as f:
        return json.load(f)


def explore_checkpoints():
    experiments = []
    default_path = os.path.expanduser("~/ray_results")
    for experiment in subdirs(default_path):
        if not experiment.startswith("DroneRescue"):
            continue
        trials = []
        environment = None
        valid = True
        for trial in subdirs(os.path.join(default_path, experiment)):
            checkpoints = []
            results = list(load_results(os.path.join(default_path, experiment, trial, "result.json")))
            config = load_params(os.path.join(default_path, experiment, trial, "params.json"))
            if len(results) == 0:
                break
            if config["env_config"].get("version", -1) < GridWorldEnv.VERSION:
                # If the config is using an old version of the simulation environment, don't include
                valid = False
            if environment is None:
                environment = config["env_config"]

            for i, checkpoint in enumerate(subdirs(os.path.join(default_path, experiment, trial))):
                checkpoints.append({
                    "name": checkpoint,
                    "episode_reward_mean": results[i]["episode_reward_mean"],
                    "path": os.path.join(default_path, experiment, trial, checkpoint, checkpoint.replace("_", "-"))
                })
                # Check that all the environments are the same for each checkpoint in a trial
                assert results[i]["config"]["env_config"] == environment
            if len(checkpoints) > 0:
                trials.append({
                    "name": trial,
                    "checkpoints": checkpoints,
                    "best checkpoint": max(checkpoints, key=lambda c: c["episode_reward_mean"]),
                    "config": config
                })
        print(experiment, valid)
        if valid and len(trials) > 0:
            best_trial = max(trials, key=lambda t: t["best checkpoint"]["episode_reward_mean"])
            experiments.append({
                "name": experiment,
                "trials": trials,
                "environment": environment,
                "best trial": {
                    "trial name": best_trial["name"],
                    "checkpoint name": best_trial["best checkpoint"]["name"],
                    "episode_reward_mean": best_trial["best checkpoint"]["episode_reward_mean"],
                    "path": best_trial["best checkpoint"]["path"],
                    "config": best_trial["config"],
                }
            })
    return experiments
