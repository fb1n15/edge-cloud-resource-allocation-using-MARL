import argparse

from simulation.environment import GridWorldEnv

from ray import tune
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.framework import try_import_tf, try_import_torch

tf1, tf, tfv = try_import_tf()

parser = argparse.ArgumentParser()
parser.add_argument("--torch", action="store_true")
parser.add_argument("--as-test", action="store_true")
parser.add_argument("--stop-iters", type=int, default=10000)
parser.add_argument("--stop-reward", type=float, default=10)
parser.add_argument("--stop-timesteps", type=int, default=1000000)


def run_same_policy(args, stop):
    """Use the same policy for both agents (trivial case)."""
    config = {
        "env": GridWorldEnv,
        "framework": "torch" if args.torch else "tf",
        # "num_gpus": 1,
        "num_workers": 9,
        "num_cpus_for_driver": 1,
        "num_cpus_per_worker": 1,
        # "lr": 0.01,
        # "model": {"fcnet_hiddens": [8, 8]},
        "env_config": {
            "width": 20,
            "height": 20,
            "num_survivors": 10,
            "num_agents": 2,
            "start_world": [[]],
            "sight": 4,
            "battery": 200,
        }
    }

    results = tune.run("PG", config=config, stop=stop, verbose=1, checkpoint_freq=100)


def main():
    args = parser.parse_args()

    stop = {
        "training_iteration": args.stop_iters,
        # "timesteps_total": args.stop_timesteps,
        "episode_reward_mean": args.stop_reward,
    }
    run_same_policy(args, stop=stop)
    print("run_same_policy: ok.")


if __name__ == "__main__":
    main()
