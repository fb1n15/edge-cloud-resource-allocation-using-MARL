"""
Run this with marl.py <train/run/mapgen>
-r/--restore | restore training from checkpoint
"""
import argparse
from abc import ABC, abstractmethod
import sys
from getopt import getopt
import random

from common.checkpoint_handler import explore_checkpoints
from common.config_file_handler import load_yaml


class ExperimentChooser(ABC):
    def __init__(self, _experiments):
        self.experiments = _experiments

    @abstractmethod
    def select_experiment(self):
        pass


class CLIPromptExperimentChooser(ExperimentChooser):

    @staticmethod
    def _experiment_to_str(experiment):
        checkpoint_scores = " ".join(
            f"{checkpoint['episode_reward_mean']}" for trial in experiment["trials"] for
            checkpoint in trial["checkpoints"])
        return f"{experiment['name']} | best trial = {experiment['best trial']['episode_reward_mean']} | {experiment['environment']} | {checkpoint_scores}"

    def _ask_input_int(self):
        while True:
            try:
                response = int(input(
                    f"\nEnter experiment number (1-{len(self.experiments)}) >> ")) - 1
                return response
            except ValueError:
                print("invalid integer, please try again")

    def select_experiment(self):
        """
        Prompt the user to select which experiment to run
        :return: Name of the experiment the user picked
        """

        for i, e in enumerate(self.experiments):
            print(f"{i + 1}: {self._experiment_to_str(e)}")

        choice = self._ask_input_int()
        while not 0 <= choice < len(self.experiments):
            print("Value not in valid range, ", choice)
            choice = self._ask_input_int()

        return self.experiments[choice]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_option', choices=['train', 'run', 'mapgen'])
    # parser.add_argument('--restore', action='store_true')
    parser.add_argument('--config', type=str, help='File containing the config')
    parser.add_argument('--checkpoint', type=str, help='Path to the checkpoint to run')
    args = parser.parse_args()  # get all arguments

    restore = False
    config = load_yaml(args.config)  # the dictionary of configurations
    print(f"The env seed = {config['env-config']['seed']}")
    config['env-config']['seed'] = random.randint(0, 1000)
    print(f"The new random env seed = {config['env-config']['seed']}")

    # print(f"The trainer seed = {config['trainer-config']['seed']}")
    # config['trainer-config']['seed'] = random.randint(0, 1000)
    # print(f"The new random trainer seed = {config['trainer-config']['seed']}")

    # train the model
    if args.run_option == "train":
        from learning import training
        training.main(config)

    # execute the model
    elif args.run_option == "run":
        from visualisation import run_model
        if restore:
            raise Exception("Cannot restore for run, only train")
        # experiments = explore_checkpoints()
        # chooser = CLIPromptExperimentChooser(experiments)
        # choice = chooser.select_experiment()

        run_model.main(args.checkpoint, config)

    elif args.run_option == "mapgen":
        if restore:
            raise Exception("Cannot restore for run, only train")
        from visualisation import mapgen
        mapgen.main(config)


if __name__ == "__main__":
    main()
