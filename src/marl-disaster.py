"""
Run this with marl-disaster.py <train/run/mapgen>
-r/--restore | restore training from checkpoint

"""
from abc import ABC, abstractmethod
import sys
from getopt import getopt


from common.checkpoint_handler import explore_checkpoints


class ExperimentChooser(ABC):
    def __init__(self, _experiments):
        self.experiments = _experiments

    @abstractmethod
    def select_experiment(self):
        pass


class CLIPromptExperimentChooser(ExperimentChooser):

    @staticmethod
    def _experiment_to_str(experiment):
        checkpoint_scores = " ".join(f"{checkpoint['episode_reward_mean']}" for trial in experiment["trials"] for checkpoint in trial["checkpoints"])
        return f"{experiment['name']} | best trial = {experiment['best trial']['episode_reward_mean']} | {experiment['environment']} | {checkpoint_scores}"

    def _ask_input_int(self):
        while True:
            try:
                response = int(input(f"\nEnter experiment number (1-{len(self.experiments)}) >> ")) - 1
                return response
            except ValueError:
                print("invalid integer, please try again")

    def select_experiment(self):
        """
        Prompt the user to select which experiment to run
        :return: Name of the experiment the user picked
        """

        for i, e in enumerate(self.experiments):
            print(f"{i+1}: {self._experiment_to_str(e)}")

        choice = self._ask_input_int()
        while not 0 <= choice < len(self.experiments):
            print("Value not in valid range, ", choice)
            choice = self._ask_input_int()

        return self.experiments[choice]


def main(argv):
    if len(argv) <= 1:
        raise Exception("Must specify train/run")
    opts, args = getopt(argv[2:], "r", ["restore="])
    restore = None
    for opt, arg in opts:
        if opt in ("-r", "--restore"):
            restore = arg

    if argv[1] == "train":
        from learning import training
        training.main(restore=restore)

    elif argv[1] == "run":
        from visualisation import run_model
        if restore is not None:
            raise Exception("Cannot restore for run, only train")
        experiments = explore_checkpoints()
        chooser = CLIPromptExperimentChooser(experiments)
        run_model.main(chooser.select_experiment())

    elif argv[1] == "mapgen":
        if restore is not None:
            raise Exception("Cannot restore for run, only train")
        from visualisation import mapgen
        mapgen.main()


if __name__ == "__main__":
    main(sys.argv)
