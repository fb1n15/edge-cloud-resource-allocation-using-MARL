from simulation.gridworld import GridWorldModel
from simulation.agent import Agent


def main():
    gridworld = GridWorldModel(20, 20, 5, [Agent(10, 10, rot=0)])


if __name__ == "__main__":
    main()