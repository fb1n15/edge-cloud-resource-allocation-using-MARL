import ray
from ray import tune

ray.init()

tune.run("PPO",
         config={"env": "CartPole-v1",
                 "framework": "torch",
                 "evaluation_interval": 2,  # Evaluate every 2 episodes
                 "evaluation_duration": 2,  # Number of episodes to evaluate
                 "num_workers": 6,
                 "num_envs_per_worker": 4,
                 "num_gpus": 0
                 },
        reuse_actors=True
         )
