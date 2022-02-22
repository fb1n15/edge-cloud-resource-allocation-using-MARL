import os
import ray
import ray.rllib.agents.ppo as ppo
import shutil

ray.shutdown()
ray.init(ignore_reinit_error=True)

print("ray.init")

CHECKPOINT_ROOT = "tmp/ppo/taxi"
shutil.rmtree(CHECKPOINT_ROOT, ignore_errors=True, onerror=None)
print("checkpoint_root")

ray_results = os.getenv("HOME") + "/ray_results/"
shutil.rmtree(ray_results, ignore_errors=True, onerror=None)
print("ray_results")

SELECT_ENV = "Taxi-v3"

config = ppo.DEFAULT_CONFIG.copy()
config["log_level"] = "WARN"

agent = ppo.PPOTrainer(config, env=SELECT_ENV)
print("PPO Trainer")

N_ITER = 30
s = "{:3d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:6.2f} saved {}"

for n in range(N_ITER):
    print(f"iteration {n}:")
    result = agent.train()
    file_name = agent.save(CHECKPOINT_ROOT)

    print(s.format(
        n + 1,
        result["episode_reward_min"],
        result["episode_reward_mean"],
        result["episode_reward_max"],
        result["episode_len_mean"],
        file_name
        ))
