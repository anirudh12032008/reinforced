# train_agent.py
from stable_baselines3 import PPO
import pufferlib
import pufferlib.emulation
from pufferlib.ocean import freeway

# Use freeway's built-in environment
env_fn = lambda: freeway.make_env(num_envs=1, level=0)
env = pufferlib.emulation.GymnasiumPufferEnv(env_fn)

# Train PPO
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)
model.save("ppo_freeway")
