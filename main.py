!pip install stable-baselines3[extra]
!pip install gymnasium numpy


import numpy as np
import gymnasium as gym
from gymnasium import spaces

class SimpleFreewayEnv(gym.Env):
    def __init__(self, width=12, height=12, render_mode=None, difficulty=5):
        self.width = width
        self.height = height
        self.difficulty = np.clip(difficulty, 1, 10)
        self.spawn_chance = 0.1 + 0.1 * self.difficulty
        self.render_mode = render_mode

        self.observation_space = spaces.Box(low=0, high=2, shape=(height, width), dtype=np.float32)
        self.action_space = spaces.Discrete(3)

        self.score = 0 
        self.reset()

    def reset(self, seed=None, options=None):
        self.agent_pos = [self.height - 1, self.width // 2]
        self.cars = []
        self._spawn_cars()
        self.step_count = 0
        return self._get_obs(), {}

    def _spawn_cars(self):
        self.cars = []
        for row in range(1, self.height - 1):
            if np.random.rand() < self.spawn_chance:
                direction = 1 if row % 2 == 0 else -1
                col = 0 if direction == 1 else self.width - 1
                self.cars.append({'pos': [row, col], 'dir': direction})

    def _maybe_spawn_car(self):
        row = np.random.randint(1, self.height - 1)
        direction = 1 if row % 2 == 0 else -1
        col = 0 if direction == 1 else self.width - 1
        if np.random.rand() < self.spawn_chance:
            self.cars.append({'pos': [row, col], 'dir': direction})

    def step(self, action):
        old_row = self.agent_pos[0]

        if action == 1:
            self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
        elif action == 2:
            self.agent_pos[0] = min(self.height - 1, self.agent_pos[0] + 1)

        for car in self.cars:
            car['pos'][1] += car['dir']
        self.cars = [car for car in self.cars if 0 <= car['pos'][1] < self.width]

        for car in self.cars:
            if car['pos'] == self.agent_pos:
                return self._get_obs(), -1.0, True, False, {}

        if self.agent_pos[0] == 0:
            self.score += 1
            return self._get_obs(), 1.0, True, False, {}

        self._maybe_spawn_car()

        reward = 0.0
        if self.agent_pos[0] < old_row:
            reward += 0.25
        else:
            reward -= 0.1

        return self._get_obs(), reward, False, False, {}

    def _get_obs(self):
        grid = np.zeros((self.height, self.width), dtype=np.float32)
        grid[self.agent_pos[0], self.agent_pos[1]] = 1.0
        for car in self.cars:
            r, c = car['pos']
            if 0 <= c < self.width:
                grid[r, c] = 2.0
        return grid

    def render(self):
        grid = self._get_obs()
        symbols = {0.0: '⬛', 1.0: '🟢', 2.0: '🚗'}
        cell_width = 2
        top_border = '🟦' * ((self.width * cell_width) + 2)
        print(top_border)
        for row_idx, row in enumerate(grid):
            line = ''
            for val in row:
                symbol = symbols.get(val, '❓')
                line += symbol + ' '
            if row_idx == 0:
                print('🏁 ' + line + '🏁')
            elif row_idx == self.height - 1:
                print('🟫 ' + line + '🟫')
            else:
                print('⬛ ' + line + '⬛')
        print(top_border)
        print(f"🏆 Score: {self.score}   🚦 Difficulty: {self.difficulty}/10\n")

    def close(self):
        pass




from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv


def make_env():
    return SimpleFreewayEnv(difficulty=6)  

env = DummyVecEnv([make_env])


model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=200_000) 
model.save("ppo_simple_freeway")




env = SimpleFreewayEnv(render_mode='human', difficulty=5)
import time
obs, _ = env.reset()
total_score = 0

model = PPO.load("ppo_simple_freeway")

env = SimpleFreewayEnv(render_mode='human', difficulty=1)

while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = env.step(action)
    env.render()
    print(f"🎯 Reward this step: {reward}\n")
    time.sleep(0.1)

    if done:
        if reward == 1.0:
            total_score += 1
            print(f"✅ Success! Total Wins: {total_score}\n")
            if total_score >= 5:
                print("🎉 Max Score Achieved! Exiting...")
                break
            obs, _ = env.reset()
        else:
            print("💥 Crashed! Exiting...")
            break


# human mode 
# PLAY AS HUMAN!!!
# env = SimpleFreewayEnv( difficulty=10)
# obs, _ = env.reset()
# env.render()

# done = False
# while not done:

#     action = input("Enter action (0 = stay, 1 = up, 2 = down): ")
#     try:
#         action = int(action)
#         obs, reward, done, truncated, info = env.step(action)
#         env.render()
#         print(f"Reward: {reward}")
#     except:
#         print("Invalid input. Please enter 0, 1, or 2.")

