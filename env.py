import numpy as np
import gymnasium as gym
from typing import Any, SupportsFloat


class PowerupRewardWrapper(gym.vector.VectorWrapper):
    """파워업 아이템을 먹으면 리워드 제공"""

    def __init__(self, env: gym.vector.VectorEnv, coeff: SupportsFloat):
        super().__init__(env)
        self.coeff = float(coeff)
        self.status_to_number = {None: 0, "small": 0, "tall": 1, "fireball": 2}
        self.prev_powerup = np.zeros(env.num_envs, dtype=np.int8)

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        observations, rewards, terminations, truncations, infos = self.env.step(actions)

        curr_powerup = np.asarray([self.status_to_number[x] for x in infos["status"]], dtype=np.int8)
        extra_reward = np.sign(curr_powerup - self.prev_powerup) * self.coeff

        self.prev_powerup = curr_powerup
        rewards = rewards + extra_reward

        return observations, rewards, terminations, truncations, infos


def make_vector_env(n_envs: int):
    from mario_rl.env import make_env

    env = gym.vector.AsyncVectorEnv([lambda: make_env() for _ in range(n_envs)])
    env = PowerupRewardWrapper(env, coeff=1)
    return env
