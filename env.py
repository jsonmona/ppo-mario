import numpy as np
import gymnasium as gym
from typing import Any, SupportsFloat


class BlockRewardWrapper(gym.vector.VectorWrapper):
    """각 환경별로 첫 N개의 에피소드동안 리워드를 막음"""

    def __init__(self, env: gym.vector.VectorEnv, episodes: int):
        super().__init__(env)
        self.episodes = episodes
        self.disabled = False
        self.reset_cnt = np.zeros(env.num_envs, dtype=np.int32)

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        if self.disabled:
            return self.env.step(actions)

        observations, rewards, terminations, truncations, infos = self.env.step(actions)

        done = np.logical_or(terminations, truncations)
        self.reset_cnt[done] += 1

        block_reward = self.reset_cnt < self.episodes
        rewards[block_reward] = 0

        if not np.any(block_reward):
            self.disabled = True

        return observations, rewards, terminations, truncations, infos


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
    env = BlockRewardWrapper(env, episodes=1000)
    env = PowerupRewardWrapper(env, coeff=100)
    return env
