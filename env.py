import gymnasium as gym


def make_vector_env(n_envs: int):
    from mario_rl.env import make_env

    env = gym.vector.AsyncVectorEnv([lambda: make_env() for _ in range(n_envs)])
    return env
