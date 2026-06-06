# flake8: noqa: E402

import os
import AutoROM


# Hack due to UV package manager
autorom_dir = os.path.dirname(AutoROM.__file__)
rom_dir = os.path.join(autorom_dir, "roms")

os.environ["ALE_ROMS_DIR"] = rom_dir


import gymnasium as gym
from beartype import beartype
from ale_py.vector_env import AtariVectorEnv


@beartype
def make_env() -> gym.Env:
    return gym.wrappers.FrameStackObservation(
        gym.wrappers.AtariPreprocessing(
            gym.make("ALE/MontezumaRevenge-v5", frameskip=1),
            frame_skip=4,
            grayscale_obs=True,
            screen_size=(160, 192),
        ),
        stack_size=1,
        padding_type="zero",
    )


@beartype
def make_vector_env(num_envs: int) -> AtariVectorEnv:
    return AtariVectorEnv(
        "montezuma_revenge",
        num_envs=num_envs,
        frameskip=4,
        grayscale=True,
        stack_num=1,
        img_height=64,
        img_width=64,
    )
