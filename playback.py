import cv2
import av
import torch
import gymnasium as gym
import numpy as np

from typing import Any
from mario_rl.env import (
    _make_super_mario_bros,
    MaxAndSkipWrapper,
    GrayResizeWrapper,
    FrameStackWrapper,
    DEFAULT_EVAL_CONFIG,
)
from agent import Agent

MODEL_PATH = "runs/20260607_165945/804115.pth"


class VideoRecordingWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, filepath: str = "playback.mkv", fps: int = 60):
        super().__init__(env)
        self.filepath = filepath
        self.fps = fps
        self.container = None
        self.stream = None

    def step(self, action: int):
        obs, reward, terminated, truncated, info = super().step(action)
        self._record_frame(obs)
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs: Any):
        obs, info = super().reset(**kwargs)
        self._record_frame(obs)
        return obs, info

    def _record_frame(self, obs: np.ndarray) -> None:
        H, W, _ = obs.shape

        obs = cv2.resize(obs, (W * 2, H * 2), interpolation=cv2.INTER_NEAREST)
        H, W, _ = obs.shape

        if self.container is None:
            self.container = av.open(self.filepath, mode="w")
            self.stream = self.container.add_stream(
                "libx264", options={"crf": "10", "preset": "ultrafast"}, rate=self.fps
            )
            self.stream.width = W
            self.stream.height = H
            self.stream.pix_fmt = "yuv420p"

        if self.stream is None:
            raise ValueError("stream must been initialized by now")

        frame = av.VideoFrame.from_ndarray(obs, "rgb24")
        self.container.mux(self.stream.encode(frame))

    def close(self) -> None:
        if self.container is not None:
            assert self.stream is not None

            self.container.mux(self.stream.encode())
            self.container.close()

        super().close()


def main() -> None:
    config = DEFAULT_EVAL_CONFIG

    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = None

    for env_id in config.eval_env_ids:
        stage_name = env_id[15:-3]
        env = _make_super_mario_bros(env_id)
        env = VideoRecordingWrapper(env, filepath=f"playback-{stage_name}.mp4", fps=60)
        env = MaxAndSkipWrapper(env, skip=config.frame_skip)
        env = GrayResizeWrapper(env, out_hw=config.resize, grayscale=config.grayscale)
        env = FrameStackWrapper(env, k=config.frame_stack)
        env = gym.wrappers.TimeLimit(env, max_episode_steps=config.max_steps_per_episode)

        # Load agent
        if agent is None:
            agent = Agent.load(MODEL_PATH, env.observation_space, env.action_space, device=device)

        agent.reset()

        obs, info = env.reset()
        done = False

        steps = 0
        returns = 0.0

        while not done:
            action = agent.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

            steps += 1
            returns += float(reward)

        env.close()

        print()
        print(f"Env: {env_id}")
        print(f"Total steps: {steps}")
        print(f"Total return: {returns:.1f}")


if __name__ == "__main__":
    main()
