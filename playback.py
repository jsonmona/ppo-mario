import os
import sys
from typing import Any

import av
import gymnasium as gym
import numpy as np

from mario_rl.env import (
    _make_super_mario_bros,
    MaxAndSkipWrapper,
    GrayResizeWrapper,
    FrameStackWrapper,
    DEFAULT_EVAL_CONFIG,
)
from agent import Agent

MODEL_PATH = "runs/20260606_235308/4682.pth"

class VideoRecordingWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, filepath: str = "playback.mkv", fps: int = 60):
        super().__init__(env)
        self.filepath = filepath
        self.container = av.open(self.filepath, mode="w", format="matroska")
        
        # Determine height and width
        dummy_render = self.env.render()
        if isinstance(dummy_render, np.ndarray):
            h, w, _ = dummy_render.shape
        else:
            h, w = 240, 256
            
        self.stream = self.container.add_stream(
            "libx264", 
            options={"crf": "10", "preset": "ultrafast"}, 
            rate=fps
        )
        self.stream.width = w
        self.stream.height = h
        self.stream.pix_fmt = "yuv420p"
        
    def step(self, action: int):
        obs, reward, terminated, truncated, info = super().step(action)
        self._record_frame()
        return obs, reward, terminated, truncated, info
        
    def reset(self, **kwargs: Any):
        obs, info = super().reset(**kwargs)
        self._record_frame()
        return obs, info
        
    def _record_frame(self) -> None:
        frame_data = self.env.render()
        if isinstance(frame_data, np.ndarray):
            frame = av.VideoFrame.from_ndarray(frame_data, format="rgb24")
            for packet in self.stream.encode(frame):
                self.container.mux(packet)
                
    def close(self) -> None:
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()
        super().close()

def main() -> None:
    config = DEFAULT_EVAL_CONFIG
    
    # Create the environment matching training exactly, 
    # but inserting the VideoRecordingWrapper right after the base env
    env = _make_super_mario_bros(config.env_id)
    env = VideoRecordingWrapper(env, filepath="playback.mkv", fps=60)
    env = MaxAndSkipWrapper(env, skip=config.frame_skip)
    env = GrayResizeWrapper(env, out_hw=config.resize, grayscale=config.grayscale)
    env = FrameStackWrapper(env, k=config.frame_stack)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=config.max_steps_per_episode)
    
    # Load agent
    agent = Agent.load(MODEL_PATH, env.observation_space, env.action_space, device="cpu")
    agent.reset()
    
    obs, info = env.reset()
    done = False
    
    while not done:
        action = agent.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        
    env.close()

if __name__ == "__main__":
    main()
