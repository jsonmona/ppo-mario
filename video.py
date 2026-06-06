import os
import numpy as np
import av

from abc import ABC, abstractmethod
from typing import Callable
from fractions import Fraction
from beartype import beartype


OPTIONS = {
    "crf": "10",
    "preset": "ultrafast",
}


class VideoWriter(ABC):
    @abstractmethod
    def step(self, obs: np.ndarray | Callable[[], np.ndarray], done: bool, global_step: int) -> str | None:
        pass


class DummyVideoWriter(VideoWriter):
    def __init__(self, *args, **kwargs):
        pass

    def step(self, obs: np.ndarray | Callable[[], np.ndarray], done: bool, global_step: int) -> str | None:
        pass


@beartype
class PeriodicVideoWriter(VideoWriter):
    def __init__(self, run_dir: str, rate: float, period: int = 100):
        self.run_dir = run_dir
        self.period = period
        self.rate = Fraction(rate)
        self.wait = 0
        self.container = None
        self.encoder = None

    def step(self, obs: np.ndarray | Callable[[], np.ndarray], done: bool, global_step: int) -> str | None:
        if self.container is None:
            if not done:
                return

            self.wait -= 1
            if 0 < self.wait:
                return

            if callable(obs):
                obs = obs()

            if len(obs.shape) == 2:
                h, w = obs.shape
            elif len(obs.shape) == 3:
                _, h, w = obs.shape

            video_path = os.path.join(self.run_dir, f"{global_step}.mkv")

            self.wait = self.period
            self.container = av.open(video_path, "w", "matroska")
            self.encoder = self.container.add_stream(
                "libx264", options=dict(OPTIONS), rate=self.rate, height=h, width=w
            )

            # Start writing at next frame
            return video_path
        else:
            assert self.encoder is not None

        if callable(obs):
            obs = obs()

        if len(obs.shape) == 2:
            obs = np.expand_dims(obs, 0)
        if obs.shape[0] == 1:
            obs = np.tile(obs, [3, 1, 1])

        obs = np.moveaxis(obs, 0, -1)

        frame = av.VideoFrame.from_ndarray(obs, "rgb24")
        self.container.mux(self.encoder.encode(frame))

        if done:
            self.container.close()
            self.encoder = None
            self.container = None


@beartype
def create_videowriter(run_dir: str, rate: float, period: int = 100, disabled: bool = False) -> VideoWriter:
    if disabled:
        return DummyVideoWriter()

    return PeriodicVideoWriter(run_dir, rate, period)
