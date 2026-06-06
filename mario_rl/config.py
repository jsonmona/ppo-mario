from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class EvalConfig:
    """평가 실행 조건.

    Attributes:
        env_id: 평가에 사용할 실제 gym-super-mario-bros 환경 ID.
        eval_seeds: 반복 평가에 사용할 seed 목록.
        max_steps_per_episode: 한 에피소드 최대 스텝(무한 루프 방지).
        frame_skip: 프레임 스킵(행동 반복) 수.
        frame_stack: 누적할 프레임 수(시간 정보 제공).
        resize: 관측 이미지 리사이즈 크기 (H, W).
        grayscale: 흑백 변환 여부.
        time_limit_seconds: 에이전트 1회 평가의 wall-clock 제한(초). 초과 시 중단.
        video_fps: 평가 영상 저장 시 사용할 FPS.
    """

    env_id: str = "SuperMarioBros-1-1-v0"
    eval_seeds: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    max_steps_per_episode: int = 4000
    frame_skip: int = 4
    frame_stack: int = 4
    resize: tuple = (84, 84)
    grayscale: bool = True
    time_limit_seconds: float = 600.0
    video_fps: int = 15


DEFAULT_EVAL_CONFIG = EvalConfig()
