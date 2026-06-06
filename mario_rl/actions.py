"""
Canonical action definition for the Super Mario project.

Every student model and every evaluator script must use this order.  The order
matches gym_super_mario_bros.actions.SIMPLE_MOVEMENT.
"""

from __future__ import annotations

ACTION_MEANINGS = (
    "NOOP",
    "RIGHT",
    "RIGHT_JUMP",
    "RIGHT_RUN",
    "RIGHT_RUN_JUMP",
    "JUMP",
    "LEFT",
)

MARIO_MOVEMENT = [
    ["NOOP"],
    ["right"],
    ["right", "A"],
    ["right", "B"],
    ["right", "A", "B"],
    ["A"],
    ["left"],
]


def describe_actions() -> dict[int, str]:
    return {i: name for i, name in enumerate(ACTION_MEANINGS)}

