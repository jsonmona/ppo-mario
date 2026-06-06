import os
import sys
from typing import Optional

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from mario_rl.interface import AgentMetadata, BaseAgent
from model import Backbone, Actor


class Agent(BaseAgent):
    """Submission entry point."""

    TEAM_ID = "teamXX"
    MEMBERS = ["name1", "name2"]
    METHOD = "PPO"
    BACKBONE = "cnn+gru"

    def __init__(self, observation_space, action_space, device: str = "cpu"):
        super().__init__(observation_space, action_space, device)
        n_actions = int(action_space.n)
        self.net = Actor(n_actions).to(device)
        self.net.eval()
        self.state = Backbone.new_state(1).to(device)
        self.rng = np.random.default_rng(42)

    def reset(self) -> None:
        self.state = Backbone.new_state(1).to(self.device)
        self.rng = np.random.default_rng(42)

    @torch.no_grad()
    def act(self, observation: np.ndarray) -> int:
        obs = torch.as_tensor(np.asarray(observation), dtype=torch.uint8, device=self.device)
        if obs.ndim == 3:
            obs = obs.unsqueeze(0)

        self.state, action_logit, _, _ = self.net.forward_single_step(self.state, obs)

        probs = torch.softmax(action_logit, dim=1)[0].cpu().numpy()

        probs = probs.astype(np.float64)
        probs /= probs.sum()
        action = self.rng.choice(len(probs), p=probs)

        return int(action)

    @classmethod
    def load(cls, path, observation_space, action_space, device: str = "cpu") -> "BaseAgent":
        ckpt = torch.load(path, map_location=device)

        agent = cls(observation_space, action_space, device)
        state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        agent.net.load_state_dict(state_dict)
        agent.net.eval()
        return agent

    def metadata(self) -> Optional[AgentMetadata]:
        return AgentMetadata(
            team_id=self.TEAM_ID,
            members=self.MEMBERS,
            method=self.METHOD,
            backbone=self.BACKBONE,
        )

    def save(self, path: str) -> None:
        torch.save(
            {
                "state_dict": self.net.state_dict(),
                "method": self.METHOD,
                "team_id": self.TEAM_ID,
                "members": self.MEMBERS,
                "backbone": self.BACKBONE,
            },
            path,
        )
