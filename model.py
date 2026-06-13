import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()

        self.vision = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.rnn = nn.GRUCell(3136, 256)

        torch.nn.init.orthogonal_(self.rnn.weight_ih, 1.0)
        torch.nn.init.orthogonal_(self.rnn.weight_hh, 1.0)

    @classmethod
    def new_state(cls, batch_size: int) -> Tensor:
        return torch.zeros((batch_size, 256), dtype=torch.float32)

    def forward_multi_step(
        self,
        state: Tensor,
        obs: Tensor,
        dones: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        n_seq = obs.shape[0]
        n_batch = obs.shape[1]

        x = obs.reshape(-1, *obs.shape[2:])
        x = x.to(torch.float32) * (1 / 127.5) - 1
        x = self.vision(x)
        x = x.reshape(n_seq, n_batch, -1)

        ys = []

        for t in range(n_seq):
            state = self.rnn(x[t], state)
            ys.append(state.clone())

            if dones is not None:
                state = state.masked_fill(dones[t].unsqueeze(-1), 0)

        y = torch.stack(ys)
        assert y.shape == (n_seq, n_batch, 256)

        return state, y

    def forward_single_step(
        self,
        state: Tensor,
        obs: Tensor,
        done: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        obs = torch.unsqueeze(obs, 0)

        if done is not None:
            done = torch.unsqueeze(done, 0)

        next_state, x = self.forward_multi_step(state, obs, done)

        return next_state, x[0]


class Actor(nn.Module):
    def __init__(self, n_actions: int):
        super().__init__()

        self.backbone = Backbone()
        self.action = nn.Linear(256, n_actions)
        self.value_ext = nn.Linear(256, 1)

        torch.nn.init.orthogonal_(self.action.weight, 0.01)
        torch.nn.init.orthogonal_(self.value_ext.weight, 1.0)

    def forward_multi_step(self, state, obs, dones):
        n_seq = obs.shape[0]

        next_state, latent = self.backbone.forward_multi_step(state, obs, dones)

        fold_latent = latent.reshape(-1, 256)

        fold_action = self.action(fold_latent)
        fold_value_ext = self.value_ext(fold_latent)[..., 0]

        action = fold_action.reshape(n_seq, -1, fold_action.shape[-1])
        value_ext = torch.reshape(fold_value_ext, (n_seq, -1))

        return next_state, action, value_ext

    def forward_single_step(self, state, obs, done=None):
        next_state, latent = self.backbone.forward_single_step(state, obs, done)

        action_logit = self.action(latent)
        value_ext = self.value_ext(latent)[..., 0]

        return next_state, action_logit, value_ext


class Critic(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = Backbone()
        self.value_ext = nn.Linear(256, 1)

    def forward_multi_step(self, state, obs, dones):
        n_seq = obs.shape[0]

        next_state, latent = self.backbone.forward_multi_step(state, obs, dones)

        fold_latent = latent.reshape(-1, 256)

        fold_value_ext = self.value_ext(fold_latent)[..., 0]

        value_ext = torch.reshape(fold_value_ext, (n_seq, -1))

        return next_state, value_ext

    def forward_single_step(self, state, obs, done=None):
        next_state, latent = self.backbone.forward_single_step(state, obs, done)

        value_ext = self.value_ext(latent)[..., 0]

        return next_state, value_ext
