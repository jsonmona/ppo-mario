import os
import time
import torch
import numpy as np
import torch.nn.functional as F
import gymnasium as gym
from typing import Optional, Tuple, Any, SupportsFloat
from torch import nn
from torch import Tensor
from torch.distributions.categorical import Categorical
from rich.progress import track
from dataclasses import dataclass
from datetime import datetime
from tensorboardX import SummaryWriter

from video import create_videowriter
from model import Backbone, Actor, Critic
from env import make_vector_env


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def symexp(x: Tensor):
    return x.sign() * torch.expm1(x.abs())


def symlog(x: Tensor):
    return x.sign() * torch.log1p(x.abs())


class LinearWarmup(torch.optim.lr_scheduler.LRScheduler):
    def __init__(
        self, optimizer: torch.optim.Optimizer, warmup_steps: int, start_lr: SupportsFloat, last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        self.start_lr = float(start_lr)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch > self.warmup_steps:
            return self.base_lrs

        return [
            self.start_lr + (base_lr - self.start_lr) * (self.last_epoch / self.warmup_steps)
            for base_lr in self.base_lrs
        ]


class RunningMeanStd:
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon: float = 1e-4, shape: tuple[int, ...] = ()):
        self.mean = np.zeros(shape, "float64")
        self.var = np.ones(shape, "float64")
        self.count = epsilon

    def update(self, x: np.ndarray):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = m_2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count


class RewardForwardFilter:
    def __init__(self, gamma: float):
        self.rewems = None
        self.gamma = gamma

    def update(self, rews: np.ndarray):
        if self.rewems is None:
            self.rewems = rews
        else:
            self.rewems = self.rewems * self.gamma + rews
        return self.rewems


@dataclass
class Rollout:
    obs: Tensor
    actions: Tensor
    logprobs: Tensor
    rewards_ext: Tensor
    dones: Tensor
    values_ext: Tensor

    @classmethod
    def new(cls, n_seq: int, n_envs: int) -> "Rollout":
        s = n_seq
        b = n_envs

        return cls(
            obs=torch.zeros((s, b, 4, 84, 84), dtype=torch.uint8),
            actions=torch.zeros((s, b), dtype=torch.int32),
            logprobs=torch.zeros((s, b), dtype=torch.float32),
            rewards_ext=torch.zeros((s, b), dtype=torch.float32),
            dones=torch.zeros((s, b), dtype=torch.bool),
            values_ext=torch.zeros((s, b), dtype=torch.float32),
        )

    def to(self, *args, **kwargs):
        return Rollout(
            obs=self.obs.to(*args, **kwargs),
            actions=self.actions.to(*args, **kwargs),
            logprobs=self.logprobs.to(*args, **kwargs),
            rewards_ext=self.rewards_ext.to(*args, **kwargs),
            dones=self.dones.to(*args, **kwargs),
            values_ext=self.values_ext.to(*args, **kwargs),
        )


def calc_action(
    logits: Tensor,
    action: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    probs = Categorical(logits=logits)
    if action is None:
        action = probs.sample()

    return action, probs.log_prob(action), probs.entropy()


def bootstrap_gae(
    values: Tensor,
    rewards: Tensor,
    dones: Tensor,
    next_value: Tensor,
    next_done: Tensor,
    gamma: float,
    gae_lambda: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    n_seq = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    lastgaelam = 0

    nextvalues = next_value

    for t in range(n_seq - 1, -1, -1):
        nextnonterminal = torch.logical_not(dones[t]).float()

        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam

        nextvalues = values[t]

    returns = advantages + values

    return advantages, returns


def train():
    torch.set_float32_matmul_precision("medium")
    torch.manual_seed(42)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    run_dir = "./runs/" + datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(run_dir, flush_secs=30)

    n_envs = 128
    n_batch_size = 128
    n_actions = 12
    n_seq = 16
    n_iterations = 300_000_000 // (n_seq * n_envs)  # 300M env steps
    clip_coef = 0.1
    ent_coef = 0.01
    max_grad_norm = 1.0

    n_policy_update_epochs = 2
    n_policy_batch_envs = 128
    n_value_update_epochs = 1
    n_value_batch_envs = 32
    n_distill_update_epochs = 2
    n_distill_batch_envs = 32

    assert n_envs % n_batch_size == 0

    env = make_vector_env(n_envs)

    assert env.single_action_space.n == n_actions  # type: ignore

    actor = Actor(n_actions).to(device)
    critic = Critic().to(device)

    # Resume with more actions
    start_iter = 96939
    old_actor = Actor(7)
    old_actor.load_state_dict(torch.load("runs/20260616_111052/1551025-96939-actor.pth"))
    actor.backbone.load_state_dict(old_actor.backbone.state_dict())
    actor.value_ext.load_state_dict(old_actor.value_ext.state_dict())
    with torch.no_grad():
        actor.action.weight[:7].copy_(old_actor.action.weight)
        actor.action.bias[:7].copy_(old_actor.action.bias)
        actor.action.bias[7:] -= 1
    critic.load_state_dict(torch.load("runs/20260616_111052/1551025-96939-critic.pth"))

    del old_actor
    actor.requires_grad_(True)
    critic.requires_grad_(True)

    writer = SummaryWriter(run_dir, flush_secs=30)

    next_obs, _ = env.reset()
    next_obs = torch.tensor(next_obs).to(device)
    next_done = torch.zeros((n_envs), dtype=torch.bool, device=device)
    next_actor_state = Backbone.new_state(n_envs).to(device)
    next_critic_state = Backbone.new_state(n_envs).to(device)

    episodic_returns = np.zeros(n_envs, dtype=np.float32)
    episodic_return_history = np.zeros(256, dtype=np.float32)
    episodic_return_history_pos = 0

    rollout = Rollout.new(n_seq, n_envs).to(device)

    actor_opt = torch.optim.AdamW(actor.parameters(), lr=0.0001, eps=1e-8, weight_decay=1e-4)
    critic_opt = torch.optim.AdamW(critic.parameters(), lr=0.0001, eps=1e-8, weight_decay=1e-4)

    actor_lr = LinearWarmup(actor_opt, 100, 1e-7)
    critic_lr = LinearWarmup(critic_opt, 100, 1e-7)

    video = create_videowriter(run_dir, 60 / 4, period=50, disabled=False)

    for iteration in track(range(start_iter, n_iterations + 1), description="Training..."):
        time_iter_start = time.monotonic()

        # Step LR scheduler
        actor_lr.step()
        critic_lr.step()

        initial_actor_state = next_actor_state.clone()
        initial_critic_state = next_critic_state.clone()

        average_reward = 0.0

        with torch.no_grad():
            for step in range(n_seq):
                next_actor_state, action_logit, _ = actor.forward_single_step(next_actor_state, next_obs, next_done)
                action, logprob, _ = calc_action(logits=action_logit)

                env.step_async(action.cpu().numpy())

                next_critic_state, cv_ext = critic.forward_single_step(next_critic_state, next_obs, next_done)

                rollout.obs[step] = next_obs
                rollout.dones[step] = next_done
                rollout.actions[step] = action
                rollout.logprobs[step] = logprob
                rollout.values_ext[step] = cv_ext

                start_new = video.step(
                    lambda: next_obs[0, -1].cpu().numpy(),
                    bool(next_done[0].item()),
                    iteration * n_seq + step,
                )

                if start_new is not None:
                    torch.save(actor.state_dict(), os.path.splitext(start_new)[0] + f"-{iteration}-actor.pth")
                    torch.save(critic.state_dict(), os.path.splitext(start_new)[0] + f"-{iteration}-critic.pth")

                next_obs, ext_reward, terminations, truncations, _ = env.step_wait()
                next_done = np.logical_or(terminations, truncations)
                average_reward += np.sum(ext_reward) / (ext_reward.size * n_seq)
                episodic_returns += ext_reward

                if np.any(next_done):
                    for x in episodic_returns[next_done]:
                        episodic_return_history[episodic_return_history_pos] = x
                        episodic_return_history_pos = (episodic_return_history_pos + 1) % len(episodic_return_history)
                    episodic_returns[next_done] = 0

                next_obs = torch.from_numpy(next_obs).to(device)
                next_done = torch.from_numpy(next_done).to(device)

                rollout.rewards_ext[step] = torch.from_numpy(ext_reward).to(device)

        with torch.no_grad():
            _, next_v_ext = critic.forward_single_step(next_critic_state, next_obs, next_done)

            _, ret_ext = bootstrap_gae(
                symexp(rollout.values_ext),
                rollout.rewards_ext,
                rollout.dones,
                symexp(next_v_ext),
                next_done,
                0.99,
                0.95,
            )

            adv_ext, _ = bootstrap_gae(
                symexp(rollout.values_ext),
                rollout.rewards_ext,
                rollout.dones,
                symexp(next_v_ext),
                next_done,
                0.99,
                0.8,
            )

            advantages = adv_ext
            adv_std, adv_mean = torch.std_mean(advantages)
            advantages = (advantages - adv_mean) / (adv_std + 1e-8)

        log_approx_kl = 0.0
        log_clipfracs = 0.0
        log_v_loss = 0.0
        log_pg_loss = 0.0
        log_entropy_loss = 0.0
        log_distill_loss = 0.0

        # Optimize policy
        for local_epoch in range(n_policy_update_epochs):
            idx_order = torch.randperm(n_envs, device=device, dtype=torch.int32)
            idx_order = idx_order.view(n_envs // n_policy_batch_envs, n_policy_batch_envs)

            for mb_idx in idx_order:
                mb_obs = rollout.obs[:, mb_idx]
                mb_logprobs = rollout.logprobs[:, mb_idx]
                mb_actions = rollout.actions[:, mb_idx]
                mb_dones = rollout.dones[:, mb_idx]
                mb_advantages = advantages[:, mb_idx]

                mb_actor_state = initial_actor_state[mb_idx]

                mb_actor_state, newlogit, _ = actor.forward_multi_step(mb_actor_state, mb_obs, mb_dones)
                _, newlogprob, newentropy = calc_action(newlogit, mb_actions)
                logratio = newlogprob - mb_logprobs
                ratio = torch.exp(logratio)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)

                mb_valid_mask = torch.logical_not(mb_dones)
                mb_valid_count = torch.clamp(mb_valid_mask.sum(), min=1.0)

                pg_loss = (torch.max(pg_loss1, pg_loss2) * mb_valid_mask).sum() / mb_valid_count
                entropy_loss = newentropy.mean()

                actor_loss = pg_loss - entropy_loss * ent_coef

                actor_opt.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
                actor_opt.step()

                # Put back the RNN state
                if local_epoch == n_policy_update_epochs - 1:
                    next_actor_state[mb_idx] = mb_actor_state.detach()

                # Logging info
                with torch.no_grad():
                    divisor = len(idx_order) * n_policy_update_epochs
                    log_approx_kl += ((ratio - 1) - logratio).mean().item() / divisor
                    log_clipfracs += ((ratio - 1.0).abs() > clip_coef).float().mean().item() / divisor
                    log_pg_loss += pg_loss.item() / divisor
                    log_entropy_loss += entropy_loss.item() / divisor

        # Optimize critic
        for local_epoch in range(n_value_update_epochs):
            idx_order = torch.randperm(n_envs, device=device, dtype=torch.int32)
            idx_order = idx_order.view(n_envs // n_value_batch_envs, n_value_batch_envs)

            for mb_idx in idx_order:
                mb_obs = rollout.obs[:, mb_idx]
                mb_dones = rollout.dones[:, mb_idx]
                mb_ret_ext = ret_ext[:, mb_idx]

                mb_critic_state = initial_critic_state[mb_idx]

                mb_critic_state, newv_ext = critic.forward_multi_step(mb_critic_state, mb_obs, mb_dones)

                v_loss_ext = F.mse_loss(newv_ext, symlog(mb_ret_ext))
                v_loss = v_loss_ext

                critic_opt.zero_grad()
                v_loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
                critic_opt.step()

                # Put back the RNN state and value
                if local_epoch == n_value_update_epochs - 1:
                    next_critic_state[mb_idx] = mb_critic_state.detach()
                    rollout.values_ext[:, mb_idx] = newv_ext.detach()

                # Logging info
                with torch.no_grad():
                    divisor = len(idx_order) * n_value_update_epochs
                    log_v_loss += v_loss.item() / divisor

        # Optimize distillation
        for local_epoch in range(n_distill_update_epochs):
            idx_order = torch.randperm(n_envs, device=device, dtype=torch.int32)
            idx_order = idx_order.view(n_envs // n_distill_batch_envs, n_distill_batch_envs)

            for mb_idx in idx_order:
                mb_obs = rollout.obs[:, mb_idx]
                mb_dones = rollout.dones[:, mb_idx]
                mb_v_ext = rollout.values_ext[:, mb_idx]

                mb_actor_state = initial_actor_state[mb_idx]

                # Optimize policy (distillation component)
                _, _, v_ext_from_actor = actor.forward_multi_step(mb_actor_state, mb_obs, mb_dones)

                distill_loss_ext = F.mse_loss(mb_v_ext, v_ext_from_actor)
                distill_loss = distill_loss_ext

                actor_loss = distill_loss * 0.5

                actor_opt.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
                actor_opt.step()

                # Logging info
                with torch.no_grad():
                    divisor = len(idx_order) * n_distill_update_epochs
                    log_distill_loss += distill_loss.item() / divisor

        writer.add_scalar("loss/approx_kl", log_approx_kl, iteration)
        writer.add_scalar("loss/clipfrac", log_clipfracs, iteration)
        writer.add_scalar("loss/value_loss", log_v_loss, iteration)
        writer.add_scalar("loss/policy_loss", log_pg_loss, iteration)
        writer.add_scalar("loss/entropy_loss", log_entropy_loss, iteration)
        writer.add_scalar("loss/distill_loss", log_distill_loss, iteration)
        writer.add_scalar("env/episodic_return", np.mean(episodic_return_history), iteration)
        writer.add_scalar("env/average_reward", average_reward, iteration)
        writer.add_scalar("debug/time_per_iter", time.monotonic() - time_iter_start, iteration)

    torch.save(actor.state_dict(), os.path.join(run_dir, "final-actor.pth"))
    torch.save(critic.state_dict(), os.path.join(run_dir, "final-critic.pth"))
    writer.close()


if __name__ == "__main__":
    train()
