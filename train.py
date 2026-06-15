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
    rng = np.random.Generator(np.random.PCG64(12345678))

    run_dir = "./runs/" + datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(run_dir, flush_secs=30)

    n_envs = 128
    n_batch_size = 128
    n_actions = 7
    n_seq = 32
    n_iterations = 1_000_000_000 // (n_seq * n_envs)  # 1B env steps
    n_update_epochs = 4
    clip_coef = 0.1
    ent_coef = 0.01
    max_grad_norm = 1.0

    assert n_envs % n_batch_size == 0

    env = make_vector_env(n_envs)

    actor = Actor(n_actions).to(device)
    critic = Critic().to(device)

    # Resume
    actor.load_state_dict(torch.load("runs/20260614_141537/1408010.pth", map_location=device))
    start_iter = 175520 // n_update_epochs

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

    actor_opt = torch.optim.AdamW(actor.parameters(), lr=0.0002, eps=1e-8, weight_decay=1e-4)
    critic_opt = torch.optim.AdamW(critic.parameters(), lr=0.0002, eps=1e-8, weight_decay=1e-4)

    video = create_videowriter(run_dir, 60 / 4, period=50, disabled=False)

    for iteration in track(range(start_iter, n_iterations + 1), description="Training..."):
        time_iter_start = time.monotonic()

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
                    torch.save(actor.state_dict(), os.path.splitext(start_new)[0] + ".pth")

                next_obs, ext_reward, terminations, truncations, _ = env.step_wait()
                next_done = np.logical_or(terminations, truncations)
                ext_reward /= 20
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

        adv_ext, ret_ext = bootstrap_gae(
            rollout.values_ext,
            rollout.rewards_ext,
            rollout.dones,
            next_v_ext,
            next_done,
            0.999,
            0.95,
        )

        advantages = adv_ext
        adv_std, adv_mean = torch.std_mean(advantages)
        advantages = (advantages - adv_mean) / (adv_std + 1e-8)

        for local_epoch in range(n_update_epochs):
            log_approx_kl = 0.0
            log_clipfracs = 0.0
            log_v_loss = 0.0
            log_pg_loss = 0.0
            log_entropy_loss = 0.0
            log_distill_loss = 0.0

            idx_order = rng.choice(n_envs, size=n_envs, replace=False)
            idx_order = np.reshape(idx_order, (n_envs // n_batch_size, n_batch_size))
            idx_order.sort(axis=-1)

            for mb_idx in idx_order:
                mb_obs = rollout.obs[:, mb_idx]
                mb_logprobs = rollout.logprobs[:, mb_idx]
                mb_actions = rollout.actions[:, mb_idx]
                mb_dones = rollout.dones[:, mb_idx]
                mb_advantages = advantages[:, mb_idx]
                mb_ret_ext = ret_ext[:, mb_idx]
                mb_v_ext = rollout.values_ext[:, mb_idx]

                mb_actor_state = initial_actor_state[mb_idx]
                mb_critic_state = initial_critic_state[mb_idx]

                # Optimize critic
                mb_critic_state, newv_ext = critic.forward_multi_step(mb_critic_state, mb_obs, mb_dones)

                v_loss_ext = F.huber_loss(newv_ext, mb_ret_ext)
                v_loss = v_loss_ext

                critic_opt.zero_grad()
                v_loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
                critic_opt.step()

                # Optimize policy
                mb_actor_state, newlogit, v_ext_from_actor = actor.forward_multi_step(mb_actor_state, mb_obs, mb_dones)
                _, newlogprob, newentropy = calc_action(newlogit, mb_actions)
                logratio = newlogprob - mb_logprobs
                ratio = torch.exp(logratio)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)

                mb_valid_mask = torch.logical_not(mb_dones)
                mb_valid_count = torch.clamp(mb_valid_mask.sum(), min=1.0)

                pg_loss = (torch.max(pg_loss1, pg_loss2) * mb_valid_mask).sum() / mb_valid_count
                distill_loss_ext = F.huber_loss(mb_v_ext, v_ext_from_actor)
                distill_loss = distill_loss_ext
                entropy_loss = newentropy.mean()

                actor_loss = pg_loss + distill_loss * 0.5 - entropy_loss * ent_coef

                actor_opt.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
                actor_opt.step()

                # Put back the RNN state and value
                if local_epoch == n_update_epochs - 1:
                    next_actor_state[mb_idx] = mb_actor_state.detach()
                next_critic_state[mb_idx] = mb_critic_state.detach()
                rollout.values_ext[:, mb_idx] = newv_ext.detach()

                # Logging info
                with torch.no_grad():
                    divisor = len(idx_order)

                    log_approx_kl += ((ratio - 1) - logratio).mean().item() / divisor
                    log_clipfracs += ((ratio - 1.0).abs() > clip_coef).float().mean().item() / divisor
                    log_v_loss += v_loss.item() / divisor
                    log_pg_loss += pg_loss.item() / divisor
                    log_entropy_loss += entropy_loss.item() / divisor
                    log_distill_loss += distill_loss.item() / divisor

            update_step = iteration * n_update_epochs + local_epoch
            writer.add_scalar("loss/approx_kl", log_approx_kl, update_step)
            writer.add_scalar("loss/clipfrac", log_clipfracs, update_step)
            writer.add_scalar("loss/value_loss", log_v_loss, update_step)
            writer.add_scalar("loss/policy_loss", log_pg_loss, update_step)
            writer.add_scalar("loss/entropy_loss", log_entropy_loss, update_step)
            writer.add_scalar("loss/distill_loss", log_distill_loss, update_step)

            if local_epoch == 0:
                writer.add_scalar("env/episodic_return", np.mean(episodic_return_history), update_step)
                writer.add_scalar("env/average_reward", average_reward, update_step)
                writer.add_scalar("debug/time_per_iter", time.monotonic() - time_iter_start, update_step)

    torch.save(actor.state_dict(), os.path.join(run_dir, "final.pth"))
    writer.close()


if __name__ == "__main__":
    train()
