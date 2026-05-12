"""
Deep Q-Network (DQN) for CPU Priority Scheduling
=================================================
Reference: Mnih et al., "Human-level control through deep reinforcement
           learning", Nature 2015.

Key components:
  - Q-Network: maps observation -> Q-values for each priority action
  - Target Network: frozen copy, updated periodically for stability
  - Replay Buffer: stores (s, a, r, s', done) transitions
  - Epsilon-greedy exploration: decays over training
  - Double DQN: uses online net to select action, target net to evaluate
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from collections import deque
import random
import gymnasium as gym

from network import FeedForwardNN


# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Stores transitions and samples random mini-batches for training."""

    def __init__(self, capacity: int = 50_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done):
        self.buffer.append((
            obs.astype(np.float32),
            int(action),
            float(reward),
            next_obs.astype(np.float32),
            bool(done),
        ))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)
        return (
            torch.tensor(np.array(obs),      dtype=torch.float32),
            torch.tensor(np.array(actions),  dtype=torch.long),
            torch.tensor(np.array(rewards),  dtype=torch.float32),
            torch.tensor(np.array(next_obs), dtype=torch.float32),
            torch.tensor(np.array(dones),    dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ---------------------------------------------------------------------------
# DQN Agent
# ---------------------------------------------------------------------------

class DQN:
    """
    Double DQN agent for CPU priority scheduling.

    Args:
        env             : PrioritySchedulerEnv
        lr              : learning rate
        gamma           : discount factor
        epsilon_start   : initial exploration rate
        epsilon_end     : minimum exploration rate
        epsilon_decay   : multiplicative decay per step
        batch_size      : mini-batch size for updates
        buffer_capacity : max transitions in replay buffer
        target_update   : steps between target network syncs
    """

    def __init__(self,
                 env: gym.Env,
                 lr: float = 1e-3,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.05,
                 epsilon_decay: float = 0.995,
                 batch_size: int = 64,
                 buffer_capacity: int = 50_000,
                 target_update: int = 500):

        self.env           = env
        self.gamma         = gamma
        self.epsilon       = epsilon_start
        self.epsilon_end   = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size    = batch_size
        self.target_update = target_update

        # FIX: obs_dim now uses shape[1]=7 automatically (was 6)
        obs_dim = (env.observation_space.shape[0] *
                   env.observation_space.shape[1])   # (encoder_context+1) * 7
        act_dim = env.action_space.n

        # Online Q-network (trained every step)
        self.q_net        = FeedForwardNN(obs_dim, act_dim)
        # Target Q-network (frozen, synced periodically)
        self.target_net   = FeedForwardNN(obs_dim, act_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer    = Adam(self.q_net.parameters(), lr=lr)
        self.replay       = ReplayBuffer(buffer_capacity)
        self.total_steps  = 0

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def get_action(self, obs: np.ndarray, greedy: bool = False) -> int:
        """Epsilon-greedy action selection."""
        if not greedy and random.random() < self.epsilon:
            return self.env.action_space.sample()
        obs_t = torch.tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            q_vals = self.q_net(obs_t)
        return int(torch.argmax(q_vals).item())

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def update(self) -> float | None:
        """Sample a mini-batch and perform one gradient step. Returns loss."""
        if len(self.replay) < self.batch_size:
            return None

        obs, actions, rewards, next_obs, dones = self.replay.sample(
            self.batch_size
        )

        # Current Q-values: Q(s, a)
        q_values = self.q_net(obs).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN target:
        #   a* = argmax_a Q_online(s', a)
        #   target = r + gamma * Q_target(s', a*)  (if not done)
        with torch.no_grad():
            next_actions = self.q_net(next_obs).argmax(dim=1, keepdim=True)
            next_q       = self.target_net(next_obs).gather(
                               1, next_actions).squeeze(1)
            targets = rewards + self.gamma * next_q * (1 - dones)

        loss = F.mse_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def learn(self, n_steps: int) -> list[float]:
        """
        Train the DQN for `n_steps` environment steps.
        Returns list of losses (one per update step).
        """
        losses    = []
        episode   = 0
        ep_reward = 0.0
        ep_steps  = 0

        obs, _ = self.env.reset()
        flat_obs = obs.ravel().astype(np.float32)

        print(f"Starting DQN training for {n_steps:,} steps ...")
        print(f"{'Step':>10} {'Episode':>8} {'Ep Reward':>12} "
              f"{'Epsilon':>8} {'Loss':>10}")
        print("-" * 55)

        for step in range(1, n_steps + 1):
            self.total_steps += 1

            action = self.get_action(flat_obs)
            next_obs, reward, done, _, _ = self.env.step(action)
            next_flat = next_obs.ravel().astype(np.float32)

            self.replay.push(flat_obs, action, reward, next_flat, done)
            flat_obs   = next_flat
            ep_reward += reward
            ep_steps  += 1

            loss = self.update()
            if loss is not None:
                losses.append(loss)

            # Decay epsilon
            self.epsilon = max(
                self.epsilon_end,
                self.epsilon * self.epsilon_decay
            )

            # Sync target network
            if step % self.target_update == 0:
                self.target_net.load_state_dict(self.q_net.state_dict())

            # Episode end
            if done:
                episode += 1
                if episode % 50 == 0:
                    avg_loss = np.mean(losses[-100:]) if losses else float('nan')
                    print(f"{step:>10,} {episode:>8} {ep_reward:>12.2f} "
                          f"{self.epsilon:>8.4f} {avg_loss:>10.4f}")
                obs, _   = self.env.reset()
                flat_obs = obs.ravel().astype(np.float32)
                ep_reward = 0.0
                ep_steps  = 0

        print(f"\nDQN training complete. Total episodes: {episode}")
        return losses

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: str = "model_weights/dqn_scheduler.pt"):
        torch.save(self.q_net.state_dict(), path)
        print(f"DQN model saved to {path}")

    def load(self, path: str):
        self.q_net.load_state_dict(
            torch.load(path, map_location="cpu")
        )
        self.target_net.load_state_dict(self.q_net.state_dict())
        print(f"DQN model loaded from {path}")