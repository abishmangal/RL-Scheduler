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

    def __init__(self, capacity: int = 100_000):
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
                 epsilon_decay: float = 0.9995,
                 batch_size: int = 64,
                 buffer_capacity: int = 100_000,
                 target_update: int = 1000):

        self.env           = env
        self.gamma         = gamma
        self.epsilon       = epsilon_start
        self.epsilon_end   = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size    = batch_size
        self.target_update = target_update

        # Calculate flattened observation dimension
        obs_shape = env.observation_space.shape  # (encoder_context+1, 5)
        obs_dim = obs_shape[0] * obs_shape[1]    # e.g., 31 * 5 = 155
        act_dim = env.action_space.n             # max_priority (e.g., 10)

        print(f"DQN Initialized:")
        print(f"  Observation shape: {obs_shape}")
        print(f"  Flattened dim: {obs_dim}")
        print(f"  Action dim: {act_dim}")

        # Online Q-network (trained every step)
        self.q_net        = FeedForwardNN(obs_dim, act_dim)
        # Target Q-network (frozen, synced periodically)
        self.target_net   = FeedForwardNN(obs_dim, act_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer    = Adam(self.q_net.parameters(), lr=lr)
        self.replay       = ReplayBuffer(buffer_capacity)
        self.total_steps  = 0
        self.losses       = []

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

        obs, actions, rewards, next_obs, dones = self.replay.sample(self.batch_size)

        # Current Q-values: Q(s, a)
        q_values = self.q_net(obs).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN target:
        #   a* = argmax_a Q_online(s', a)
        #   target = r + gamma * Q_target(s', a*)  (if not done)
        with torch.no_grad():
            next_actions = self.q_net(next_obs).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_obs).gather(1, next_actions).squeeze(1)
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

    def learn(self, n_steps: int, log_interval: int = 10000, eval_interval: int = 100000):
        """
        Train the DQN for `n_steps` environment steps.

        Args:
            n_steps: Total number of environment steps to train for
            log_interval: How often to print progress
            eval_interval: How often to evaluate greedy policy
        """
        losses = []
        episode = 0
        episode_reward = 0.0
        episode_length = 0
        
        # Tracking for logging
        episode_rewards = []
        episode_lengths = []

        obs, _ = self.env.reset()
        flat_obs = obs.ravel().astype(np.float32)

        print("\n" + "="*80)
        print(f"Starting DQN training for {n_steps:,} steps...")
        print(f"Buffer capacity: {self.replay.buffer.maxlen:,}")
        print(f"Batch size: {self.batch_size}")
        print(f"Target update: every {self.target_update} steps")
        print(f"Epsilon: {self.epsilon:.3f} → {self.epsilon_end:.3f}")
        print("="*80)
        print(f"{'Step':>12} {'Episode':>8} {'Ep Reward':>12} {'Epsilon':>8} {'Loss':>10}")
        print("-" * 55)

        for step in range(1, n_steps + 1):
            self.total_steps += 1

            # Select and execute action
            action = self.get_action(flat_obs)
            next_obs, reward, done, _, _ = self.env.step(action)
            next_flat = next_obs.ravel().astype(np.float32)

            # Store transition
            self.replay.push(flat_obs, action, reward, next_flat, done)
            flat_obs = next_flat
            episode_reward += reward
            episode_length += 1

            # Update network
            loss = self.update()
            if loss is not None:
                losses.append(loss)

            # Decay epsilon
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

            # Sync target network
            if step % self.target_update == 0:
                self.target_net.load_state_dict(self.q_net.state_dict())

            # Episode end
            if done:
                episode += 1
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                
                # Print progress
                if episode % 50 == 0:
                    avg_loss = np.mean(losses[-100:]) if losses else float('nan')
                    avg_reward = np.mean(episode_rewards[-50:]) if episode_rewards else 0
                    print(f"{step:>12,} {episode:>8} {avg_reward:>12.2f} "
                          f"{self.epsilon:>8.4f} {avg_loss:>10.6f}")
                
                # Reset environment
                obs, _ = self.env.reset()
                flat_obs = obs.ravel().astype(np.float32)
                episode_reward = 0.0
                episode_length = 0

        print(f"\n{'='*80}")
        print(f"DQN training complete!")
        print(f"Total steps: {self.total_steps:,}")
        print(f"Total episodes: {episode}")
        print(f"Final epsilon: {self.epsilon:.4f}")
        print(f"Average reward (last 100 episodes): {np.mean(episode_rewards[-100:]):.2f}")
        print(f"{'='*80}\n")
        
        return losses

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, n_episodes: int = 10) -> dict:
        """
        Evaluate the greedy policy over multiple episodes.
        
        Returns:
            Dictionary with average reward, episode length, etc.
        """
        self.q_net.eval()
        rewards = []
        lengths = []
        
        for ep in range(n_episodes):
            obs, _ = self.env.reset()
            flat_obs = obs.ravel().astype(np.float32)
            ep_reward = 0
            ep_length = 0
            done = False
            
            while not done:
                action = self.get_action(flat_obs, greedy=True)
                obs, reward, done, _, _ = self.env.step(action)
                flat_obs = obs.ravel().astype(np.float32)
                ep_reward += reward
                ep_length += 1
            
            rewards.append(ep_reward)
            lengths.append(ep_length)
        
        self.q_net.train()
        
        return {
            'avg_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'avg_length': np.mean(lengths),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards)
        }

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: str = "model_weights/dqn_scheduler.pt"):
        torch.save({
            'q_net_state_dict': self.q_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'total_steps': self.total_steps
        }, path)
        print(f"DQN model saved to {path}")

    def load(self, path: str):
        checkpoint = torch.load(path, map_location="cpu")
        self.q_net.load_state_dict(checkpoint['q_net_state_dict'])
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.total_steps = checkpoint['total_steps']
        print(f"DQN model loaded from {path}")