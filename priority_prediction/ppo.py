"""
Proximal Policy Optimization (PPO) for CPU Priority Scheduling
===============================================================
Reference: Schulman et al., "Proximal Policy Optimization Algorithms", 2017.

Algorithm summary:
  1. Collect a batch of trajectories using the current policy (actor).
  2. Compute advantages using rewards-to-go and the critic (value function).
  3. Normalise advantages for stability.
  4. Perform several epochs of gradient updates on the clipped surrogate
     objective, keeping the new policy close to the old one.
  5. Repeat.

Key implementation details:
  - Actor: maps flattened observation -> logits over discrete priorities.
  - Critic: maps flattened observation -> scalar state value.
  - Both share the same FeedForwardNN architecture (2 hidden layers of 64).
  - The action distribution is Categorical (proper for Discrete action spaces).
  - Advantages are normalised across the batch for stability.
"""

import numpy as np
import torch
import gymnasium as gym
from torch.distributions import Categorical
from torch.optim import Adam
from torch.nn import MSELoss

from network import FeedForwardNN

class PPO:
    def __init__(self, env: gym.Env, obs_enc_dim: int) -> None:
        """
        Initialise the PPO agent.

        Args:
            env: The PrioritySchedulerEnv instance.
            obs_enc_dim: (unused, kept for compatibility) intended size of an
                         optional observation encoder network.
        """
        # Environment information
        self.env = env
        self.obs_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
        self.obs_enc_dim = obs_enc_dim
        self.act_dim = env.action_space.n
        #self.act_dim = env.action_space.shape[0]

        # Hyperparameters
        self._init_hyperparameters()

        # ALG STEP 1 — Initialise policy (actor) and value (critic) networks.
        # The actor outputs logits for each discrete priority action.
        # The critic outputs a scalar value estimate for the state.
        self.actor = FeedForwardNN(self.obs_dim, self.act_dim)
        self.critic = FeedForwardNN(self.obs_dim, 1)

        # Network optimizers
        self.actor_optim = Adam(self.actor.parameters(), lr = self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr = self.lr)

    def _init_hyperparameters(self):
        """Set PPO hyperparameters."""
        self.timesteps_per_batch = 2048   # Steps collected before each update
        self.max_timesteps_per_episode = 200
        self.gamma = 0.99                  # Discount factor
        self.n_updates_per_iteration = 10  # Epochs of optimisation per batch
        self.clip = 0.2                    # PPO clip range
        self.lr = 3e-4                     # Learning rate

    def learn(self, n_steps):
        """Main training loop.  Runs until n_steps total environment steps."""
        import time, datetime

        n = 0 # number of steps taken
        iteration = 0
        learn_start = time.time()

        while n < n_steps: # ALG STEP 2 — loop until enough steps
            iter_start = time.time()

            # ALG STEP 3 — Collect a batch of trajectories with the current policy
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens, batch_rews = self.rollout()

            # Calculate how many timesteps collected in batch
            n += np.sum(batch_lens)
            iteration += 1

            # Calculate V_{phi, k} — value estimates for each state in the batch
            V, _ = self.evaluate(batch_obs, batch_acts)

            # ALG STEP 5 — Compute advantages: rewards-to-go minus value baseline
            A_k = batch_rtgs - V.detach()

            # Normalise advantages for stability (empirical finding)
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            # ALG STEP 6 & 7 — Perform multiple epochs of PPO update
            actor_loss_sum = 0.0
            critic_loss_sum = 0.0
            for _ in range(self.n_updates_per_iteration):
                # Re-evaluate with updated policy to get new log probs
                V, curr_log_probs = self.evaluate(batch_obs, batch_acts)

                # Probability ratio r_t(theta) = pi_theta / pi_theta_old
                ratios = torch.exp(curr_log_probs - batch_log_probs)

                # Clipped surrogate objective (PPO公式):
                # L_CLIP = min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t)
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1-self.clip, 1+self.clip) * A_k

                actor_loss = (-torch.min(surr1, surr2)).mean()
                # Value loss: MSE between critic predictions and rewards-to-go
                critic_loss = MSELoss()(V, batch_rtgs)
                
                actor_loss_sum += actor_loss.item()
                critic_loss_sum += critic_loss.item()

                # Update actor
                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()

                # Update critic
                self.critic_optim.zero_grad()
                critic_loss.backward(retain_graph=True)
                self.critic_optim.step()

            avg_actor_loss = actor_loss_sum / self.n_updates_per_iteration
            avg_critic_loss = critic_loss_sum / self.n_updates_per_iteration

            # Reward stats from the batch
            ep_rew_sums = [sum(ep) for ep in batch_rews]
            mean_rew = np.mean(ep_rew_sums)
            min_rew = np.min(ep_rew_sums)
            max_rew = np.max(ep_rew_sums)
            mean_ep_len = np.mean(batch_lens)

            iter_time = time.time() - iter_start
            elapsed = time.time() - learn_start
            remaining = (elapsed / n) * (n_steps - n) if n > 0 else 0
            pct = min(100.0, 100.0 * n / n_steps)
            print(
                f"  Iter {iteration:3d} | Steps {int(n):6d}/{n_steps} ({pct:5.1f}%) | "
                f"Time {iter_time:.1f}s | ETA {remaining:.0f}s | "
                f"Rew {mean_rew:.1f} [{min_rew:.1f}, {max_rew:.1f}] | "
                f"EpLen {mean_ep_len:.0f} | "
                f"ALoss {avg_actor_loss:.4f} | CLoss {avg_critic_loss:.4f}"
            )

    def rollout(self):
        """
        Collect a batch of experience by running the current policy.

        Resets the environment repeatedly, stepping through episodes until
        timesteps_per_batch steps have been collected.  Returns tensors of
        observations, actions, log-probabilities, rewards-to-go, episode
        lengths, and raw rewards.
        """
        # batch data
        batch_obs = []                  # batch observations
        batch_acts = []                 # batch actions
        batch_log_probs = []            # log probs of each action
        batch_rews = []                 # batch rewards
        batch_rtgs = []                 # batch rewards to go
        batch_lens = []                 # episodic lengths in batch

        # Number of timesteps run this batch
        t = 0

        while t < self.timesteps_per_batch: 
            # Rewards this episode
            ep_rews = []

            obs, _ = self.env.reset()
            obs = np.ravel(obs)
            done = False

            for ep_t in range(self.max_timesteps_per_episode):
                # Increment timesteps run this batch
                t += 1

                # Collect observation
                batch_obs.append(obs)

                action, log_prob = self.get_action(obs)
                obs, rew, done, _, _ = self.env.step(action)
                obs = np.ravel(obs)

                # Collect reward, action, and log_prob
                ep_rews.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)
            
                if done:
                    break

            # Collect episodic length and rewards
            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)
        
        batch_obs = torch.tensor(np.array(batch_obs), dtype=torch.float)
        batch_acts = torch.tensor(np.array(batch_acts), dtype=torch.long)
        batch_log_probs = torch.tensor(np.array(batch_log_probs), dtype=torch.float)

        # ALG STEP 4 - Compute rewards-to-go
        batch_rtgs = self.compute_rtgs(batch_rews)

        # Return batch data
        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens, batch_rews

    def get_action(self, obs):
        """
        Sample an action from the current policy.

        The actor produces logits over priority levels.  A Categorical
        distribution is built from these logits, and one action is sampled.
        Returns the integer action and its log-probability.
        """
        logits = self.actor(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.item()
    
    def compute_rtgs(self, batch_rews):
        """
        Compute rewards-to-go (discounted cumulative rewards) for each step.

        Walk backwards through each episode, accumulating:
            G_t = r_t + gamma * G_{t+1}
        Inserting at the front keeps time-ordering intact.
        """
        batch_rtgs = []

        for ep_rews in reversed(batch_rews):
            discounted_reward = 0
            for rew in reversed(ep_rews):
                discounted_reward = rew + (discounted_reward * self.gamma)
                batch_rtgs.insert(0, discounted_reward)

        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)
        return batch_rtgs

    def evaluate(self, batch_obs, batch_acts):
        """
        Compute state values and action log-probabilities for a batch.

        Used during the PPO update:
          - critic  -> V(s)  for the advantage estimate
          - actor   -> log pi(a|s) for the clipped surrogate objective
        """
        V = self.critic(batch_obs).squeeze()
        logits = self.actor(batch_obs)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(batch_acts)
        return V, log_probs

