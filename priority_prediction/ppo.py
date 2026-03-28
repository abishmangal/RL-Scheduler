import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.optim import Adam
from torch.nn import MSELoss
import gymnasium as gym

from network import FeedForwardNN


class PPO:
    def __init__(self, env: gym.Env, obs_enc_dim: int) -> None:
        # Environment information
        self.env = env
        self.obs_dim = env.observation_space.shape[0] * env.observation_space.shape[1]  # flattened (encoder_context+1) * 6
        self.obs_enc_dim = obs_enc_dim
        self.act_dim = env.action_space.n  # discrete action count

        # Hyperparameters
        self._init_hyperparameters()

        # Actor: outputs logits over discrete actions
        self.actor = FeedForwardNN(self.obs_dim, self.act_dim)
        # Critic: outputs single value estimate
        self.critic = FeedForwardNN(self.obs_dim, 1)

        # Network optimizers
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

    def _init_hyperparameters(self):
        self.timesteps_per_batch = 2048
        self.max_timesteps_per_episode = 200
        self.gamma = 0.99
        self.n_updates_per_iteration = 10
        self.clip = 0.2
        self.lr = 3e-4

    def learn(self, n_steps):
        n = 0  # total timesteps so far
        iteration = 0

        while n < n_steps:
            # Collect batch of experience
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()

            n += np.sum(batch_lens)
            iteration += 1

            # Compute baseline value estimates
            V, _ = self.evaluate(batch_obs, batch_acts)

            # Advantage estimation
            A_k = batch_rtgs - V.detach()

            # Normalize advantages for training stability
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            # PPO update loop
            for update in range(self.n_updates_per_iteration):
                V, curr_log_probs = self.evaluate(batch_obs, batch_acts)

                # Probability ratio: pi_theta / pi_theta_old
                ratios = torch.exp(curr_log_probs - batch_log_probs)

                # Clipped surrogate objective
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = MSELoss()(V, batch_rtgs)

                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()

                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

            print(f"Iteration {iteration} | Timesteps: {n}/{n_steps} | "
                  f"Actor Loss: {actor_loss.item():.4f} | Critic Loss: {critic_loss.item():.4f} | "
                  f"Avg Episode Length: {np.mean(batch_lens):.1f}")

    def rollout(self):
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_rtgs = []
        batch_lens = []

        t = 0

        while t < self.timesteps_per_batch:
            ep_rews = []

            obs, _ = self.env.reset()
            obs = np.ravel(obs).astype(np.float32)  # flatten (encoder_context+1, 6) -> 1D
            done = False

            for ep_t in range(self.max_timesteps_per_episode):
                t += 1

                batch_obs.append(obs)

                action, log_prob = self.get_action(obs)
                obs, rew, done, _, _ = self.env.step(action)
                obs = np.ravel(obs).astype(np.float32)

                ep_rews.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                if done:
                    break

            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)

        batch_obs = torch.tensor(np.array(batch_obs), dtype=torch.float)
        batch_acts = torch.tensor(np.array(batch_acts), dtype=torch.long)    # FIX: long for discrete
        batch_log_probs = torch.tensor(np.array(batch_log_probs), dtype=torch.float)
        batch_rtgs = self.compute_rtgs(batch_rews)

        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens

    def get_action(self, obs):
        obs_tensor = torch.tensor(obs, dtype=torch.float)

        # Actor outputs logits → Categorical distribution for discrete actions
        logits = self.actor(obs_tensor)
        dist = Categorical(logits=logits)   # FIX: Categorical instead of MultivariateNormal

        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.detach().item(), log_prob.detach().item()

    def compute_rtgs(self, batch_rews):
        batch_rtgs = []

        for ep_rews in reversed(batch_rews):
            discounted_reward = 0
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)

        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)
        return batch_rtgs

    def evaluate(self, batch_obs, batch_acts):
        V = self.critic(batch_obs).squeeze()

        # FIX: Categorical log_prob for discrete actions
        logits = self.actor(batch_obs)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(batch_acts)

        return V, log_probs

    def save(self, path="ppo_scheduler.pt"):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
        }, path)
        print(f"Model saved to {path}")

    def load(self, path="ppo_scheduler.pt"):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        print(f"Model loaded from {path}")