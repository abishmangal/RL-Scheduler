"""
DPO (Direct Preference Optimization) for CPU Priority Scheduling
Current Environment Version - 6 features, includes time quantum
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import gymnasium as gym


# ---------------- MODEL ----------------
class FeedForwardNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


# ---------------- DATASET ----------------
class PreferenceDataset(Dataset):
    """Stores preference pairs (observation, winner_action, loser_action)"""
    
    def __init__(self):
        self.obs = []
        self.aw = []
        self.al = []

    def add(self, obs, a_w, a_l):
        self.obs.append(obs.astype(np.float32))
        self.aw.append(a_w)
        self.al.append(a_l)

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.obs[idx]),
            torch.tensor(self.aw[idx]),
            torch.tensor(self.al[idx]),
        )


# ---------------- PREFERENCE COLLECTION ----------------
def collect_preferences(env, n_pairs=1000, horizon=15, n_candidates=4, seed=42):
    """
    Collect preference pairs by comparing random action rollouts.
    
    Args:
        env: Environment
        n_pairs: Number of preference pairs to collect
        horizon: How many steps to simulate for each candidate
        n_candidates: Number of actions to compare per state
        seed: Random seed
    """
    rng = np.random.default_rng(seed)
    dataset = PreferenceDataset()

    act_dim = env.action_space.n
    collected = 0

    print(f"\nCollecting {n_pairs} preference pairs...")

    while collected < n_pairs:
        obs, _ = env.reset()
        flat = np.ravel(obs).astype(np.float32)
        done = False

        while not done and collected < n_pairs:
            # Sample random candidate actions
            candidates = rng.choice(act_dim, size=min(n_candidates, act_dim), replace=False)
            scores = {}

            # Evaluate each candidate with a shallow rollout
            for c in candidates:
                env_copy = copy.deepcopy(env)
                _, r, done2, _, _ = env_copy.step(c)
                total = r
                steps = 0

                while not done2 and steps < horizon:
                    # Use random action for remainder of rollout
                    random_action = rng.choice(act_dim)
                    _, r, done2, _, _ = env_copy.step(random_action)
                    total += r
                    steps += 1

                scores[c] = total

            # Winner = highest cumulative reward, Loser = lowest
            winner = max(scores, key=scores.get)
            loser = min(scores, key=scores.get)

            if winner != loser:
                dataset.add(flat, winner, loser)
                collected += 1

                if collected % 200 == 0:
                    print(f"  {collected}/{n_pairs}")

            # Take a step with the winner action to advance the state
            obs, _, done, _, _ = env.step(winner)
            flat = np.ravel(obs).astype(np.float32)

    print("Preference collection complete.")
    return dataset


# ---------------- DPO CLASS ----------------
class DPO:
    """
    Direct Preference Optimization for CPU scheduling.
    
    Args:
        env: Environment
        ref_actor_path: Path to reference model weights (e.g., trained PPO)
        beta: Temperature parameter controlling deviation from reference
        lr: Learning rate
        batch_size: Batch size for training
    """
    
    def __init__(self, env, ref_actor_path=None, beta=0.1, lr=1e-4, batch_size=64):

        self.env = env
        self.beta = beta
        self.batch_size = batch_size

        # Calculate observation dimension for initial environment (5 features)
        obs_sample, _ = env.reset()
        obs_dim = np.ravel(obs_sample).shape[0]  # (encoder_context+1) * 5
        act_dim = env.action_space.n

        print(f"DPO Initialized:")
        print(f"  Observation dim: {obs_dim}")
        print(f"  Action dim: {act_dim}")

        # Policy network (trainable)
        self.policy = FeedForwardNN(obs_dim, act_dim)
        # Reference policy (frozen)
        self.ref_policy = FeedForwardNN(obs_dim, act_dim)

        # Load reference weights if provided
        if ref_actor_path:
            w = torch.load(ref_actor_path, map_location="cpu")
            self.policy.load_state_dict(w)
            self.ref_policy.load_state_dict(w)
            print(f"  Loaded reference from: {ref_actor_path}")
        else:
            # Initialize with same random weights
            self.ref_policy.load_state_dict(self.policy.state_dict())

        # Freeze reference policy
        for p in self.ref_policy.parameters():
            p.requires_grad = False

        self.opt = Adam(self.policy.parameters(), lr=lr)

    def _logp(self, model, obs, act):
        """Compute log probability of actions under given model"""
        logits = model(obs)
        logp = F.log_softmax(logits, dim=-1)
        return logp.gather(1, act.unsqueeze(1)).squeeze(1)

    def dpo_loss(self, obs, aw, al):
        """
        Compute DPO loss for a batch of preferences.
        
        Loss = -log σ(β * [(log π_θ(aw) - log π_ref(aw)) - 
                           (log π_θ(al) - log π_ref(al))])
        """
        # Log probabilities for winner and loser actions under current policy
        lp_w = self._logp(self.policy, obs, aw)
        lp_l = self._logp(self.policy, obs, al)

        # Log probabilities under reference policy
        lr_w = self._logp(self.ref_policy, obs, aw)
        lr_l = self._logp(self.ref_policy, obs, al)

        # DPO objective
        logits = self.beta * ((lp_w - lr_w) - (lp_l - lr_l))

        return -F.logsigmoid(logits).mean()

    def train(self, dataset, n_epochs=5, verbose=True):
        """Train the policy using collected preferences"""
        
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        print(f"\nStarting DPO training for {n_epochs} epochs...")
        print(f"  Dataset size: {len(dataset)}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Beta: {self.beta}")
        print("-" * 50)

        for ep in range(n_epochs):
            total_loss = 0
            self.policy.train()

            for obs, aw, al in loader:
                loss = self.dpo_loss(obs, aw, al)

                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
                self.opt.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(loader)
            
            if verbose:
                print(f"Epoch {ep+1}/{n_epochs} | loss={avg_loss:.4f}")

        print("DPO training finished.")

    def get_action(self, obs, greedy=True):
        """Get action from trained policy"""
        with torch.no_grad():
            x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            logits = self.policy(x)
            if greedy:
                return torch.argmax(logits, dim=1).item()
            else:
                # Sample from distribution (exploration)
                probs = F.softmax(logits, dim=-1)
                return torch.multinomial(probs, 1).item()

    def evaluate(self, env, n_episodes=5):
        """Evaluate the trained policy"""
        self.policy.eval()
        rewards = []
        lengths = []
        
        for ep in range(n_episodes):
            obs, _ = env.reset()
            flat_obs = np.ravel(obs).astype(np.float32)
            done = False
            ep_reward = 0
            ep_length = 0
            
            while not done:
                action = self.get_action(flat_obs, greedy=True)
                obs, reward, done, _, _ = env.step(action)
                flat_obs = np.ravel(obs).astype(np.float32)
                ep_reward += reward
                ep_length += 1
            
            rewards.append(ep_reward)
            lengths.append(ep_length)
        
        results = {
            'avg_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'avg_length': np.mean(lengths),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards)
        }
        
        print(f"\nEvaluation Results ({n_episodes} episodes):")
        print(f"  Avg Reward: {results['avg_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"  Avg Length: {results['avg_length']:.1f}")
        
        return results

    def save(self, path):
        torch.save(self.policy.state_dict(), path)
        print(f"DPO saved → {path}")

    def load(self, path):
        self.policy.load_state_dict(torch.load(path, map_location='cpu'))
        self.ref_policy.load_state_dict(self.policy.state_dict())
        print(f"DPO loaded from {path}")