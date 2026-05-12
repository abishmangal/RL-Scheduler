import copy
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'priority_prediction'))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import gymnasium as gym

# FIX: import from network.py instead of redefining
from network import FeedForwardNN


# ---------------- DATASET ----------------
class PreferenceDataset(Dataset):
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

            candidates = rng.choice(act_dim, size=min(n_candidates, act_dim), replace=False)

            scores = {}

            for c in candidates:
                env_copy = copy.deepcopy(env)
                _, r, done2, _, _ = env_copy.step(c)

                total = r
                steps = 0

                while not done2 and steps < horizon:
                    _, r, done2, _, _ = env_copy.step(0)
                    total += r
                    steps += 1

                scores[c] = total

            winner = max(scores, key=scores.get)
            loser = min(scores, key=scores.get)

            if winner != loser:
                dataset.add(flat, winner, loser)
                collected += 1

                if collected % 200 == 0:
                    print(f"  {collected}/{n_pairs}")

            obs, _, done, _, _ = env.step(winner)
            flat = np.ravel(obs).astype(np.float32)

    print("Preference collection complete.")
    return dataset


# ---------------- DPO CLASS ----------------
class DPO:
    def __init__(self, env, ref_actor_path=None, beta=0.1, lr=1e-4, batch_size=64):

        self.env = env
        self.beta = beta
        self.batch_size = batch_size

        obs_sample, _ = env.reset()
        obs_dim = np.ravel(obs_sample).shape[0]
        act_dim = env.action_space.n

        # FIX: uses imported FeedForwardNN from network.py
        self.policy = FeedForwardNN(obs_dim, act_dim)
        self.ref_policy = FeedForwardNN(obs_dim, act_dim)

        if ref_actor_path:
            w = torch.load(ref_actor_path, map_location="cpu")
            self.policy.load_state_dict(w)
            self.ref_policy.load_state_dict(w)
        else:
            self.ref_policy.load_state_dict(self.policy.state_dict())

        for p in self.ref_policy.parameters():
            p.requires_grad = False

        self.opt = Adam(self.policy.parameters(), lr=lr)

    def _logp(self, model, obs, act):
        logits = model(obs)
        logp = F.log_softmax(logits, dim=-1)
        return logp.gather(1, act.unsqueeze(1)).squeeze(1)

    def dpo_loss(self, obs, aw, al):

        lp_w = self._logp(self.policy, obs, aw)
        lp_l = self._logp(self.policy, obs, al)

        lr_w = self._logp(self.ref_policy, obs, aw)
        lr_l = self._logp(self.ref_policy, obs, al)

        logits = self.beta * ((lp_w - lr_w) - (lp_l - lr_l))

        return -F.logsigmoid(logits).mean()

    def train(self, dataset, n_epochs=5):

        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for ep in range(n_epochs):
            total = 0
            self.policy.train()

            for obs, aw, al in loader:
                loss = self.dpo_loss(obs, aw, al)

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                total += loss.item()

            print(f"Epoch {ep+1}/{n_epochs} | loss={total/len(loader):.4f}")

        print("DPO training finished.")

    def get_action(self, obs):
        with torch.no_grad():
            x = torch.tensor(obs).float().unsqueeze(0)
            return torch.argmax(self.policy(x), dim=1).item()

    def save(self, path):
        torch.save(self.policy.state_dict(), path)
        print("DPO saved →", path)