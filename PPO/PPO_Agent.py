import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Align params w MAPPO version
class PPOUpdater:
    def __init__(self, model, lr=3e-4, gamma=0.99, clip_val=0.2, ent_coef=0.05):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_val = clip_val
        self.ent_coef = ent_coef
        self.buffer = RolloutBuffer()

    def update(self):
        # prep data from buffer
        states, actions, log_probs_old, rewards, values, dones = self.buffer.get_all()

        # calc advantages
        # How much better was move than critic expected
        advantages = self._calculate_advantages(rewards, values, dones)
        # targets = advantages + values  # The 'true' value the critic should have predicted
        values = values.squeeze()
        targets = advantages + values.detach()

        # normalize
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO epocs
        for _ in range(3):
            # Get current policy predictions
            policy_logits, current_values = self.model(states)

            # Action distribution maths
            dist = torch.distributions.Categorical(logits=policy_logits)
            log_probs_new = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            # Ratio for clipping: r(t) = pi_new / pi_old
            ratio = torch.exp(log_probs_new - log_probs_old)

            # clipped obj
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_val, 1.0 + self.clip_val) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # mse
            # value_loss = nn.MSELoss()(current_values.squeeze(), targets)
            current_values = current_values.squeeze()
            value_loss = nn.MSELoss()(current_values, targets)

            # loss total
            loss = policy_loss + 0.5 * value_loss - self.ent_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        # Decay eploratyion
        self.ent_coef = max(0.001, self.ent_coef * 0.9995)
        self.buffer.clear()

    def _calculate_advantages(self, rewards, values, dones, lam=0.95):
        advantages = torch.zeros_like(rewards)
        gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * lam * (1 - dones[t]) * gae
            advantages[t] = gae

        return advantages

# Old implementation
class RolloutBuffer:
    def __init__(self):
        self.clear()

    def store(self, s, a, r, lp, v, d):
        self.states.append(s)
        self.actions.append(a)
        self.rewards.append(r)
        self.log_probs.append(lp)
        self.values.append(v)
        self.dones.append(d)

    def clear(self):
        self.states, self.actions, self.rewards = [], [], []
        self.log_probs, self.values, self.dones = [], [], []

    def get_all(self):
        # Flatten and convert to Tensors
        return (torch.tensor(np.array(self.states), dtype=torch.float32),
                torch.cat(self.actions),
                torch.cat(self.log_probs),
                torch.tensor(self.rewards, dtype=torch.float32),
                torch.cat(self.values),
                torch.tensor(self.dones, dtype=torch.float32))