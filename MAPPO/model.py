import torch
import torch.nn as nn


class SharedActorCritic(nn.Module):
    def __init__(self, input_dim=96, num_actions=6):
        super(SharedActorCritic, self).__init__()

        # Shared MLP local features
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # Actor local
        self.actor = nn.Linear(128, num_actions)

        # Critic global state, Both agents
        self.critic = nn.Sequential(
            nn.Linear(input_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x, global_state):
        """
        x: (batch, 96)
        global_state: (batch, 192)
        """
        shared = self.shared(x)
        logits = self.actor(shared)
        value = self.critic(global_state)
        return logits, value