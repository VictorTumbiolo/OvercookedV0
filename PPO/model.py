import torch.nn as nn

# Pytorch ANN
class SharedActorCritic(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        # Relu for consistnency
        self.shared = nn.Sequential(
            nn.Linear(96, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        self.actor = nn.Linear(128, num_actions)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = self.shared(x)
        return self.actor(x), self.critic(x)