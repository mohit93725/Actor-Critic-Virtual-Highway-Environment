import torch 
import torch.nn as nn

class ActorCritic(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(input_dim, 512),
        nn.LayerNorm(512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, 256),
        nn.LayerNorm(256),
        nn.ReLU()
        )

        self.actor = nn.Linear(256, output_dim)
        self.critic = nn.Linear(256, 1)

    def forward(self, x):
        x = self.model(x)
        return self.actor(x), self.critic(x)


model = ActorCritic(input_dim=25, output_dim=5)

