import torch
import torch.nn as nn
from torch.distributions import Dirichlet
import torch.nn.functional as F
import torch.distributions as dist


class ActorCritic(nn.Module):
    def __init__(self, state_dim):
        super(ActorCritic, self).__init__()

        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU()
        )

        self.policy_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )

        self.value_head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, state):
        features = self.shared(state)
        logits = self.policy_head(features)
        value = self.value_head(features)
        return logits, value

    def act(self, state):
        logits, value = self.forward(state)

        logits = torch.nan_to_num(logits, nan=0.0)

        weights = F.softmax(logits, dim=-1)

        weights = torch.clamp(weights, min=1e-5)

        distribution = torch.distributions.Dirichlet(weights)
        action = distribution.sample()
        log_prob = distribution.log_prob(action)

        return action.detach(), log_prob.detach(), value.detach()

    def evaluate(self, state, action):
        logits, value = self.forward(state)

        logits = torch.nan_to_num(logits, nan=0.0)

        weights = F.softmax(logits, dim=-1)

        weights = torch.clamp(weights, min=1e-5)

        distribution = torch.distributions.Dirichlet(weights)

        log_probs = distribution.log_prob(action)
        entropy = distribution.entropy()
        return log_probs, value, entropy
