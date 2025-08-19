# rl/policy.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Dirichlet


class ActorCritic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes=(128, 128),
        min_alpha: float = 1e-2,
        max_alpha: float = 100.0,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.min_alpha = float(min_alpha)
        self.max_alpha = float(max_alpha) if max_alpha is not None else None

        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_sizes[0]),
            nn.Tanh(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.Tanh(),
            nn.Linear(hidden_sizes[1], action_dim),
        )

        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_sizes[0]),
            nn.Tanh(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.Tanh(),
            nn.Linear(hidden_sizes[1], 1),
        )

    def _alpha(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.actor(x)
        alpha = F.softplus(raw) + self.min_alpha

        if self.max_alpha is not None:
            alpha = torch.clamp(alpha, max=self.max_alpha)
        alpha = torch.nan_to_num(
            alpha,
            nan=self.min_alpha,
            posinf=(self.max_alpha if self.max_alpha is not None else 1e6),
            neginf=self.min_alpha,
        )
        return alpha

    @torch.no_grad()
    def act(self, state: torch.Tensor):
        alpha = self._alpha(state)
        dist = Dirichlet(alpha)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        value = self.critic(state)
        return action, log_prob, value

    def evaluate(self, states: torch.Tensor, actions: torch.Tensor):
        actions = torch.clamp(actions, min=1e-8)
        actions = actions / actions.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        alpha = self._alpha(states)
        dist = Dirichlet(alpha)

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        values = self.critic(states).squeeze(-1)

        return log_probs, values, entropy
