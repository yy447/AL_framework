import torch
from rl.policy import ActorCritic
import torch.nn.functional as F
from rl.replay_buffer import RolloutBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class PPO:
    def __init__(self, state_dim, config):
        self.state_dim = state_dim
        self.action_dim = config.get("action_dim", 3)

        self.gamma = config.get("gamma", 0.99)
        self.lam = config.get("gae_lambda", 0.95)  # GAE λ
        self.eps_clip = config.get("eps_clip", 0.2)
        self.K_epochs = config.get("K_epochs", 4)
        self.lr = config.get("lr", 3e-4)
        self.entropy_coef = config.get("entropy_coef", 0.02)

        self.policy = ActorCritic(state_dim, self.action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)

        self.policy_old = ActorCritic(state_dim, self.action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.buffer = RolloutBuffer()

    def select_action(self, state):
        state_tensor = torch.as_tensor(
            state, dtype=torch.float32, device=device
        ).unsqueeze(0)
        with torch.no_grad():
            action, log_prob, state_value = self.policy_old.act(state_tensor)

        self.buffer.states.append(state_tensor)  # (1, state_dim)
        self.buffer.actions.append(action.squeeze(0))  # (action_dim,)
        self.buffer.logprobs.append(log_prob.squeeze(0))  # ()
        self.buffer.state_values.append(state_value.squeeze(0))  # ()
        return action.cpu().numpy()[0]

    def compute_gae(self, rewards, values, dones):
        advantages = []
        gae = 0.0
        next_values = values[1:] + [0.0]
        for r, v, nv, d in zip(
            reversed(rewards), reversed(values), reversed(next_values), reversed(dones)
        ):
            delta = r + self.gamma * nv * (1 - d) - v
            gae = delta + self.gamma * self.lam * (1 - d) * gae
            advantages.insert(0, gae)
        return torch.as_tensor(advantages, dtype=torch.float32, device=device)

    def update(self):
        if len(self.buffer.rewards) == 0:
            return

        rewards = list(self.buffer.rewards)
        dones = list(self.buffer.is_terminals)
        values = [v.item() for v in self.buffer.state_values]  # V(s_t)

        # --- GAE & returns ---
        advantages = self.compute_gae(rewards, values, dones)  # (T,)
        values_t = torch.as_tensor(values, dtype=torch.float32, device=device)  # (T,)
        returns = advantages + values_t  # (T,)

        if advantages.numel() > 1:
            std = advantages.std(unbiased=False)
            if float(std) > 1e-8:
                advantages = (advantages - advantages.mean()) / (std + 1e-8)

        old_states = torch.cat(self.buffer.states, dim=0).to(device)  # (T, state_dim)
        old_actions = torch.stack(self.buffer.actions, dim=0).to(
            device
        )  # (T, action_dim)
        old_logprobs = torch.stack(self.buffer.logprobs, dim=0).to(device)  # (T,)

        for _ in range(self.K_epochs):
            log_probs, state_values, dist_entropy = self.policy.evaluate(
                old_states, old_actions
            )  # (T,), (T,1) or (T,), (T,)

            ratios = torch.exp(log_probs - old_logprobs.detach())  # (T,)

            surr1 = ratios * advantages
            surr2 = (
                torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            )

            # Value loss
            value_loss = F.mse_loss(state_values.view(-1), returns.view(-1))

            # loss
            loss = (
                -torch.min(surr1, surr2).mean()
                + 0.5 * value_loss
                - self.entropy_coef * dist_entropy.mean()
            )

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()
