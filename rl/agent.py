import torch
import torch.nn as nn
from rl.policy import ActorCritic
import torch.nn.functional as F
from rl.replay_buffer import RolloutBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class PPO:
    def __init__(self, state_dim, config):
        self.state_dim = state_dim
        self.action_dim = 3

        self.gamma = config.get("gamma", 0.99)
        self.eps_clip = config.get("eps_clip", 0.2)
        self.K_epochs = config.get("K_epochs", 4)
        self.lr = config.get("lr", 0.0003)

        # create policy network
        self.policy = ActorCritic(state_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)

        self.policy_old = ActorCritic(state_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.buffer = RolloutBuffer()

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action, log_prob, state_value = self.policy_old.act(state_tensor)

        self.buffer.states.append(state_tensor)
        self.buffer.actions.append(action.squeeze(0))
        self.buffer.logprobs.append(log_prob.squeeze(0))
        self.buffer.state_values.append(state_value.squeeze(0))

        return action.cpu().numpy()[0]

    def update(self):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(
            reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)
        ):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)

        if len(rewards) > 1:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        else:
            rewards = rewards - rewards.mean()

        old_states = torch.cat(self.buffer.states, dim=0)
        old_actions = torch.stack(self.buffer.actions, dim=0)
        old_logprobs = torch.stack(self.buffer.logprobs, dim=0)
        old_state_values = torch.stack(self.buffer.state_values, dim=0)

        advantages = rewards.detach() - old_state_values.detach()

        for _ in range(self.K_epochs):
            log_probs, state_values, dist_entropy = self.policy.evaluate(
                old_states, old_actions
            )

            ratios = torch.exp(log_probs - old_logprobs.detach())

            surr1 = ratios * advantages
            surr2 = (
                torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            )
            state_values = state_values.squeeze()
            if state_values.dim() == 0:
                state_values = state_values.unsqueeze(0)

            if rewards.dim() == 0:
                rewards = rewards.unsqueeze(0)

            value_loss = F.mse_loss(state_values, rewards)

            loss = (
                -torch.min(surr1, surr2).mean()
                + 0.5 * value_loss
                - 0.01 * dist_entropy.mean()
            )

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())

        self.buffer.clear()
