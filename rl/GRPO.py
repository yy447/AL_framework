import torch
import torch.nn as nn
import torch.nn.functional as F
from rl.replay_buffer import RolloutBuffer
import numpy as np
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc_pi = nn.Linear(128, action_dim)

        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        nn.init.constant_(self.fc2.bias, 0.0)
        nn.init.orthogonal_(self.fc_pi.weight, gain=0.01)
        nn.init.constant_(self.fc_pi.bias, 0.0)

    def forward(self, state):
        x = F.elu(self.fc1(state))
        x = F.elu(self.fc2(x))
        logits = self.fc_pi(x)
        return logits

    def act(self, state):
        logits = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, logits

    def evaluate(self, state, action=None):
        logits = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)

        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action.squeeze(-1))

        entropy = dist.entropy()
        return action, log_prob, entropy, logits


class GRPO:
    def __init__(self, state_dim, config):
        self.state_dim = state_dim
        self.action_dim = 3

        self.gamma = config.get("gamma", 0.99)
        self.eps_clip = config.get("eps_clip", 0.15)
        self.K_epochs = config.get("K_epochs", 3)
        self.lr = config.get("lr", 0.0002)
        self.kl_coef = config.get("kl_coef", 0.02)
        self.entropy_coef = config.get("entropy_coef", 0.01)
        self.max_grad_norm = config.get("max_grad_norm", 0.5)
        self.update_every = config.get("update_every", 5)
        self.reward_window = config.get("reward_window", 20)

        self.reward_stats = {"mean": 0, "std": 1}
        self.reward_history = deque(maxlen=self.reward_window)

        self.policy = Actor(state_dim, self.action_dim).to(device)
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=self.lr, eps=1e-5
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=5
        )

        self.policy_old = Actor(state_dim, self.action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.buffer = RolloutBuffer()
        self.last_metrics = {}
        self.kl_history = deque(maxlen=20)

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            _, _, _, logits = self.policy_old.evaluate(state_tensor)

        probs = F.softmax(logits, dim=-1).cpu().numpy()[0]

        action = np.random.choice(len(probs), p=probs)
        log_prob = torch.log(torch.tensor(probs[action]))

        self.buffer.states.append(state_tensor)
        self.buffer.actions.append(torch.tensor(action))
        self.buffer.logprobs.append(log_prob)
        self.buffer.logits.append(logits.squeeze(0))

        return probs

    def compute_advantages(self, rewards):
        """Calculating Relative Advantage Using Within-Batch REWARD Rankings"""
        # initialize rewards
        rewards = np.array(rewards)
        if len(self.reward_history) > 5:
            recent_mean = np.mean(list(self.reward_history)[-5:])
            recent_std = np.std(list(self.reward_history)[-5:]) + 1e-8
            rewards = (rewards - recent_mean) / recent_std

        # calculate rank advantages
        rank = np.argsort(np.argsort(rewards))
        advantages = (rank - rank.mean()) / (rank.std() + 1e-8)

        # Further standardization advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return torch.tensor(advantages, dtype=torch.float32).to(device)

    def update(self):
        # 0. Skip if buffer is empty
        if len(self.buffer.rewards) == 0:
            return

        # 1. Calculate discounted returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(
            reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)
        ):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            self.reward_history.append(reward)
        # Update Reward Statistics
        if len(self.reward_history) > 0:
            self.reward_stats["mean"] = 0.9 * self.reward_stats["mean"] + 0.1 * np.mean(
                list(self.reward_history)
            )
            self.reward_stats["std"] = 0.9 * self.reward_stats["std"] + 0.1 * np.std(
                list(self.reward_history)
            )

        # 2. GRPO dominance calculation (based on relative ranking within batch)
        advantages = self.compute_advantages(rewards)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)

        # 3. Preparing batch data
        old_states = torch.cat(self.buffer.states, dim=0)
        old_actions = torch.stack(self.buffer.actions, dim=0)
        old_logprobs = torch.stack(self.buffer.logprobs, dim=0)
        old_logits = torch.stack(self.buffer.logits, dim=0)

        # 4. Multi-epoch optimization
        policy_losses = []
        kl_divs = []
        entropies = []

        for _ in range(self.K_epochs):
            # Evaluating Current Strategies
            _, log_probs, entropy, new_logits = self.policy.evaluate(
                old_states, old_actions
            )

            # Calculate the probability ratio
            ratios = torch.exp(log_probs - old_logprobs.detach())

            # GRPO loss function
            surr1 = ratios * advantages.detach()
            surr2 = (
                torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip)
                * advantages.detach()
            )
            policy_loss = -torch.min(surr1, surr2).mean()

            # KL dispersion penalty (difference in distribution between old and new strategies)
            old_dist = torch.distributions.Categorical(logits=old_logits.detach())
            new_dist = torch.distributions.Categorical(logits=new_logits)
            kl_div = torch.distributions.kl.kl_divergence(new_dist, old_dist).mean()

            # Adaptive KL Penalty
            current_kl = kl_div.item()
            self.kl_history.append(current_kl)

            # Increase penalties if KL dispersion is consistently high
            if len(self.kl_history) > 5 and np.mean(list(self.kl_history)[-5:]) > 0.05:
                kl_coef = min(self.kl_coef * 1.5, 0.1)
            # Reduce penalties if KL dispersion is consistently low
            elif (
                len(self.kl_history) > 10
                and np.mean(list(self.kl_history)[-10:]) < 0.01
            ):
                kl_coef = max(self.kl_coef * 0.8, 0.005)
            else:
                kl_coef = self.kl_coef

            # final loss
            loss = policy_loss + kl_coef * kl_div - self.entropy_coef * entropy.mean()

            self.optimizer.zero_grad()
            loss.backward()

            # gradient cropping
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)

            self.optimizer.step()

            # record metrics
            policy_losses.append(policy_loss.item())
            kl_divs.append(kl_div.item())
            entropies.append(entropy.mean().item())

        # recore average metrics
        self.last_metrics = {
            "kl_div": np.mean(kl_divs),
            "policy_loss": np.mean(policy_losses),
            "entropy": np.mean(entropies),
            "reward_mean": rewards.mean().item(),
            "kl_coef_used": kl_coef,
        }

        # 5. update old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # Adjusting the learning rate according to KL dispersion
        if np.mean(kl_divs) < 0.01:
            self.scheduler.step(rewards.mean().item())

        self.buffer.clear()

        return self.last_metrics
