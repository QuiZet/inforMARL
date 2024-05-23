import torch
import torch.nn as nn
import torch.optim as optim
from .mappo import MAPPO, compute_returns

class MAPPOPolicy:
    def __init__(self, obs_dim, action_dim, lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_param=0.2, ppo_epoch=10, num_mini_batch=32, value_loss_coef=0.5, entropy_coef=0.01):
        self.model = MAPPO(obs_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

    def update(self, rollouts):
        # Separate rollouts by agent type to handle different observation sizes
        adversary_rollouts = [r for r in rollouts if 'adversary' in r['obs'].keys()]
        agent_rollouts = [r for r in rollouts if 'agent' in r['obs'].keys()]

        def process_rollouts(rollouts):
            obs = torch.cat([r['obs'][list(r['obs'].keys())[0]].unsqueeze(0) for r in rollouts])
            actions = torch.cat([r['actions'][list(r['actions'].keys())[0]].unsqueeze(0) for r in rollouts])
            rewards = [r['rewards'][0] for r in rollouts]
            masks = [r['masks'][0] for r in rollouts]
            next_obs = torch.cat([r['next_obs'][list(r['next_obs'].keys())[0]].unsqueeze(0) for r in rollouts])
            
            rewards = torch.tensor(rewards, dtype=torch.float32)
            masks = torch.tensor(masks, dtype=torch.float32)

            next_value = self.model.critic(next_obs).detach()
            returns = compute_returns(next_value, rewards, masks, self.gamma)
            returns = torch.tensor(returns, dtype=torch.float32)  # Convert returns to a tensor

            advantages = returns - self.model.critic(obs).detach()

            policy_losses = []
            value_losses = []
            entropy_losses = []

            for _ in range(self.ppo_epoch):
                data_generator = self._mini_batch_generator(obs, actions, returns, advantages, masks)

                for sample in data_generator:
                    obs_batch, actions_batch, return_batch, adv_batch, mask_batch = sample

                    values, action_log_probs, dist_entropy = self.model.evaluate_actions(obs_batch, actions_batch)

                    ratio = torch.exp(action_log_probs - action_log_probs.detach())  # Fix to use action_log_probs
                    surr1 = ratio * adv_batch
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_batch

                    policy_loss = -torch.min(surr1, surr2).mean()
                    value_loss = (return_batch - values).pow(2).mean()

                    policy_losses.append(policy_loss.item())
                    value_losses.append(value_loss.item())
                    entropy_losses.append(dist_entropy.mean().item())

                    self.optimizer.zero_grad()
                    (value_loss * self.value_loss_coef + policy_loss - dist_entropy.mean() * self.entropy_coef).backward()
                    self.optimizer.step()

            avg_policy_loss = sum(policy_losses) / len(policy_losses) if policy_losses else None
            avg_value_loss = sum(value_losses) / len(value_losses) if value_losses else None
            avg_entropy_loss = sum(entropy_losses) / len(entropy_losses) if entropy_losses else None
            
            return avg_policy_loss, avg_value_loss, avg_entropy_loss
        
        # Process rollouts for adversaries and agents separately
        adv_losses = process_rollouts(adversary_rollouts) if adversary_rollouts else (None, None, None)
        agent_losses = process_rollouts(agent_rollouts) if agent_rollouts else (None, None, None)

        return adv_losses, agent_losses

    def _mini_batch_generator(self, obs, actions, returns, advantages, masks):
        batch_size = obs.size(0)
        mini_batch_size = max(1, batch_size // self.num_mini_batch)
        
        # Ensure the number of mini-batches does not exceed the batch size
        num_mini_batches = min(self.num_mini_batch, batch_size)
        mini_batch_size = max(1, batch_size // num_mini_batches)
        
        permutation = torch.randperm(batch_size)
        
        for start in range(0, batch_size, mini_batch_size):
            end = min(start + mini_batch_size, batch_size)
            if end > batch_size:
                break
            
            yield obs[permutation[start:end]], actions[permutation[start:end]], returns[permutation[start:end]], advantages[permutation[start:end]], masks[permutation[start:end]]
