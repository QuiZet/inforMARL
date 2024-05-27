import torch
import torch.nn as nn
import torch.optim as optim
from .mappo import MAPPO, compute_returns

class MAPPOPolicy:
    def __init__(self, obs_dim, action_dim, agent_obs_dim, agent_action_dim, lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_param=0.2, ppo_epoch=10, num_mini_batch=32, value_loss_coef=0.5, entropy_coef=0.01):
        self.adversary_model = MAPPO(obs_dim, action_dim)
        self.agent_model = MAPPO(agent_obs_dim, agent_action_dim)
        self.adversary_optimizer = optim.Adam(self.adversary_model.parameters(), lr=lr)
        self.agent_optimizer = optim.Adam(self.agent_model.parameters(), lr=lr)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

    def get_action(self, obs, model_type='adversaty'):
        model=self.adversary_model if model_type == 'adversary' else self.agent_model
        action, log_prob = model.get_action(obs)
        return action, log_prob

    def load_model(self, path, model_type='adversary'):
        if model_type == 'adversary':
            self.adversary_model.load_state_dict(torch.load(path))
        elif model_type == 'agent':
            self.agent_model.load_state_dict(torch.load(path))
    
    def update(self, rollouts):
        print("Update function called")
        print(f"Total rollouts: {len(rollouts)}")

        # Separate rollouts by agent type to handle different observation sizes
        adversary_rollouts = [r for r in rollouts if 'adversary' in list(r['obs'].keys())[0]]
        agent_rollouts = [r for r in rollouts if 'agent' in list(r['obs'].keys())[0]]

        print(f"Adversary rollouts: {len(adversary_rollouts)},\n Agent rollouts: {len(agent_rollouts)}")

        def process_rollouts(rollouts, model, optimizer):
            if not rollouts:
                print("No rollouts to process")
                return None, None, None

            print(f"Rollouts: {rollouts}")

            # Extract the first agent key from the first rollout as a template
            agent_key = list(rollouts[0]['obs'].keys())[0]
            print(f"Agent key: {agent_key}")

            try:
                obs_list = [r['obs'][agent_key].unsqueeze(0) for r in rollouts if agent_key in r['obs']]
                actions_list = [r['actions'][agent_key].unsqueeze(0) for r in rollouts if agent_key in r['actions']]
                rewards_list = [r['rewards'][agent_key][0] for r in rollouts if agent_key in r['rewards']]
                masks_list = [r['masks'][agent_key][0] for r in rollouts if agent_key in r['masks']]
                next_obs_list = [r['next_obs'][agent_key].unsqueeze(0) for r in rollouts if agent_key in r['next_obs']]

                # Print the lists to debug the data
                print("obs_list:", obs_list)
                print("actions_list:", actions_list)
                print("rewards_list:", rewards_list)
                print("masks_list:", masks_list)
                print("next_obs_list:", next_obs_list)

                # Convert lists to tensors
                obs = torch.cat(obs_list) if obs_list else torch.tensor([])
                actions = torch.cat(actions_list) if actions_list else torch.tensor([])
                rewards = torch.tensor(rewards_list, dtype=torch.float32) if rewards_list else torch.tensor([])
                masks = torch.tensor(masks_list, dtype=torch.float32) if masks_list else torch.tensor([])
                next_obs = torch.cat(next_obs_list) if next_obs_list else torch.tensor([])
            except KeyError as e:
                print(f"KeyError: {e}")
                print("Rollout keys available:", rollouts[0].keys())
                print("Observation keys available:", rollouts[0]['obs'].keys())
                raise e

            print("Next obs:", next_obs)
            print(f"Next obs shape: {next_obs.shape}")
            
            next_value = model.critic(next_obs).detach()
            print(f"Next value shape: {next_value.shape}")
            print("Next value:", next_value)
            returns = compute_returns(next_value, rewards, masks, self.gamma)
            print("Computed returns:", returns)

            # Ensure returns is a tensor
            returns = torch.stack(returns) if isinstance(returns, list) else returns
            print("Returns as tensor:", returns)

            advantages = returns - model.critic(obs).detach()
            print("Advantages:", advantages)

            policy_losses = []
            value_losses = []
            entropy_losses = []

            for _ in range(self.ppo_epoch):
                data_generator = self._mini_batch_generator(obs, actions, returns, advantages, masks)

                for sample in data_generator:
                    obs_batch, actions_batch, return_batch, adv_batch, mask_batch = sample

                    values, action_log_probs, dist_entropy = model.evaluate_actions(obs_batch, actions_batch)
                    print("Values:", values)
                    print("Action log probs:", action_log_probs)
                    print("Dist entropy:", dist_entropy)

                    ratio = torch.exp(action_log_probs - action_log_probs.detach())
                    surr1 = ratio * adv_batch
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_batch

                    policy_loss = -torch.min(surr1, surr2).mean()
                    value_loss = (return_batch - values).pow(2).mean()

                    policy_losses.append(policy_loss.item())
                    value_losses.append(value_loss.item())
                    entropy_losses.append(dist_entropy.mean().item())

                    optimizer.zero_grad()
                    (value_loss * self.value_loss_coef + policy_loss - dist_entropy.mean() * self.entropy_coef).backward()
                    optimizer.step()

            avg_policy_loss = sum(policy_losses) / len(policy_losses) if policy_losses else None
            avg_value_loss = sum(value_losses) / len(value_losses) if value_losses else None
            avg_entropy_loss = sum(entropy_losses) / len(entropy_losses) if entropy_losses else None
            
            return avg_policy_loss, avg_value_loss, avg_entropy_loss
        
        # Process rollouts for adversaries and agents separately
        adv_losses = process_rollouts(adversary_rollouts, self.adversary_model, self.adversary_optimizer) if adversary_rollouts else (None, None, None)
        agent_losses = process_rollouts(agent_rollouts, self.agent_model, self.agent_optimizer) if agent_rollouts else (None, None, None)

        return adv_losses, agent_losses

    def _mini_batch_generator(self, obs, actions, returns, advantages, masks):
        batch_size = obs.size(0)
        mini_batch_size = max(1, batch_size // self.num_mini_batch)
        
        num_mini_batches = min(self.num_mini_batch, batch_size)
        mini_batch_size = max(1, batch_size // num_mini_batches)
        
        permutation = torch.randperm(batch_size)
        
        for start in range(0, batch_size, mini_batch_size):
            end = min(start + mini_batch_size, batch_size)
            if end > batch_size:
                break
            
            yield obs[permutation[start:end]], actions[permutation[start:end]], returns[permutation[start:end]], advantages[permutation[start:end]], masks[permutation[start:end]]
