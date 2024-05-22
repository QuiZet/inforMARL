import torch
from algorithms.mappo import MAPPO, make_env
from algorithms.MAPPOPolicy import MAPPOPolicy

def train():
    env = make_env()
    initial_obs_tuple = env.reset()
    initial_obs = initial_obs_tuple[0]  # Extract the actual observations from the tuple
    print(f'Initial observations: {initial_obs}')
    print(f'Type of initial observations: {type(initial_obs)}')
    agents = env.possible_agents

    obs_dims = {agent: env.observation_space(agent).shape[0] for agent in agents}
    action_dims = {agent: env.action_space(agent).n for agent in agents}
    policies = {agent: MAPPOPolicy(obs_dims[agent], action_dims[agent]) for agent in agents}

    num_episodes = 1000
    for episode in range(num_episodes):
        obs = initial_obs
        episode_rewards = {agent: 0 for agent in agents}

        done = {agent: False for agent in agents}
        while not all(done.values()):
            actions = {agent: policies[agent].model.get_action(torch.tensor(obs[agent], dtype=torch.float32))[0] for agent in agents if not done[agent]}
            actions = {agent: int(actions[agent]) for agent in actions}  # Convert actions to int for discrete action space
            next_obs_tuple = env.step(actions)
            next_obs = next_obs_tuple[0]  # Extract the actual observations from the tuple
            rewards = next_obs_tuple[1]
            dones = next_obs_tuple[2]
            truncations = next_obs_tuple[3]
            infos = next_obs_tuple[4]
            
            next_obs = {agent: torch.tensor(next_obs[agent], dtype=torch.float32) for agent in agents}
            print(f'Next observations: {next_obs}')
            print(f'Type of next observations: {type(next_obs)}')

            for agent in agents:
                if not done[agent] and not dones[agent] and not truncations[agent]:
                    rollouts = {
                        'obs': torch.tensor(obs[agent], dtype=torch.float32),
                        'actions': torch.tensor([actions[agent]], dtype=torch.int64),
                        'rewards': [rewards[agent]],  # Ensure rewards is a list
                        'masks': [not dones[agent]],  # Ensure masks is a list
                        'next_obs': next_obs[agent],
                        'old_action_log_probs': policies[agent].model.get_action(torch.tensor(obs[agent], dtype=torch.float32))[1]
                    }
                    print(f"Debug: Rollouts - obs: {rollouts['obs']}, actions: {rollouts['actions']}, rewards: {rollouts['rewards']}, masks: {rollouts['masks']}, next_obs: {rollouts['next_obs']}")
                    policies[agent].update(rollouts)

            obs = next_obs
            done = dones
            for agent in agents:
                episode_rewards[agent] += rewards[agent]

        total_episode_reward = sum(episode_rewards.values())
        print(f'Episode {episode}, Total Reward: {total_episode_reward}')
        initial_obs_tuple = env.reset()
        initial_obs = initial_obs_tuple[0]

if __name__ == "__main__":
    train()