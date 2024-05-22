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

            rollouts = []
            for agent in agents:
                if not done[agent] and not dones[agent] and not truncations[agent]:
                    rollouts.append({
                        'obs': torch.tensor(obs[agent], dtype=torch.float32),
                        'actions': torch.tensor([actions[agent]]),
                        'rewards': [rewards[agent]],
                        'masks': [True],
                        'next_obs': torch.tensor(next_obs[agent], dtype=torch.float32)
                    })

            # Debug: Display the collected rollouts
            print("Debug: Collected rollouts")
            for r in rollouts:
                print(f"obs: {r['obs'].shape}, actions: {r['actions'].shape}, rewards: {r['rewards']}, masks: {r['masks']}, next_obs: {r['next_obs'].shape}")

            for agent in agents:
                if not done[agent] and not dones[agent] and not truncations[agent]:
                    policies[agent].update(rollouts)

            obs = next_obs
            done = dones

if __name__ == "__main__":
    train()
