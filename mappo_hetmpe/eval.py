import torch
from algorithms.mappo import MAPPO, make_env
from algorithms.MAPPOPolicy import MAPPOPolicy

def evaluate():
    env = make_env()
    initial_obs = env.reset()
    agents = env.possible_agents

    obs_dims = {agent: env.observation_space(agent).shape[0] for agent in agents}
    action_dims = {agent: env.action_space(agent).n for agent in agents}
    policies = {agent: MAPPOPolicy(obs_dims[agent], action_dims[agent]) for agent in agents}

    num_episodes = 100
    total_rewards = {agent: 0 for agent in agents}
    for episode in range(num_episodes):
        obs = initial_obs
        done = {agent: False for agent in agents}

        while not all(done.values()):
            actions = {agent: policies[agent].model.get_action(obs[agent])[0] for agent in agents if not done[agent]}
            actions = {agent: int(actions[agent]) for agent in actions}  # Convert actions to int for discrete action space
            next_obs, rewards, dones, truncations, infos = env.step(actions)

            for agent in agents:
                if not done[agent] and not dones[agent] and not truncations[agent]:
                    rollouts = {
                        'obs': torch.tensor(obs[agent], dtype=torch.float32),
                        'actions': torch.tensor([actions[agent]], dtype=torch.int64),
                        'rewards': [rewards[agent]],  # Ensure rewards is a list
                        'masks': [dones[agent]],      # Ensure masks is a list
                        'next_obs': torch.tensor(next_obs[agent], dtype=torch.float32),
                        'old_action_log_probs': policies[agent].model.get_action(torch.tensor(obs[agent], dtype=torch.float32))[1]
                    }
                    policies[agent].update(rollouts)

            obs = next_obs
            done = dones
            for agent in agents:
                total_rewards[agent] += rewards[agent]

        initial_obs = env.reset()

    print(f'Average rewards: {sum(total_rewards.values()) / num_episodes}')

if __name__ == "__main__":
    evaluate()
