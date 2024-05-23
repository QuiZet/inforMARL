import torch
from algorithms.mappo import MAPPO, make_env
from algorithms.MAPPOPolicy import MAPPOPolicy

import wandb
import argparse

def train():
    
    #Initialize wandb
    wandb.init(project="mappo_hetmpe", entity="yungisimon")
    
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
        print(f"\nStarting episode {episode}")
        obs = initial_obs
        episode_rewards = {agent: 0 for agent in agents}

        done = {agent: False for agent in agents}
        rollouts = []  # Collect rollouts for all agents
        while not all(done.values()):
            actions = {agent: policies[agent].model.get_action(torch.tensor(obs[agent], dtype=torch.float32))[0] for agent in agents if not done[agent]}
            actions = {agent: int(actions[agent]) for agent in actions}  # Convert actions to int for discrete action space
            next_obs_tuple = env.step(actions)
            next_obs = next_obs_tuple[0]  # Extract the actual observations from the tuple
            rewards = next_obs_tuple[1]
            dones = next_obs_tuple[2]
            truncations = next_obs_tuple[3]
            infos = next_obs_tuple[4]

            next_obs = {agent: torch.tensor(next_obs[agent], dtype=torch.float32) for agent in agents if agent in next_obs}
            #print(f'Next observations: {next_obs}')
            #print(f'Type of next observations: {type(next_obs)}')

            for agent in agents:
                if agent not in next_obs:
                    print(f"Warning: No next observation for {agent}")
                    continue  # Skip this agent if no next observation is available
                if not done[agent] and not dones[agent] and not truncations[agent]:
                    rollouts.append({
                        'obs': {agent: torch.tensor(obs[agent], dtype=torch.float32).clone().detach()},
                        'actions': {agent: torch.tensor([actions[agent]]).clone().detach()},
                        'rewards': {agent: [rewards[agent]]},
                        'masks': {agent: [True]},
                        'next_obs': {agent: torch.tensor(next_obs[agent], dtype=torch.float32).clone().detach()}
                    })

            # Update the policy for each agent after collecting sufficient rollouts
            if len(rollouts) >= 10:  # Ensure a reasonable number of rollouts
                print(f"Updating policies with {len(rollouts)} rollouts")
                for agent in agents:
                    policies[agent].update(rollouts)
                rollouts = []  # Clear rollouts after update

            obs = next_obs
            done = dones
            print(f"Done status: {done}")
            
            # Update episode rewards
            for agent in agents:
                episode_rewards[agent] += rewards.get(agent, 0)


        # Log metrics to wandb
        wandb.log({f"episode_reward_{agent}": reward for agent, reward in episode_rewards.items()})
        wandb.log({"episode": episode})
        
        # Print episode summary
        print(f"Episode {episode} summary:")
        for agent, reward in episode_rewards.items():
            print(f"  Agent {agent} reward: {reward}")
        
        # Reset the environment at the end of each episode
        initial_obs_tuple = env.reset()
        initial_obs = initial_obs_tuple[0]
        #print(f"Reset environment for next episode. Initial observations: {initial_obs}")

    # Save the final model
    for agent, policy in policies.items():
        torch.save(policy.model.state_dict(), f"{agent}_policy_model.pth")
        wandb.save(f"{agent}_policy_model.pth")

if __name__ == "__main__":
    train()

