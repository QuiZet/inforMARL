import torch
from algorithms.mappo import MAPPO, make_env
from algorithms.MAPPOPolicy import MAPPOPolicy

import wandb
import argparse
from collections import defaultdict

def train(args):
    
    if args.log_wandb:
        # Initialize wandb with the new workspace and project
        wandb.init(
            project="mappo_hetmpe",  # Replace with your project name under utokyo-marl
            entity="utokyo-marl"  # Specify the workspace
        )
    
    env = make_env()
    initial_obs_tuple = env.reset()
    initial_obs = initial_obs_tuple[0]  # Extract the actual observations from the tuple
    agents = env.possible_agents

    obs_dims = {agent: env.observation_space(agent).shape[0] for agent in agents}
    action_dims = {agent: env.action_space(agent).n for agent in agents}
    policies = {agent: MAPPOPolicy(obs_dims[agent], action_dims[agent]) for agent in agents}

    num_episodes = 1000
    log_interval = 100
    all_rewards = {agent: [] for agent in agents}

    for episode in range(num_episodes):
        # Print starting episode
        if (episode + 1) % log_interval == 0 or episode == 0:
            print(f"\nStarting episode {episode + 1}")
            
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

            if all(dones.values()) or all(truncations.values()):
                break

            next_obs = {agent: torch.tensor(next_obs[agent], dtype=torch.float32).clone().detach() for agent in agents if agent in next_obs}

            for agent in agents:
                if agent not in next_obs:
                    print(f"Warning: No next observation for {agent} at episode {episode + 1}")
                    continue  # Skip this agent if no next observation is available
                if not done[agent] and not dones[agent] and not truncations[agent]:
                    rollouts.append({
                        'obs': {agent: torch.tensor(obs[agent], dtype=torch.float32).clone().detach()},
                        'actions': {agent: torch.tensor([actions[agent]]).clone().detach()},
                        'rewards': {agent: [rewards[agent]]},
                        'masks': {agent: [True]},
                        'next_obs': {agent: next_obs[agent].clone().detach()}
                    })

            # Update the policy for each agent after collecting sufficient rollouts
            if len(rollouts) >= 10:  # Ensure a reasonable number of rollouts
                if (episode + 1) % log_interval == 0 or episode == 0:
                    print(f"Updating policies with {len(rollouts)} rollouts")
                losses = []
                for agent in agents:
                    adv_losses, agent_losses = policies[agent].update(rollouts)
                    losses.append((adv_losses, agent_losses))
                rollouts = []  # Clear rollouts after update

                # Print and log losses
                for agent, (adv_losses, agent_losses) in zip(agents, losses):
                    if 'adversary' in agent:
                        adv_policy_loss, adv_value_loss, adv_entropy_loss = adv_losses
                        if adv_policy_loss is not None:
                            if (episode + 1) % log_interval == 0:
                                print(f"Adversary {agent} - Policy Loss: {adv_policy_loss}, Value Loss: {adv_value_loss}, Entropy Loss: {adv_entropy_loss}")
                            if args.log_wandb:
                                wandb.log({
                                    f"policy_loss_{agent}": adv_policy_loss,
                                    f"value_loss_{agent}": adv_value_loss,
                                    f"entropy_loss_{agent}": adv_entropy_loss
                                })
                        elif (episode + 1) % log_interval == 0:
                            print(f"Adversary {agent} - Policy Loss: {adv_policy_loss}, Value Loss: {adv_value_loss}, Entropy Loss: {adv_entropy_loss}")
                            
                    else:
                        agent_policy_loss, agent_value_loss, agent_entropy_loss = agent_losses
                        if agent_policy_loss is not None:
                            if (episode + 1) % log_interval == 0:
                                print(f"Agent {agent} - Policy Loss: {agent_policy_loss}, Value Loss: {agent_value_loss}, Entropy Loss: {agent_entropy_loss}")
                            if args.log_wandb:
                                wandb.log({
                                    f"policy_loss_{agent}": agent_policy_loss,
                                    f"value_loss_{agent}": agent_value_loss,
                                    f"entropy_loss_{agent}": agent_entropy_loss
                                })
                        elif (episode + 1) % log_interval == 0:
                            print(f"Agent {agent} - Policy Loss: {agent_policy_loss}, Value Loss: {agent_value_loss}, Entropy Loss: {agent_entropy_loss}")

            obs = next_obs
            done = dones

            # Update episode rewards
            for agent in agents:
                episode_rewards[agent] += rewards.get(agent, 0)

        # Append rewards to tracking list
        for agent in agents:
            all_rewards[agent].append(episode_rewards[agent])

        # Log metrics to wandb for each episode
        if args.log_wandb:
            wandb.log({f"episode_reward_{agent}": reward for agent, reward in episode_rewards.items()})
        
        # Print and log summary every log_interval episodes
        if (episode + 1) % log_interval == 0:
            avg_rewards = {agent: sum(all_rewards[agent][-log_interval:]) / log_interval for agent in agents}
            if args.log_wandb:
                wandb.log({f"avg_reward_{agent}": avg_reward for agent, avg_reward in avg_rewards.items()})

            print(f"Episode {episode + 1} summary (last {log_interval} episodes):")
            for agent, avg_reward in avg_rewards.items():
                print(f"  Agent {agent} average reward: {avg_reward}")

        # Reset the environment at the end of each episode
        initial_obs_tuple = env.reset()
        initial_obs = initial_obs_tuple[0]

    # Save the final model
    for agent, policy in policies.items():
        torch.save(policy.model.state_dict(), f"{agent}_policy_model.pth")
        if args.log_wandb:
            wandb.save(f"{agent}_policy_model.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_wandb", action='store_true', help="Log to Weights and Biases")
    args = parser.parse_args()
    
    train(args)
