import os
import torch
from algorithms.mappo import make_env
from algorithms.MAPPOPolicy import MAPPOPolicy

import cv2
import wandb
import argparse
from collections import defaultdict
from datetime import datetime

def render_env_with_opencv(env):
    img = env.render()
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow('Simple Tag Environment', img_bgr)
    cv2.waitKey(1)

def train(args):
    if args.log_wandb:
        wandb.init(
            project="mappo_hetmpe",
            entity="utokyo-marl"
        )

    env = make_env()
    initial_obs_tuple = env.reset()
    initial_obs = initial_obs_tuple[0]
    agents = env.possible_agents

    obs_dims = {agent: env.observation_space(agent).shape[0] for agent in agents}
    action_dims = {agent: env.action_space(agent).n for agent in agents}
    policies = {
        agent: MAPPOPolicy(
            obs_dims[agent],
            action_dims[agent],
            obs_dims[agent],
            action_dims[agent]
        ) for agent in agents
    }

    if args.load_model:
        for agent in agents:
            model_path = os.path.join(args.load_model, f"{agent}_policy_model.pth")
            model_type = 'adversary' if 'adversary' in agent else 'agent'
            policies[agent].load_model(model_path, model_type=model_type)

    output_dir = args.output_dir or f"models/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)

    num_episodes = 10000
    log_interval = 100
    all_rewards = {agent: [] for agent in agents}

    for episode in range(num_episodes):
        if (episode + 1) % log_interval == 0 or episode == 0:
            print(f"\nStarting episode {episode + 1}")

        obs = initial_obs
        episode_rewards = {agent: 0 for agent in agents}

        done = {agent: False for agent in agents}
        rollouts = []
        while not all(done.values()):
            actions = {agent: policies[agent].get_action(torch.tensor(obs[agent], dtype=torch.float32), model_type='adversary' if 'adversary' in agent else 'agent')[0] for agent in agents if not done[agent]}
            actions = {agent: int(actions[agent]) for agent in actions}

            next_obs_tuple = env.step(actions)
            next_obs = next_obs_tuple[0]
            rewards = next_obs_tuple[1]
            dones = next_obs_tuple[2]
            truncations = next_obs_tuple[3]
            infos = next_obs_tuple[4]

            if args.render:
                render_env_with_opencv(env)

            if all(dones.values()) or all(truncations.values()):
                break

            next_obs = {agent: torch.tensor(next_obs[agent], dtype=torch.float32).clone().detach() for agent in agents if agent in next_obs}

            for agent in agents:
                if agent not in next_obs:
                    print(f"Warning: No next observation for {agent} at episode {episode + 1}")
                    continue
                if not done[agent] and not dones[agent] and not truncations[agent]:
                    agent_type = 'adversary' if 'adversary' in agent else 'agent'
                    rollout = {
                        'type': agent_type,
                        'obs': {agent: torch.tensor(obs[agent], dtype=torch.float32).clone().detach()},
                        'actions': {agent: torch.tensor([actions[agent]]).clone().detach()},
                        'rewards': {agent: [rewards[agent]]},
                        'masks': {agent: [True]},
                        'next_obs': {agent: next_obs[agent].clone().detach()}
                    }
                    print(f"Collected rollout for {agent}: {rollout}")
                    rollouts.append(rollout)

            print(f"Collected {len(rollouts)} rollouts")
            if len(rollouts) >= 10:
                if (episode + 1) % log_interval == 0 or episode == 0:
                    print(f"Updating policies with {len(rollouts)} rollouts")
                losses = []
                for agent in agents:
                    adv_losses, agent_losses = policies[agent].update(rollouts)
                    print(f"Losses for agent {agent}: {adv_losses}, {agent_losses}")  # Debugging print
                    losses.append((adv_losses, agent_losses))
                rollouts = []

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
                        else:
                            print(f"Adversary {agent} - Losses are None")
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
                        else:
                            print(f"Agent {agent} - Losses are None")

            obs = next_obs
            done = dones

            for agent in agents:
                episode_rewards[agent] += rewards.get(agent, 0)

        for agent in agents:
            all_rewards[agent].append(episode_rewards[agent])

        if args.log_wandb:
            wandb.log({f"episode_reward_{agent}": reward for agent, reward in episode_rewards.items()})

        if (episode + 1) % log_interval == 0:
            avg_rewards = {agent: sum(all_rewards[agent][-log_interval:]) / log_interval for agent in agents}
            if args.log_wandb:
                wandb.log({f"avg_reward_{agent}": avg_reward for agent, avg_reward in avg_rewards.items()})

            print(f"Episode {episode + 1} summary (last {log_interval} episodes):")
            for agent, avg_reward in avg_rewards.items():
                print(f"  Agent {agent} average reward: {avg_reward}")

        initial_obs_tuple = env.reset()
        initial_obs = initial_obs_tuple[0]

    for agent, policy in policies.items():
        model_path = os.path.join(output_dir, f"{agent}_policy_model.pth")
        torch.save(policy.adversary_model.state_dict() if 'adversary' in agent else policy.agent_model.state_dict(), model_path)
        if args.log_wandb:
            wandb.save(model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_wandb", action='store_true', help="Log to Weights and Biases")
    parser.add_argument("--load_model", type=str, help="Path to folder containing the model to load")
    parser.add_argument("--render", action='store_true', help="Render the environment")
    parser.add_argument("--output_dir", type=str, help="Directory to save models")
    args = parser.parse_args()

    train(args)
