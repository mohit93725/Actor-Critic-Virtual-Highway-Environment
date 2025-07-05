import gymnasium as gym
import torch
import torch.nn.functional as F
from torch.optim import Adam
import highway_env
import random
from torch.distributions import Categorical

from model import ActorCritic
from utils import compute_gae


def train():
    
    env = gym.make('highway-v0')

    obs, _ = env.reset()

    model = ActorCritic(input_dim=25, output_dim=5)

    episodes = 1000;
    gamma = 0.9;
    max_steps = 50;
    beta_1 = 0.05
    beta_2 = 0.0001
    opt = Adam(model.parameters(), lr=5e-5)


    Net_Reward = []
    for i in range(episodes):
        log_probs, rewards, values = [], [], []
        obs, _ = env.reset();
        terminated = False
        step = 0
        
        while not terminated and step <max_steps:

            state = torch.tensor(obs).view(-1).unsqueeze(0)
            logits, value = model(state)
            probs = F.softmax(logits, dim = -1)
            dist = Categorical(probs)
            entropy = dist.entropy().mean()
            action = dist.sample()
            log_prob = dist.log_prob(action)

            new_state, reward, terminated, _, _ = env.step(action.item())
            
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            step += 1

        values_tensor = [v.item() for v in values]
        G = compute_gae(rewards, values_tensor)

        advantage = G - torch.cat(values).squeeze(-1)
        advantage = (advantage - advantage.mean())/(advantage.std() + 1e-6)

        log_probs = torch.cat(log_probs).squeeze(-1)
        
        actor_loss = -log_probs * advantage
        critic_loss = 0.5 * (advantage**2)

        frac = i/episodes
        
        beta = beta_1 * (1-frac) + beta_2 * frac
        total_loss = (actor_loss + critic_loss - beta * entropy)
        total_loss = torch.mean(total_loss)
        Reward = sum(rewards)

        Net_Reward.append(Reward)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        opt.zero_grad()
        total_loss.backward()
        opt.step()

        if i%10 == 0:
            print(f" Ep {i}  Loss {total_loss:.3f} | Reward {Reward:.3f} | beta {beta:.4f} | Entropy {entropy:.3f}")

    torch.save(model.state_dict(), f"model_ep{i}.pt")
