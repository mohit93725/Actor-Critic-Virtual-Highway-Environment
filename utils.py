import torch

def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    values = values + [0]
    gae, returns = 0, []
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t+1] - values[t]
        gae = delta + gamma * lam * gae
        returns.insert(0, gae + values[t])
    return torch.tensor(returns)
