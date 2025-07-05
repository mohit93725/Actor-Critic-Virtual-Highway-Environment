# A2C on `highway-v0` using PyTorch

This repository contains a clean, modular implementation of the **Advantage Actor-Critic (A2C)** algorithm on the [`highway-v0`](https://github.com/eleurent/highway-env) environment using **PyTorch** and **Generalized Advantage Estimation (GAE)**.

---

## Algorithm Overview

The **Advantage Actor-Critic (A2C)** method is a policy gradient algorithm that simultaneously learns:
- A **policy** (actor) to select actions
- A **value function** (critic) to estimate future rewards

This implementation uses:
- Generalized Advantage Estimation (GAE)
- Entropy regularization with annealed β
- Fully-connected neural networks with LayerNorm and Dropout

---

##  Project Structure

a2c_highway/
├── model.py # Actor-Critic neural network definition
├── utils.py # GAE computation and other helpers
├── train.py # Core training loop
├── main.py # Entry point to start training

Run Training

python main.py

    Trains an A2C agent on the highway-v0 environment.

    Saves model weights at the final episode: model_epXXX.pt.
