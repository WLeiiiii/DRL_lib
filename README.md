# DRL_lib

## Introduction

A Deep Reinforcement Learning library based on PyTorch.

## Requirements

- Python 3.10
- PyTorch 1.12.1
- Gymnasium 0.29.1
- CUDA 11.3
- ...

```
pip install -r requirements.txt
```

## TODO

- [x] [Q-Learning](algorithms/QLearning/q_learning.py)
- [x] [SARSA](algorithms/SARSA/sarsa.py)
- [x] [DQN](algorithms/DQN/dqn.py)
- [x] [Double DQN](algorithms/DQN/dqn.py)
- [x] [Dueling DQN](algorithms/DQN/dqn.py)
- [x] [Prioritized Experience Replay DQN](algorithms/DQN/dqn_per.py)
- [ ] Deep Deterministic Policy Gradient
- [ ] Twin Delayed Deep Deterministic Policy Gradient
- [ ] Soft Actor-Critic
- [ ] Twin Delayed Soft Actor-Critic
- [ ] Proximal Policy Optimization
- [ ] Trust Region Policy Optimization
- [ ] Asynchronous Advantage Actor-Critic
- [ ] ...

## Reference

1. [PyTorch](https://pytorch.org/)
2. [Spinning Up in Deep RL](https://openai.com/research/spinning-up-in-deep-rl)
3. [Gymnasium](https://gymnasium.farama.org/)
4. ...