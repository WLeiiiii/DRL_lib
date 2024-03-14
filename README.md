# DRL_lib

Status: Under development...

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
- [x] [Monte Carlo](algorithms/MonteCarlo/single_step_monte_carlo.py)
- [x] [DQN](algorithms/DQN/dqn.py)
- [x] [Double DQN](algorithms/DQN/dqn.py)
- [x] [Dueling DQN](algorithms/DQN/dqn.py)
- [x] [Prioritized Experience Replay DQN](algorithms/DQN/dqn_per.py)
- [x] [DQN with NoisyNet](algorithms/DQN/dqn_noisy.py)
- [x] [Distributional DQN(C51)](algorithms/DQN/dqn_c51.py)
- [x] [Deep Deterministic Policy Gradient(DDPG)](algorithms/DDPG/ddpg.py)
- [x] [Twin Delayed Deep Deterministic Policy Gradient(TD3)](algorithms/DDPG/td3.py)
- [x] [Soft Actor-Critic(Discrete)](algorithms/SAC/sac_discrete.py)
- [x] [Soft Actor-Critic(Continuous)](algorithms/SAC/sac_continuous.py)
- [ ] Twin Delayed Soft Actor-Critic
- [x] [Proximal Policy Optimization](algorithms/PPO/ppo.py)
- [ ] Trust Region Policy Optimization
- [ ] Asynchronous Advantage Actor-Critic
- [ ] ...

## Reference

1. [PyTorch](https://pytorch.org/)
2. [Spinning Up in Deep RL](https://openai.com/research/spinning-up-in-deep-rl)
3. [Gymnasium](https://gymnasium.farama.org/)
4. ...