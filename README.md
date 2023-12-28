# Reinforcement learning

## Project for reinforcement learning algorithms

1. [MDP Value Iteration](https://github.com/ajeetwankhede/reinforcement_learning/blob/main/mdp_value_iteration.py):
```bash
Update equation: 
V_k+1(s,a) = max a [ R + gamma * V_k(s,a) ]
```

2. [Tabluar Q-learning](https://github.com/ajeetwankhede/reinforcement_learning/blob/main/frozen_lake/frozen_lake_q_learning.py):
```bash
Update equation:
Q_k+1(s,a) = Q_k(s,a) + alpha * [ R + gamma * max a [ Q_k(s',a) ] - Q_k(s,a) ]
```
Reference links:
https://gymnasium.farama.org/tutorials/training_agents/FrozenLake_tuto/ 


3. [DQN](https://github.com/ajeetwankhede/reinforcement_learning/blob/main/cart_pole/cart_pole_dqn.py):
```bash
Update equation:
Q_predicted = Q_net(s,a)
Q_expected = R + gamma * max a [ Q_target_net(s',a) ]
NN loss fcn = (Q_predicted, Q_expected)
```
Reference links:
https://blog.gofynd.com/building-a-deep-q-network-in-pytorch-fa1086aa5435
