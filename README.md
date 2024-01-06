# Reinforcement learning

## Project for reinforcement learning algorithms

1. [MDP Value Iteration](https://github.com/ajeetwankhede/reinforcement_learning/blob/main/mdp_value_iteration.py):
```bash
Update equation:
V_k+1(s) = max_over_a [ R + gamma * V_k(sn) ]
```

2. [Tabluar Q-learning](https://github.com/ajeetwankhede/reinforcement_learning/blob/main/frozen_lake/frozen_lake_q_learning.py):
```bash
Update equation:
Q_k+1(s,a) = Q_k(s,a) + alpha * [ R + gamma * max_over_a [ Q_k(sn,a) ] - Q_k(s,a) ]
```
Reference links:
https://gymnasium.farama.org/tutorials/training_agents/FrozenLake_tuto/ 


3. [DQN](https://github.com/ajeetwankhede/reinforcement_learning/blob/main/cart_pole/cart_pole_dqn.py):
```bash
Update equation:
Q_predicted = Q_net(s,a)
Q_expected = R + gamma * max_over_a [ Q_target_net(sn,a) ]
MSE loss fcn = (Q_predicted, Q_expected)
```
Reference links:
https://blog.gofynd.com/building-a-deep-q-network-in-pytorch-fa1086aa5435


4. [REINFORCE](https://github.com/ajeetwankhede/reinforcement_learning/blob/main/cart_pole/cart_pole_reinforce.py)
```bash
Update equation:
Policy Update = sum over t=0:T [ log( pi(st,at) ) * discounted sum of future Rt ]
```
Reference links:
https://gymnasium.farama.org/tutorials/training_agents/reinforce_invpend_gym_v26/ 

5. [Vanila Policy Gradient]()
```bash
Update equation:
Baseline: Value function
Policy Update = sum over t=0:T [ log( pi(st,at) ) * ( discounted sum of future Rt - basline ) ]
```

6. [A2C]()
```bash
Update equation:
Baseline: Value function
Policy Update = sum over t=0:T [ log( pi(st,at) ) * ( discounted sum of future Rt - basline ) ]
```
Reference links:
https://gymnasium.farama.org/tutorials/gymnasium_basics/vector_envs_tutorial/ 

## [References](https://github.com/ajeetwankhede/reinforcement_learning/blob/main/references.md)