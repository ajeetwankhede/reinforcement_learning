# Problem: Reach goal with minimum steps
# Gird: 4x4
#  _ _ _ _
# |S|_|_|_|
# |_|_|_|_|
# |_|_|_|_| 
# |_|_|_|G|
# 
# States: 16 (from (1,1) to (4,4))
# Actions: 4 - Up, Down, Left, Right
# Transitions: Mostly deterministic, but with a small chance of moving randomly
# Rewards: -1 for each move, +10 for reaching goal (4,4)
# Discount Factor (y): 0.9

import numpy as np

def value_iteration(grid_size, goal_reward, move_reward, goal, discount_factor, threshold):
    # Initialize value grid
    value_grid = np.zeros(grid_size)
    value_grid[goal] = goal_reward
    # print(value_grid)

    # Possible actions
    actions = [(0,1), (1,0), (0,-1), (-1,0)] # Right, Down, Left, Up

    # Function to get valid next states
    def get_valid_states(state):
        valid_states = []
        for action in actions:
            next_state = (state[0] + action[0], state[1] + action[1])
            if 0 <= next_state[0] < grid_size[0] and 0 <= next_state[1] < grid_size[1]:
                valid_states.append(next_state)
        return valid_states
    
    # Iterate until convergence
    while True:
        delta = 0
        new_value_grid = np.copy(value_grid)
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                if (i,j) != goal:
                    v = value_grid[i,j]
                    valid_states = get_valid_states((i,j))
                    # Update Rule: V_k+1 = max over all actions [ R(s,a,s') + gamma * V_k(s,a) ]
                    new_value_grid[i,j] = max([move_reward + discount_factor * value_grid[next_state] for next_state in valid_states])
                    delta = max(delta, abs(v - new_value_grid[i,j]))
        value_grid = new_value_grid
        # print(value_grid)

        # Check for convergence
        if delta < threshold:
            break

    return value_grid

def derive_policy(value_grid, grid_size, goal):
    policy_grid = np.full(grid_size, ' ')
    actions = ['R', 'D', 'L', 'U']
    action_offsets = [(0,1), (1,0), (0,-1), (-1,0)]

    # Function to get valid next states with actions
    def get_valid_states_with_actions(state):
        valid_states = []
        for idx, action in enumerate(action_offsets):
            next_state = (state[0] + action[0], state[1] + action[1])
            if 0 <= next_state[0] < grid_size[0] and 0 <= next_state[1] < grid_size[1]:
                valid_states.append((next_state, actions[idx]))
        return valid_states

    for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                if (i,j) != goal:
                    valid_states = get_valid_states_with_actions((i,j))
                    best_action = max(valid_states, key=lambda x: value_grid[x[0]])
                    policy_grid[i,j] = best_action[1]

    return policy_grid

# Grid parameters
grid_size = (4,4)
goal_reward = 10
move_reward = -1
goal = (3,3)
discount_factor = 0.9
threshold = 0.01

# Perform Value Iteration
value_grid = value_iteration(grid_size, goal_reward, move_reward, goal, discount_factor, threshold)
print(value_grid)

# Derive the policy from the value grid
policy_grid = derive_policy(value_grid, grid_size, goal)
print(policy_grid)
