# RL-Assignment2-Part1-GridWorld
This repository contains the solution for Part 1 of Assignment 2 in the Reinforcement Learning course. It implements [Monte Carlo or other specified methods on a 5x5 gridworld environment 
# RL-Assignment2-Part1-GridWorld

This repository contains the solution for Part 1 of Assignment 2 in the Reinforcement Learning course. It implements dynamic programming methods to evaluate and optimize policies on a 5x5 gridworld with special states: blue (0,1, +5 reward, jumps to red), green (0,4, +2.5 reward, 50% jump to yellow/red), red (3,2), and yellow (4,4). The environment has deterministic normal moves (0 reward), off-grid penalties (-0.5), and a discount factor of γ = 0.95.

## Features
- **Environment**: 5x5 gridworld with stochastic transitions for blue and green states.
- **Methods**:
  - Question 1: Evaluates a random policy (equiprobable moves) using explicit Bellman equations and iterative policy evaluation.
  - Question 2: Finds the optimal policy using policy iteration, value iteration, and Bellman optimality (via value iteration).
- **Outputs**: Value function heatmaps and policy plots (saved as PNGs) for each method, showing state values and optimal actions.
- **Reproducibility**: Random seed set to 42; convergence threshold ε = 1e-4; max iterations 1000.

## Requirements
- Python 3
- `numpy`, `matplotlib`, `seaborn`, `scipy` (`pip install numpy matplotlib seaborn scipy`)

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/RL-Assignment2-Part1-GridWorld.git
