# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 14:24:44 2023

@author: Jiayuan Liu
"""
import numpy as np
from scipy import linalg
from cartpole_continuous import Continuous_CartPoleEnv
from matplotlib import pyplot as plt

g = 9.8
mc = 1.0
mp = 0.1
lp = 0.5  # actually half the pole's length


# state matrix
a = g/(lp*(4.0/3 - mp/(mp+mc)))
A = np.array([[0, 1, 0, 0],
              [0, 0, a, 0],
              [0, 0, 0, 1],
              [0, 0, a, 0]])

# input matrix
b = -1/(lp*(4.0/3 - mp/(mp+mc)))
B = np.array([[0], [1/(mp+mc)], [0], [b]])

R = np.eye(1, dtype=int)          # choose R (weight for input)
Q = np.eye(4, dtype=int)        # choose Q (weight for state)

# solve ricatti equation
P = linalg.solve_continuous_are(A, B, Q, R)

# calculate optimal controller gain
K = np.dot(np.linalg.inv(R),
           np.dot(B.T, P))

def apply_state_controller(K, x):
    # feedback controller
    u = -np.dot(K, x)   # u = -Kx
    u = np.clip(u, -10, 10)
    action = u / 10
    return action
    
# get environment
env = Continuous_CartPoleEnv(render_mode='human')
state = env.reset()
actions = []
states = [state]
rewards = []
for i in range(1000):
    env.render()
    
    # get force direction (action) and force value (force)
    action = apply_state_controller(K, state)
    
    # apply action
    state, reward, done, _ = env.step(action)
    actions.append(action)
    states.append(state)
    rewards.append(reward)
    if done:
        print(f'Terminated after {i+1} iterations.')
        break

env.close()

u = np.dot(10, actions)
u = np.insert(u,0,0)

plt.figure(0)
plt.plot(u)
plt.xlabel('Number of steps')
plt.ylabel('Force (N)')

plt.figure(1)
plt.plot(states)
plt.xlabel('Number of steps')
plt.legend(['Cart Position', 'Cart Velocity', 'Pole Angle', 'Pole Angular Velocity'])