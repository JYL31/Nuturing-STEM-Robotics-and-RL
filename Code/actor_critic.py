# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 12:52:52 2022

@author: Jiayuan Liu

Title: actor_critic_continuous source code
Author: Phil Tabor
Date: 2019
Availability: https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/PolicyGradient/actor_critic/actor_critic_continuous.py

"""
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from cartpole_continuous import Continuous_CartPoleEnv


class GenericNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, fc3_dims, n_actions):
        super(GenericNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims # layer 1 dim
        self.fc2_dims = fc2_dims # layer 2 dim
        self.fc3_dims = fc3_dims # layer 3 dim
        self.n_actions = n_actions
        
        # network structure
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.fc3_dims)
        self.fc4 = nn.Linear(self.fc3_dims, self.n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu') # to use GPU/CPU
        self.to(self.device)
    
    # forward pass
    def forward(self, observation):
        state = T.tensor(observation, dtype=T.float).to(self.device)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x

class Agent(object):
    def __init__(self, alpha, beta, input_dims, gamma=0.99, n_actions=2,
                 layer1_size=256, layer2_size=168, layer3_size=64, n_outputs=1):
        self.gamma = gamma
        self.log_probs = None
        self.n_outputs = n_outputs
        self.actor = GenericNetwork(alpha, input_dims, layer1_size, 
                                    layer2_size, layer3_size, n_actions=n_actions)
        self.critic = GenericNetwork(beta, input_dims, layer1_size, 
                                     layer2_size, layer3_size, n_actions=1)
    
    # choose action basing on the Gaussian probability distribution that is predicted by the network
    def choose_action(self, observation):
        mu, sigma  = self.actor.forward(observation).to(self.actor.device)
        sigma = T.exp(sigma)
        action_probs = T.distributions.Normal(mu, sigma)
        probs = action_probs.sample(sample_shape=T.Size([self.n_outputs]))
        self.log_probs = action_probs.log_prob(probs).to(self.actor.device)
        action = T.tanh(probs)

        return action.item()

    def learn(self, state, reward, new_state, done):
        self.actor.optimizer.zero_grad() # make the gradient zero to ignore effects from previous episodes
        self.critic.optimizer.zero_grad()

        critic_value_ = self.critic.forward(new_state) # forward pass
        critic_value = self.critic.forward(state)
        reward = T.tensor(reward, dtype=T.float).to(self.actor.device)
        delta = ((reward + self.gamma*critic_value_*(1-int(done))) - critic_value) # temporal difference loss (advantage)

        actor_loss = -self.log_probs * delta
        critic_loss = delta**2

        (actor_loss + critic_loss).backward() # back propagation
        
        self.actor.optimizer.step()
        self.critic.optimizer.step()

def train():
    rewards = []
    env = Continuous_CartPoleEnv()
    agent = Agent(alpha=0.00005, beta=0.00001, input_dims=[4], gamma=0.99,
                  layer1_size=256, layer2_size=168, layer3_size=64)
    run = 0
    while run < 1500:
        terminal = False
        run += 1
        state = env.reset()
        step = 0
        while not terminal:
            #env.render()
            action = np.array(agent.choose_action(state)).reshape((1,))
            state_next, reward, terminal, info = env.step(action)
            #agent.learn(state, reward, state_next, terminal)
            reward = reward if not terminal else -reward
            agent.learn(state, reward, state_next, terminal)
            state = state_next
            if terminal:
                print("Run: " + str(run) + ", reward: " + str(step))
                rewards.append(step)
                break
            step += 1
    return rewards


#%%
import pandas as pd    
                  
rewards = train()
df = pd.DataFrame(rewards)
df.to_csv('A2C_results_1.csv', index=False)

#plt.plot(rewards)