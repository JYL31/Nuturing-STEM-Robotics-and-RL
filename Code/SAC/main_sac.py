"""
Title: actor_critic_continuous source code
Author: Phil Tabor
Date: 2021
Availability: https://github.com/philtabor/Youtube-Code-Repository/tree/master/ReinforcementLearning/PolicyGradient/SAC/tf2

"""

import gym
import numpy as np
from sac_tf2 import Agent
from cartpole_continuous import Continuous_CartPoleEnv

if __name__ == '__main__':
    env = Continuous_CartPoleEnv()
    agent = Agent(input_dims=env.observation_space.shape, env=env,
            n_actions=env.action_space.shape[0])
    n_games = 100

    score_history = []


    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.remember(observation, action, reward, observation_, done)
            agent.learn()
            observation = observation_
        score_history.append(score)

        print('episode ', i, 'score %.1f' % score)#, 'avg_score %.1f' % avg_score)


#%%

import pandas as pd    
                  
df = pd.DataFrame(score_history)
df.to_csv('SAC_results_2.csv', index=False)
