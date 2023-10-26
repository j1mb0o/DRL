#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
2022
By Thomas Moerland
"""

from time import sleep
import numpy as np
from Environment import StochasticWindyGridworld
from Helper import argmax

class QValueIterationAgent:
    ''' Class to store the Q-value iteration solution, perform updates, and select the greedy action '''

    def __init__(self, n_states, n_actions, gamma, threshold=0.01):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.Q_sa = np.zeros((n_states,n_actions))
        
    def select_action(self,s):
        ''' Returns the greedy best action in state s ''' 
        # TO DO: Add own code
        # env = StochasticWindyGridworld(initialize_model=True)
        
        # a = np.random.randint(0,self.n_actions) # Replace this with correct action selection
        a = argmax(self.Q_sa[s])
        return a
        
    def update(self,s,a,p_sas,r_sas):
        ''' Function updates Q(s,a) using p_sas and r_sas '''
        # TO DO: Add own code
        Q_sa_sum = 0
        for s_next in range(self.n_states):
            Q_sa_sum += p_sas[s_next] * (r_sas[s_next] + self.gamma * np.max(self.Q_sa[s_next]))
        self.Q_sa[s,a] = Q_sa_sum
        # self.Q_sa[s,a] = np.sum(p_sas*(r_sas + self.gamma * np.max(self.Q_sa[s,a])))
        pass
    
    
    
def Q_value_iteration(env, gamma=1.0, threshold=0.001):
    ''' Runs Q-value iteration. Returns a converged QValueIterationAgent object '''
    
    QIagent = QValueIterationAgent(env.n_states, env.n_actions, gamma)    
    # TO DO: IMPLEMENT Q-VALUE ITERATION HERE
    i = 0
    # while delta > threshold:
    while True:
        max_error = 0.0
        env.render(Q_sa=QIagent.Q_sa,plot_optimal_policy=True,step_pause=0.1)
        for s in range(QIagent.n_states):
            # print(f'{s=}')
            for a in range(QIagent.n_actions):
                x = QIagent.Q_sa[s,a]
                p_sas, r_sas = env.model(s,a)
                QIagent.update(s,a, p_sas, r_sas)
                max_error = np.maximum(max_error, np.abs(x - QIagent.Q_sa[s,a]))
                
                # print(f'{delta=}')
    # Plot current Q-value estimates & print max error
        # print("Q-value iteration, iteration {}, max error {}".format(i,max_error))
        i += 1
        if max_error < threshold:
            break


    return QIagent

def experiment():
    gamma = 1.0
    threshold = 0.001
    env = StochasticWindyGridworld(initialize_model=True)
    env.render()
    sleep(2)
    QIagent = Q_value_iteration(env,gamma,threshold)
    
    rewards = []
    temp_rewards = []
    # View optimal policy
    for _ in range(100):
        temp_rewards.clear()
        done = False
        s = env.reset()
        while not done:
            a = QIagent.select_action(s)
            
            s_next, r, done = env.step(a)
            env.render(Q_sa=QIagent.Q_sa,plot_optimal_policy=True,step_pause=0.1)
            s = s_next

    # TO DO: Compute mean reward per timestep under the optimal policy
            temp_rewards.append(r)
        # print(temp_rewards,len(temp_rewards))
        rewards.append(sum(temp_rewards)/len(temp_rewards))
    print("Mean reward per timestep under optimal policy: {}".format(sum(rewards)/len(rewards)))

if __name__ == '__main__':
    experiment()
