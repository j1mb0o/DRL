#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
2021
By Thomas Moerland
"""

import numpy as np
from Environment import StochasticWindyGridworld
from Helper import softmax, argmax

class NstepQLearningAgent:

    def __init__(self, n_states, n_actions, learning_rate, gamma, n):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.n = n
        self.Q_sa = np.zeros((n_states,n_actions))
        
    def select_action(self, s, policy='egreedy', epsilon=None, temp=None):
        
        if policy == 'egreedy':
            if epsilon is None:
                raise KeyError("Provide an epsilon")
                
            # TO DO: Add own code
            p = np.random.uniform(0.0,1.0)
            a = np.random.randint(0,self.n_actions) if p < epsilon else argmax(self.Q_sa[s])
            
                
        elif policy == 'softmax':
            if temp is None:
                raise KeyError("Provide a temperature")
                
            # TO DO: Add own code
            # a = np.random.randint(0,self.n_actions) # Replace this with correct action selection
            a = np.random.choice(range(self.n_actions), p=softmax(self.Q_sa[s],temp))
            
        return a
    
    def update(self, states, actions, rewards, done):
        ''' states is a list of states observed in the episode, of length T_ep + 1 (last state is appended)
        actions is a list of actions observed in the episode, of length T_ep
        rewards is a list of rewards observed in the episode, of length T_ep
        done indicates whether the final s in states is was a terminal state '''
        # TO DO: Add own code
        terminal_state = -1
        if done:
            terminal_state = states[-1]
        for t in range(len(states)):
            m = min(self.n, len(states)-t)
            if states[t+m-1] is terminal_state :
                G_t = sum( (self.gamma ** i)*rewards[t+i] for i in range(m) )
        
            else:
                G_t = sum( (self.gamma**i) * rewards[t+i] + (self.gamma**m)*np.max(self.Q_sa[states[t+m-1]]) for i in range(m) )
            self.Q_sa[states[t],actions[t]] += self.learning_rate * (G_t - self.Q_sa[states[t],actions[t]])
        pass

def n_step_Q(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy='egreedy', epsilon=None, temp=None, plot=True, n=5):
    ''' runs a single repetition of an MC rl agent
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    pi = NstepQLearningAgent(env.n_states, env.n_actions, learning_rate, gamma, n)
    rewards = []

    # TO DO: Write your n-step Q-learning algorithm here!
    act = []
    states = []
    rew = []
    terminal_state = False
    i = 0
    while i < n_timesteps:
        act.clear()
        states.clear()
        rew.clear()
        s = env.reset()
        for t in range( max_episode_length):
            a = pi.select_action(s,policy,epsilon,temp)
            s, r, done = env.step(a)
            if i >= n_timesteps:
                break
            i+=1
            if plot:
               env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=0.05) # Plot the Q-value estimates during n-step Q-learning execution

            act.append(a)
            states.append(s)
            rew.append(r)
            rewards.append(r)
            terminal_state = done
            # print(i)

            if terminal_state:
                break
        pi.update(states,act,rew,terminal_state)
    
    return rewards 

def test():
    # n_timesteps = 100000
    n_timesteps = 50000
    # max_episode_length = 100
    max_episode_length = 150
    gamma = 1.0
    learning_rate = 0.1
    n = 10
    
    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = True

    rewards = n_step_Q(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy, epsilon, temp, False, n=n)
    # print("Obtained rewards: {}".format(rewards))    
    print("Obtained rewards: {}".format(len(rewards)))    
    
if __name__ == '__main__':
    test()
