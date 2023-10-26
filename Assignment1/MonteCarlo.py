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

class MonteCarloAgent:

    def __init__(self, n_states, n_actions, learning_rate, gamma):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
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
        
    def update(self, states, actions, rewards):
        ''' states is a list of states observed in the episode, of length T_ep + 1 (last state is appended)
        actions is a list of actions observed in the episode, of length T_ep
        rewards is a list of rewards observed in the episode, of length T_ep
        done indicates whether the final s in states is was a terminal state '''
        # TO DO: Add own code
        G_t = 0
        t = len(states) - 1
        for i in range(t,-1,-1):
            G_t = rewards[i] + self.gamma*G_t
            self.Q_sa[states[i], actions[i]] += self.learning_rate*(G_t - self.Q_sa[states[i], actions[i]])
        pass

def monte_carlo(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy='egreedy', epsilon=None, temp=None, plot=True):
    ''' runs a single repetition of an MC rl agent
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    pi = MonteCarloAgent(env.n_states, env.n_actions, learning_rate, gamma)
    rewards = []

    # TO DO: Write your Monte Carlo RL algorithm here!
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
            if i >= n_timesteps:
                break
            a = pi.select_action(s,policy,epsilon,temp)
            s, r, done = env.step(a)
            
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
        pi.update(states,act,rew)
  
    # if plot:
    #    env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=0.1) # Plot the Q-value estimates during Monte Carlo RL execution

    return rewards 
    
def test():
    n_timesteps = 10000
    max_episode_length = 100
    gamma = 1.0
    learning_rate = 0.1

    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = True

    rewards = monte_carlo(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy, epsilon, temp, False)
    print("Obtained rewards: {}".format(rewards))  
            
if __name__ == '__main__':
    test()
