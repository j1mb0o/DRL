import argparse
import numpy as np
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import numpy as np
from scipy.signal import savgol_filter




SaveAction = namedtuple('SavedAction', ['log_prob', 'value'])


class ActorCritic(nn.Module):
    def __init__(self, env, obs_type):
        super(ActorCritic, self).__init__()
        self.env = env

        if obs_type == 'vector':
          self.state_size = env.observation_space.shape[0]
        elif obs_type == 'pixel':
          self.state_size = self.state_size[0]*self.state_size[1]*self.state_size[2]

        self.action_size = env.action_space.n
        self.hidden_size = 64
        
        self.affine1 = nn.Linear(self.state_size, self.hidden_size)

        self.action_head = nn.Linear(self.hidden_size, self.action_size) # actor
        self.value_head = nn.Linear(self.hidden_size, 1) # critic

        self.saved_actions = []
        self.rewards = []
        self.gamma = 0.9
        self.alpha = 0.001
        self.eps = np.finfo(np.float32).eps.item()
        self.n_steps = 5


    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_prob = F.softmax(self.action_head(x), dim=-1)
        state_values = self.value_head(x)
        return action_prob, state_values


    def select_action(self, model, state):
        probs, state_value = model(state)
        m = torch.distributions.Categorical(probs)
        action = m.sample()

        model.saved_actions.append(SaveAction(m.log_prob(action), state_value))

        return action.item()

    def compute_returns(self, model):
        R = 0
        self.saved_actions = model.saved_actions
        self.policy_losses = []
        self.value_losses = []
        returns = []

        # BOOTSTRAP
        for t in reversed(range(len(self.saved_actions))):
            log_prob, value = self.saved_actions[t]
            reward = self.rewards[t]
            R = reward + self.gamma * R
            if t >= self.n_steps:
                # R -= self.gamma**self.n_steps * self.saved_actions[t-self.n_steps][1].detach().item()
                R += self.gamma**t * self.saved_actions[t-self.n_steps][1].detach().item()
            returns.insert(0, R)

        # NOT BOOTSTRAP
        # for r in model.rewards[::-1]:
        #     R = r + self.gamma * R
        #     returns.insert(0, R)
        
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)
        return returns

    def update(self, model, optimizer):
        self.returns = self.compute_returns(model)
        weight = 0.1
        self.saved_actions = model.saved_actions
        self.policy_losses = []
        self.value_losses = []
        for (log_prob, value), R in zip(self.saved_actions, self.returns):
            # BASELINE SUBTRACTION
            advantage = R - value.item()
            m = torch.exp(log_prob)
            entropy = -torch.sum(m * log_prob)

            self.policy_losses.append(-log_prob * advantage + weight * entropy)
            self.value_losses.append(F.mse_loss(value, torch.tensor([R])))

            # NOT BASELINE SUBTRACTION
            # m = torch.exp(log_prob)
            # entropy = -torch.sum(m * log_prob)
            # policy_loss = -log_prob * R + weight * entropy

            # self.policy_losses.append(policy_loss)
            # self.value_losses.append(F.mse_loss(value, torch.tensor([float(R)])))

        optimizer.zero_grad()
        loss = torch.stack(self.policy_losses).sum() + torch.stack(self.value_losses).sum()
        loss.backward()
        optimizer.step()
        
        del model.rewards[:]
        del model.saved_actions[:]


    def train(self, model, episodes):
      
      # print(f'Settings are:\nFilename-{args.filename}\nLR-{args.learning_rate}\nGamma-{args.gamma}\n')
      optimizer = optim.Adam(model.parameters(), lr=self.alpha)
      episode = 0
      self.episode_rewards=[]

      while episode <= episodes:
          state = torch.tensor(self.env.reset(), dtype=torch.float).flatten()
          total_rewards = 0
          done = False
          t = 0
          while not done:
              action = self.select_action(model, state)
              state, reward, done = env.step(action)
              state = torch.tensor(state, dtype=torch.float).flatten()

              model.rewards.append(reward)   
              total_rewards += reward


          model.update(model, optimizer)

          if episode % 100 == 0:
              print('Episode {}\t Total reward: {:.2f}'.format(episode, total_rewards))
          self.episode_rewards.append(total_rewards)
      
          episode += 1
  
      self.episode_rewards = savgol_filter(self.episode_rewards, window_length = 50, polyorder=2 )
      plt.plot(self.episode_rewards)
      plt.xlabel("Episode")
      plt.ylabel("Reward")
      plt.title("AC - Bootstrapping rewards over Episodes")
      plt.show()

      # np.save(args.filename, np.array(rep_rewards))
        