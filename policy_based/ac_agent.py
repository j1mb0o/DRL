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

from model import Policy
from catch import Catch

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--learning-rate', type=float, default=1e-3, metavar='LR',help='learning rate (default: 1e-3)')
parser.add_argument('--filename', type=str, default='ac_agent', metavar='F')
args = parser.parse_args()



env = Catch()
env.reset()
obs_shape = np.ndarray(env.observation_space.shape)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Will use single network for both actor and critic with two heads
# https://ai.stackexchange.com/a/25060
model = Policy(state_size=np.ndarray.flatten(obs_shape).shape[0]).to(device)
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

# https://neurostars.org/t/question-about-np-finfo-float-eps/15154/2
eps = np.finfo(np.float32).eps.item()


SaveAction = namedtuple('SavedAction', ['log_prob', 'value'])

def action_select(state):
    state = torch.from_numpy(np.ndarray.flatten(state)).float().to(device)
    probs, state_value = model(state)

    m = Categorical(probs)

    action = m.sample()

    model.saved_actions.append(SaveAction(m.log_prob(action), state_value))

    return action.item()

def optimize_net():
    R = 0
    saved_actions = model.saved_actions
    policy_losses = []
    value_losses = []
    returns = []
    rewards = model.rewards
    entropy_weight = 0.51
    # for
    for rew in rewards[:-10:-1]:
        # Calculating the discounted reward
        
        R = rew + args.gamma * R
        returns.insert(0, R)

    returns = torch.tensor(returns)
    # Normilizing the data 
    returns = (returns - returns.mean()) / (returns.std() + eps)

    for (log_prob, value), R in zip(saved_actions, returns):

        policy_losses.append(-log_prob * R)
        value_losses.append(F.mse_loss(value, torch.tensor([R])))

    # optimizer.zero_grad()
    # loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

    optimizer.zero_grad()
    loss = (torch.stack(policy_losses).sum() + torch.stack(value_losses).sum() ) * entropy_weight
    loss.backward()
    optimizer.step()

    del model.rewards[:]
    del model.saved_actions[:]


def plot_res(rewards):
    print(rewards.shape)

def main():
    # run_rew = 10

    num_episodes = 2000
    max_num_steps = 1000
    num_rep  = 3
    print(f'Settings are:\nFilename-{args.filename}\nLR-{args.learning_rate}\nGamma-{args.gamma}\n')
    
    rep_rewards = []
    for rep in range(num_rep):
        episode_reward = []
        
        # Resetting the model parameters
        for layer in model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        for episode in range(num_episodes):
            state = env.reset()
            ep_reward = 0

            for t in range(max_num_steps):
                action = action_select(state)
                state, reward, done = env.step(action)

                model.rewards.append(reward)   
                ep_reward += reward

                if done:
                    break
            episode_reward.append(ep_reward)
            optimize_net()

            if episode % args.log_interval == 0:
                print('Repetistion {} Episode {}\tLast reward: {:.2f}'.format(rep+1, episode, ep_reward))
        rep_rewards.append(episode_reward)

    np.save(args.filename, np.array(rep_rewards))
        

if __name__ == '__main__':
    main()