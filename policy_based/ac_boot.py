import torch
import torch.nn.functional as F
from torch.optim import Adam
from catch import Catch
from collections import namedtuple
import numpy as np
from torch.distributions import Categorical


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'log_prob'))


class Actor(torch.nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super(Actor, self).__init__()
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(state_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, action_size)
        )


    def forward(self, x):
        logits = self.actor(x)
        action_probs = F.softmax(logits, dim=-1)
        return action_probs


    def select_action(network, state):
        #  Selects an action given current state
        
        
        #convert state to float tensor, add 1 dimension, allocate tensor on device
        state = torch.from_numpy(np.ndarray.flatten(state)).to(device)
        
        #use network to predict action probabilities
        action_probs = network(state)
        state = state.detach()
        
        #sample an action using the probability distribution
        m = Categorical(action_probs)
        action = m.sample()
        
        #return action
        return action.item(), m.log_prob(action)
        
        
        
class Critic(torch.nn.Module):
    def __init__(self, state_size, hidden_size):
        super(Critic, self).__init__()
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(state_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        critic_value = self.critic(x)
        return critic_value
    


def main():
    env = Catch()

    gamma = 0.99
    hidden_size = 128
    n_episodes = 1000
    trace_steps = 10000
    n_steps = 10
    lr = 1e-3

    state = env.reset()
    obs_shape = np.ndarray(env.observation_space.shape)
    
    actor = Actor(state_size=np.ndarray.flatten(obs_shape).shape[0], action_size= env.action_space.n,  hidden_size= hidden_size)
    critic = Critic(state_size=np.ndarray.flatten(obs_shape).shape[0], hidden_size= hidden_size)
    actor_optimizer = Adam(actor.parameters(), lr=lr)
    critic_optimizer = Adam(critic.parameters(), lr=lr)

    scores = []
    for episode in range(n_episodes):
        print(f'Episode {episode} of {n_episodes}')
        T = []
        ep_score = 0

        state = env.reset()
        done = False

        # Generate the trace       
        for step in range(trace_steps):
            # print(f'Step {step} of {n_steps}')
            action, log_prob = actor.select_action(state)
            next_state, reward, done = env.step(action)
            print(f'Action: {action}, Log_prob: {log_prob}')
            T.append(Transition(state, action, reward, log_prob))
            ep_score += reward
            if done:
                break
            state = next_state
            # print(f'Reward: {reward}')
        # print(f'Episode score: {score}')
        scores.append(ep_score)
        states = [x.state for x in T]
        actions = [x.action for x in T]
        rewards = [x.reward for x in T]
        log_probs = [x.log_prob for x in T]

        # G 
        Q_n = []
        V_fi = []
        for t in range(len(T)):
            G = 0
            if t + n_steps < len(T):
                for k in range(t, t+n_steps):
                    G += (gamma**(k-t))*rewards[k] - critic(torch.from_numpy(np.ndarray.flatten(states[k])))
                    V_fi.append(critic(torch.from_numpy(np.ndarray.flatten(states[k]))))
            else:
                for k in range(t, len(T)):
                    G += (gamma**(k-t))*rewards[k] - critic(torch.from_numpy(np.ndarray.flatten(states[k])))
                    V_fi.append(critic(torch.from_numpy(np.ndarray.flatten(states[k]))))

            
            
    

        # Update the actor and critic
        

    action_dist= actor(torch.from_numpy(np.ndarray.flatten(state)).to(device))
    critic_value = critic(torch.from_numpy(np.ndarray.flatten(state)).to(device))


if __name__ == '__main__':
    main()