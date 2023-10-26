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



env = Catch()
state = env.reset()
obs_shape = np.ndarray(env.observation_space.shape)
model = Policy(state_size=np.ndarray.flatten(obs_shape).shape[0])

state = torch.from_numpy(np.ndarray.flatten(state)).float()
print(model(state))
print(model(state))

# model.reset_parameters()
for layer in model.children():
   if hasattr(layer, 'reset_parameters'):
       layer.reset_parameters()
print(model(state))