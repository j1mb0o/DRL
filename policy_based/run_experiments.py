import os

lr = [1e-3, 2e-3, 1e-4]
gamma = [0.99, 0.9, 0.95]

for l in lr:
    for g in gamma:
        filename = "ac_agent_{}_{}.npy".format(l, g)
        os.system("python ac_agent.py --learning-rate {} --gamma {} --filename {}".format(l, g, filename))