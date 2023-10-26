
from catch import Catch
from final.AC-Agent import ActorCritic
# parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
# parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
#                     help='discount factor (default: 0.99)')
# parser.add_argument('--log-interval', type=int, default=100, metavar='N',
#                     help='interval between training status logs (default: 10)')
# parser.add_argument('--learning-rate', type=float, default=1e-3, metavar='LR',help='learning rate (default: 1e-3)')
# parser.add_argument('--filename', type=str, default='ac_agent', metavar='F')
# args = parser.parse_args()




if __name__ == '__main__':

    obs_type = 'vector'
    env = Catch(rows=7, columns=7, speed=1.0, max_steps=250, max_misses=10, observation_type=obs_type, seed=None)
   
    episodes = 2000

    agentAC_BS = ActorCritic(env, obs_type)
    agentAC_BS.train(agentAC_BS, episodes)

