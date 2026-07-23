import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dql import *



#-------------------------------TRAINING-HYPERPARAMETERS--------------------------------

BATCH_SIZE = 128    # BATCH_SIZE is the number of transitions sampled from the replay buffer
GAMMA = 0.99        # GAMMA is the discount factor as mentioned in the previous section
EPS_START = 0.9     # EPS_START is the starting value of epsilon
EPS_END = 0.01      # EPS_END is the final value of epsilon
EPS_DECAY = 2500    # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
TAU = 0.005         # TAU is the update rate of the target network
LR = 3e-4           # LR is the learning rate of the ``AdamW`` optimizer
EPISODES = 600      #The number of episodes to train for

GAME = "CartPole-v1"
RENDERMODE = "rgb_array"  #Could be "human", "ansi", "rgb_array", or None
MAXITERS = 50        #Max iterations for playing the game




def main():

    #----------------------------ENVIRONMENT-&-TRAINING-UTILITIES----------------------------

    env = gym.make(GAME, render_mode=RENDERMODE)
    #Make a visual environment:
    visual_env = gym.make(GAME, render_mode="human")
    # Get number of actions from gym action space
    n_actions = env.action_space.n
    # Get the number of state observations
    state, info = env.reset()
    n_observations = len(state)

    #Create the agent:
    dql_agent = DQLAgent(env, n_observations, n_actions)

    print("\nDEEP Q-LEARNING FOR CARTPOLE\n-------------------------------------------------------------------------------------")
    choice = input("\nTo run the program using pre-trained weights, enter '1'. To train the model yourself, enter '2' (this will take ~9min):\n> ")

    if choice.strip() == '2':
        #Train the QL Agent and get the rewards:
        print("\nTRAINING (NO VISUAL)\n-------------------------------------------------------------------------------------")
        rewards = dql_agent.train(EPISODES, BATCH_SIZE, GAMMA, EPS_START, EPS_END, EPS_DECAY, TAU, LR)
        dql_agent.save(rewards)

    
    rewards = dql_agent.load()

    #Plot results:
    plt.plot(rewards)
    plt.title("Training Rewards over Time")
    plt.xlabel("Step")
    plt.ylabel("Summed Reward")
    # plt.show()

    #Playing section:
    print("\nTRAINED AGENT PLAYING (POST-TRAINING - WITH VISUAL)\n-------------------------------------------------------------------------------------")
    dql_agent.setEnv(visual_env)
    dql_agent.play(EPISODES=3, MAXITERS=500)


main()
