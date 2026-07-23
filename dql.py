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
from tqdm import tqdm
import numpy as np
import pickle

#-------------------------------Q-NETWORK--------------------------------

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    
#--------------------------------Q-AGENT---------------------------------

class DQLAgent():

    def __init__(self, env, n_observations, n_actions):
    
        #Check for GPU
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "cpu"
        )
        
        #New instance variables:
        self.env = env
        self.policy_net = DQN(n_observations, n_actions).to(self.device)
        self.target_net = DQN(n_observations, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=3e-4, amsgrad=True)
        self.memory = ReplayMemory(10000)
        self.steps_done = 0
        self.saved_rewards = None


    def setEnv(self, env):
         self.env = env



    def save(self, rewards, path="weights.pkl"):

        self.saved_rewards = rewards
        
        data = {"policy_state": self.policy_net.state_dict(), 
                "target_state": self.target_net.state_dict(),
                "rewards": self.saved_rewards}
        
        with open(path, "wb") as file:
            pickle.dump(data, file)



    def load(self, path="weights.pkl"):
        with open(path, "rb") as file:
            data = pickle.load(file)

            self.policy_net.load_state_dict(data["policy_state"])
            self.target_net.load_state_dict(data["target_state"])
            self.saved_rewards = data["rewards"]
            return self.saved_rewards




    def train(self, EPISODES, BATCH_SIZE, GAMMA, EPS_START, EPS_END, EPS_DECAY, TAU, LR):

        episode_rewards = []

        

        print("Training for", EPISODES, "episode(s):")

        with tqdm(range(EPISODES)) as pbar:
            for i_episode in pbar:

                # Initialize the environment and get its state
                state, info = self.env.reset()
                state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                total_reward = 0.0

                for t in count():
                    action = self.select_action(state, EPS_START, EPS_END, EPS_DECAY)
                    observation, reward, terminated, truncated, _ = self.env.step(action.item())
                    total_reward += reward
                    reward = torch.tensor([reward], device=self.device)
                    done = terminated or truncated

                    if terminated:
                        next_state = None
                    else:
                        next_state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)

                    # Store the transition in memory
                    self.memory.push(state, action, next_state, reward)

                    # Move to the next state
                    state = next_state

                    # Perform one step of the optimization (on the policy network)
                    self.optimize_model(BATCH_SIZE, GAMMA)

                    # Soft update of the target network's weights
                    # θ′ ← τ θ + (1 −τ )θ′
                    target_net_state_dict = self.target_net.state_dict()
                    policy_net_state_dict = self.policy_net.state_dict()
                    for key in policy_net_state_dict:
                        target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
                    self.target_net.load_state_dict(target_net_state_dict)

                    if done:
                        episode_rewards.append(total_reward)
                        #plot_durations()
                        pbar.set_postfix({"REWARD": f"{total_reward:.1f}"})
                        break
                
        print('Complete!')
        #Store for loading:
        self.saved_rewards = episode_rewards
        return episode_rewards


    def play(self, EPISODES=5, MAXITERS=1000):
        total_epochs, total_penalties = 0, 0

        print("Playing with", MAXITERS, "iterations for", EPISODES, "episodes...")

        for _ in tqdm(range(EPISODES)):

            state, _ = self.env.reset()
            epochs, reward = 0, 0
            done = False
            total_reward = 0.0
            
            while not done and epochs < MAXITERS:

                #Convert the state to a PyTorch tensor
                state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

                q_values = self.policy_net(state)
                action = torch.argmax(q_values, dim=1).item()
                state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                self.env.render()

                total_reward += reward
                epochs += 1
            #print(total_reward)
            total_epochs += epochs

        print(f"Results after {EPISODES} episodes:")
        print(f"Average timesteps per episode: {total_epochs / EPISODES}")
        print(f"Average penalties per episode: {total_penalties / EPISODES}")

    
    def select_action(self, state, EPS_START, EPS_END, EPS_DECAY):
        
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long)
        

    def optimize_model(self, BATCH_SIZE, GAMMA):
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()


#-------------------------------REPLAY-MEMORY--------------------------------


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
