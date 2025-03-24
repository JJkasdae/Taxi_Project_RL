from env import Env
from agent import Agent
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import matplotlib.pyplot as plt
import numpy as np
import os

class Train:
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
        self.agent.setUp()
        self.no_of_episodes = 500
        self.max_steps_per_episode = 50
        self.epsilon = 0.99
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.stop = False
        self.steps = 0
        self.total_loss = 0

        self.replay_buffer = deque(maxlen=2000)
        self.batch_size = 32
        self.gamma = 0.9

        self.episode_rewards = []
        self.episode_lengths = []
        self.avg_losses = []
        self.epsilons = []

    def train(self):
        """
        Implement your training algorithm here.
        """
        for episode in range(self.no_of_episodes):
            count = 0
            self.stop = False # reset the stop flag
            state, info = self.env.reset() # reset the environment, output is a tuple
            episode_loss = 0
            episode_reward = 0
            num_updates = 0
            while (not self.stop) and count < self.max_steps_per_episode:
                if self.steps % 10 == 0:
                    self.agent.copy() # copy the weights
                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay
                state = self.one_hot_encoder(state) # one hot encode the state
                random_number = torch.rand(1)
                if random_number > self.epsilon:
                    action = self.agent.forward(state) # forward pass
                    action = torch.argmax(action).item() # Use item() to extract the value from the tensor, otherwise it will be a tensor
                else:
                    action = self.env.action_space.sample() # random action
                # print(action)
                next_state, reward, terminated, truncated, info = self.env.step(action) # step the environment
                episode_reward += reward  # Track total reward
                if terminated or truncated:
                    self.replay_buffer.append((state, action, reward, -1)) # append the replay buffer
                else:
                    self.replay_buffer.append((state, action, reward, next_state)) # append the replay buffer
                if len(self.replay_buffer) != self.replay_buffer.maxlen:
                    if terminated or truncated:
                        td_target = reward
                    else:
                        td_target = reward + self.gamma * torch.max(self.agent.forward(self.one_hot_encoder(next_state), td_target=True)) # forward pass
                    q_value = self.agent.forward(state) # forward pass
                    target_q_values = q_value.clone().detach() # detach the q_value
                    target_q_values[action] = td_target # update the q_value
                    # print("action: ", action, "reward: ", reward)
                    # print("td_target: ", td_target, "q_value: ", q_value)
                    loss = F.mse_loss(q_value[action], target_q_values[action]) # calculate the loss
                    self.total_loss += loss
                else:
                    minibatch = random.sample(self.replay_buffer, self.batch_size) # sample the replay buffer
                    for state, action, reward, next_state in minibatch:
                        if next_state == -1:
                            td_target = reward
                        else:
                            td_target = reward + self.gamma * torch.max(self.agent.forward(self.one_hot_encoder(next_state), td_target=True)) # forward pass
                        q_value = self.agent.forward(state) # forward pass
                        target_q_values = q_value.clone().detach() # detach the q_value
                        target_q_values[action] = td_target # update the q_value
                        loss = F.mse_loss(q_value[action], target_q_values[action]) # calculate the loss
                        self.total_loss += loss
                    self.total_loss /= self.batch_size # average the loss

                self.agent.optimizer.zero_grad() # zero the gradient
                self.total_loss.backward() # backward pass
                self.agent.optimizer.step() # update the weights

                episode_loss += self.total_loss.item()  # Track loss
                num_updates += 1

                if terminated or truncated:
                    self.stop = True
                state = next_state
                count += 1
                self.steps += 1
                self.total_loss = 0
            # print(type(state))
            # print(info)
            # Print episode summary
            avg_loss = episode_loss / num_updates if num_updates > 0 else 0
            print(f"Episode {episode}: Average Loss = {avg_loss:.4f}, Total Reward = {episode_reward}")

            # Store metrics for plotting
            self.episode_rewards.append(episode_reward)
            self.avg_losses.append(avg_loss)
            self.episode_lengths.append(count)
            self.epsilons.append(self.epsilon)

    def one_hot_encoder(self, state):
        """
        return a one hot vector of the state
        """
        one_hot_vector = torch.zeros(self.env.observation_space.n)
        one_hot_vector[state] = 1
        return one_hot_vector

    def save_metrics(self, episode):
        """Save performance metrics and generate plots"""
        # Create directory if it doesn't exist
        save_dir = "d:\\Self Learning\\Codes\\RL\\Taxi_Project_RL\\plots"
        os.makedirs(save_dir, exist_ok=True)
        
        plt.style.use('seaborn')  # Add a style for better visualization
        plt.figure(figsize=(15, 10))
        
        # Episode Rewards with moving average
        plt.subplot(2, 2, 1)
        plt.plot(self.episode_rewards, alpha=0.6, label='Raw')
        window_size = 10
        moving_avg = [np.mean(self.episode_rewards[max(0, i-window_size):i+1]) 
                     for i in range(len(self.episode_rewards))]
        plt.plot(moving_avg, 'r', label='Moving Average')
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.legend()
        
        plt.subplot(2, 2, 2)
        plt.plot(self.episode_lengths, color='green')
        plt.title('Episode Lengths')
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        plt.grid(True)
        
        plt.subplot(2, 2, 3)
        plt.plot(self.avg_losses, color='orange')
        plt.title('Average Loss per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.yscale('log')  # Log scale for better loss visualization
        plt.grid(True)
        
        plt.subplot(2, 2, 4)
        plt.plot(self.epsilons, color='purple')
        plt.title('Exploration Rate (Epsilon)')
        plt.xlabel('Episode')
        plt.ylabel('Epsilon')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}\\training_metrics_ep{episode}.png", 
                    dpi=300,
                    bbox_inches='tight')
        plt.close()
        
        print(f"Metrics saved at episode {episode}")

    def save_model(self, episode):
        """Save the trained model"""
        save_dir = "d:\\Self Learning\\Codes\\RL\\Taxi_Project_RL\\models"
        os.makedirs(save_dir, exist_ok=True)
        model_state = {
            'weights': self.agent.weights,
            'biases': self.agent.biases,
            'episode': episode,
        }
        torch.save(model_state, f"{save_dir}\\model_ep{episode}.pth")
        print(f"Model saved at episode {episode}")



env = Env('Taxi-v3')
agent = Agent([env.observation_space.n, 64, 64, env.action_space.n])
train = Train(env, agent)
train.train()
train.save_metrics(train.no_of_episodes)
train.save_model(train.no_of_episodes)