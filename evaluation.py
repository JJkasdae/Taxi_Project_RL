import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
from agent import Agent
from env import Env

class Evaluation:
    def __init__(self, env, agent, no_of_episodes):
        self.env = env
        self.agent = agent
        self.no_of_episodes = no_of_episodes
        self.stop = False
        self.max_steps_per_episode = 50
        self.path = []  # Store agent's path
        self.rewards = []  # Store rewards
        
    def evaluate(self):
        state, info = self.env.reset()
        count = 0
        total_reward = 0
        self.path = [(state, None, 0)]  # Initial state
        self.load_model(self.no_of_episodes)
        
        while (not self.stop) and count < self.max_steps_per_episode:
            self.env.render()  # Visualize the environment
            action = self.agent.forward(self.env.one_hot_encoder(state))
            action = torch.argmax(action).item()
            next_state, reward, terminated, truncated, info = self.env.step(action)
            
            # Store state, action, and reward
            self.path.append((next_state, action, reward))
            total_reward += reward
            
            if terminated or truncated:
                self.stop = True
                print(f"\nEpisode finished after {count} steps")
                print(f"Total reward: {total_reward}")
                self.print_path()
            
            state = next_state
            count += 1
            
            # Add a small delay to make the visualization more visible
            time.sleep(0.5)  # 0.5 second delay between steps

    def print_path(self):
        """Print the agent's path and actions"""
        actions = ['South', 'North', 'East', 'West', 'Pickup', 'Dropoff']
        print("\nAgent's path:")
        for i, (state, action, reward) in enumerate(self.path):
            if action is not None:
                print(f"Step {i}: State {state} -> Action: {actions[action]} (Reward: {reward})")
            else:
                print(f"Step {i}: Initial State {state}")

    def load_model(self, no_episodes):
        """
        Load the model from the model_name
        """
        model_path = f"d:\\Self Learning\\Codes\\RL\\Taxi_Project_RL\\models\\model_ep{no_episodes}.pth"
        model_state = torch.load(model_path)

        # Load the parameters into the agent
        self.agent.weights = model_state["weights"]
        self.agent.biases = model_state["biases"]
        print(f"Model loaded from episode {no_episodes}")

env = Env('Taxi-v3', render_mode='human')
agent = Agent([env.observation_space.n, 64, 64, env.action_space.n])
evaluation = Evaluation(env, agent, 500)
evaluation.evaluate()