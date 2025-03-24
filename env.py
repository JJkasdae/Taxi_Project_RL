import gymnasium as gym
import torch

class Env:
    def __init__(self, env_name, render_mode):
        self.env = gym.make(env_name, render_mode=render_mode)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def render(self):
        self.env.render()
    
    def one_hot_encoder(self, state):
        """
        return a one hot vector of the state
        """
        one_hot_vector = torch.zeros(self.env.observation_space.n)
        one_hot_vector[state] = 1
        return one_hot_vector
# env = Env('Taxi-v3')
# env.reset()
# print(env.step(0))
# print(env.action_space)
