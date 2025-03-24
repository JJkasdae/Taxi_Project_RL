import gymnasium as gym

class Env:
    def __init__(self, env_name):
        self.env = gym.make(env_name)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)
# env = Env('Taxi-v3')
# env.reset()
# print(env.step(0))
# print(env.action_space)
