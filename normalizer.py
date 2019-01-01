import gym_vecenv
import numpy as np


class Normalizer:
    def __init__(self, dim, epsilon=1e-8, clipob=10.0):
        self.rms = gym_vecenv.RunningMeanStd(shape=dim)
        self.clipob = clipob
        self.epsilon = epsilon 

    def update(self, obs):
        self.rms.update(obs)

    def normalize(self, obs):
        obs = np.clip((obs - self.rms.mean) / np.sqrt(self.rms.var + self.epsilon), -self.clipob, self.clipob)
        return obs