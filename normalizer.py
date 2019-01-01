import gym_vecenv
import numpy as np


class Normalizer:
    def __init__(self, dim, min_std=1e-2, clipob=10.0):
        self.rms = gym_vecenv.RunningMeanStd(shape=dim)
        self.clipob = clipob
        self.min_std = min_std 

    def update(self, obs):
        self.rms.update(obs)

    def normalize(self, obs):
        std = np.maximum(np.sqrt(self.rms.var), self.min_std) 
        obs = np.clip((obs - self.rms.mean) / std, -self.clipob, self.clipob)
        return obs