import gym_vecenv
import numpy as np
import ipdb


class GymNormalizer:
    def __init__(self, size, min_std=1e-2, clipob=10.0):
        self.rms = gym_vecenv.RunningMeanStd(shape=size)
        self.clipob = clipob
        self.min_std = min_std 

    @property
    def mean(self):
        return self.rms.mean
    
    @property
    def std(self):
        return self.rms.var**.5
    
    def update(self, obs):
        self.rms.update(obs)

    def normalize(self, obs):
        std = np.maximum(self.std, self.min_std) 
        obs = np.clip((obs - self.mean) / std, -self.clipob, self.clipob)
        return obs


class Normalizer:
    def __init__(self, size, min_std=1e-2, clipob=10.0):
        self.clipob = clipob
        self.min_std = min_std
        self.size = size

        self.local_sum = np.zeros(self.size, np.float32)
        self.local_sumsq = np.zeros(self.size, np.float32)
        self.local_count = 0
    
        self.mean = np.zeros(self.size, np.float32)
        self.std = np.ones(self.size, np.float32)

    def update(self, v):
        v = v.reshape(-1, self.size)
        self.local_sum += v.sum(axis=0)
        self.local_sumsq += (np.square(v)).sum(axis=0)
        self.local_count += v.shape[0]

    def reset(self):
        self.local_count = 0
        self.local_sum[...] = 0
        self.local_sumsq[...] = 0

    def recompute_stats(self):
        self.mean = self.local_sum / self.local_count
        std = (self.local_sumsq / self.local_count - (self.local_sum / self.local_count)**2)**.5
        self.std = np.maximum(std, self.min_std)
        self.reset()

    def normalize(self, obs):
        norm_obs = np.clip((obs - self.mean) / self.std, -self.clipob, self.clipob)
        return norm_obs


class IdentityNormalizer:
    def __init__(self, size, std=1.0):
        self.mean = np.zeros(size, np.float32)
        self.std = std*np.ones(size, np.float32)

    def update(self, v):
        pass

    def recompute_stats(self):
        pass

    def normalize(self, x):
        return (x-self.mean)/self.std

        