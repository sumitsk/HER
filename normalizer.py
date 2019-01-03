import gym_vecenv
import numpy as np


class RMSNormalizer:
    def __init__(self, size, min_std=1e-2, clipob=10.0):
        self.rms = gym_vecenv.RunningMeanStd(shape=size)
        self.clipob = clipob
        self.min_std = min_std 

    @property
    def mean(self):
        return self.rms.mean
    
    @property
    def std(self):
        var = np.maximum(self.min_std**2, self.rms.var)
        return np.maximum(self.min_std, var**.5)
    
    def update(self, obs):
        self.rms.update(obs)

    def recompute_stats(self):
        pass

    def normalize(self, obs):
        obs = np.clip((obs - self.mean) / self.std, -self.clipob, self.clipob)
        return obs


class Normalizer:
    def __init__(self, size, min_std=1e-2, clipob=10.0):
        self.clipob = clipob
        self.min_std = min_std
        self.size = size

        self.obs_sum = np.zeros(self.size, np.float32)
        self.obs_sumsq = np.zeros(self.size, np.float32)
        self.count = 0
    
        self.mean = np.zeros(self.size, np.float32)
        self.std = np.ones(self.size, np.float32)

    def update(self, v):
        v = v.reshape(-1, self.size)
        self.obs_sum += v.sum(axis=0)
        self.obs_sumsq += (np.square(v)).sum(axis=0)
        self.count += v.shape[0]

    def reset(self):
        self.count = 0
        self.obs_sum[...] = 0
        self.obs_sumsq[...] = 0

    def recompute_stats(self):
        self.mean = self.obs_sum / self.count
        var = self.obs_sumsq / self.count - (self.obs_sum / self.count)**2
        var = np.maximum(self.min_std**2, var)
        self.std = np.maximum(var**.5, self.min_std)
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

        