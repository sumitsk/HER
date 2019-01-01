import torch
import numpy as np
from collections import deque


class Learner:
    def __init__(self, policy, params):
        self.policy = policy
        self.envs = params['envs']
        self.num_processes = params['num_processes']
        
        history_len = 100
        self.success_history = deque(maxlen=history_len)
        self.exploit = params['exploit']
        self.noise_eps = 0.2 if not self.exploit else 0
        self.random_eps = 0.3 if not self.exploit else 0
        self.n_episodes = 0
        env = params['cached_env']
        self.T = env._max_episode_steps
        
    def generate_rollouts(self):
        obs = self.envs.reset()
        o, ag, g = self.policy.split_obs(obs)
        obs, goals, achieved_goals, actions, successes = [], [], [], [], []

        for t in range(self.T):
            with torch.no_grad():
                act = self.policy.get_actions(o, ag, g, noise_eps=self.noise_eps, random_eps=self.random_eps)
            # reward will be recomputed while HER sampling
            next_o, _, _, info = self.envs.step(act)
            next_o, next_ag, _ = self.policy.split_obs(next_o)
            obs.append(o.copy())
            achieved_goals.append(ag.copy())
            goals.append(g.copy())
            actions.append(act.copy())
            succ = np.reshape([i['is_success'] for i in info], (-1,1))
            successes.append(succ)
            o = next_o.copy()
            ag = next_ag.copy()
        obs.append(o.copy())
        achieved_goals.append(ag.copy())
        
        # num_processes x (T+1) x dim
        obs = np.stack(obs, 1)
        achieved_goals = np.stack(achieved_goals, 1)

        # num_processes x T x dim
        goals = np.stack(goals, 1)
        actions = np.stack(actions, 1)
        successes = np.stack(successes, 1)
        
        # HER replay buffer expects obs to be goal removed
        episode = dict(o=obs, u=actions, g=goals, ag=achieved_goals, info_is_success=successes)

        self.n_episodes += self.num_processes
        success_rate = successes[:, -1].squeeze().mean()
        self.success_history.append(success_rate)
        return episode

    def clear_history(self):
        self.success_history.clear()

    def logs(self, prefix='train'):
        assert prefix in ['train', 'test']
        logs = []
        logs += [('success_rate', np.mean(self.success_history))]
        logs += [('episode', self.n_episodes)]
        return [(prefix + '/' + key, val) for key, val in logs]

            