import torch
import numpy as np
from collections import deque

from model import Actor, Critic
from replay_buffer import ReplayBuffer
from her import HER_sampler
from normalizer import Normalizer
import utils
import gym
import ipdb


class Learner(object):
    def __init__(self, params):
        self.envs = params['envs']
        self.device = params['device']
        self.num_processes = params['num_processes']
        self.relative_goal = params['relative_goal']
        env = gym.make(params['env_name'])
        self.T = env._max_episode_steps
        
        self.max_u = 1.0 
        self.clip_obs = 200.0
        self.gamma = 1 - 1.0/self.T
        self.tau = 0.001
        history_len = 100
        self.success_history = deque(maxlen=history_len)
        self.n_episodes = 0
        self.exploit = params['exploit']
        self.noise_eps = 0.2 
        self.random_eps = 0.3 
        self.norm_clip = 5.0

        buffer_shapes = utils.get_buffer_shapes(env, self.T)
        self.obs_dim = buffer_shapes['o'][-1]
        self.goal_dim = buffer_shapes['g'][-1]
        input_size = self.obs_dim + self.goal_dim
        self.action_dim = self.envs.action_space.shape[0]
        
        self.main_actor = Actor(input_size, self.action_dim, self.max_u).to(self.device)
        self.target_actor = Actor(input_size, self.action_dim, self.max_u).to(self.device)
        self.main_critic = Critic(input_size+self.action_dim).to(self.device)
        self.target_critic = Critic(input_size+self.action_dim).to(self.device)
        utils.hard_update(self.target_actor, self.main_actor)
        utils.hard_update(self.target_critic, self.main_critic)

        self.max_grad_norm = 0.5
        self.optim_actor = torch.optim.Adam(self.main_actor.parameters(), 1e-3)
        self.optim_critic = torch.optim.Adam(self.main_critic.parameters(), 1e-4)
        self.critic_loss = torch.nn.SmoothL1Loss()
        # TODO: learning rate scheduler?? 
        
        def reward_fun(ag_2, g, info):
            return env.compute_reward(achieved_goal=ag_2, desired_goal=g, info=info)
        
        self.her_sampler = HER_sampler(replay_k=4, reward_fun=reward_fun)
        buffer_size = int(1e6)
        self.buffer = ReplayBuffer(buffer_shapes, buffer_size, self.T, self.her_sampler)
        self.o_stats = Normalizer(self.obs_dim, clipob=self.norm_clip)
        self.g_stats = Normalizer(self.goal_dim, clipob=self.norm_clip)

    def generate_rollouts(self):
        o = self.envs.reset()
        o, g = self.split_obs(o)
        obs, goals, actions, successes = [], [], [], []

        for t in range(self.T):
            with torch.no_grad():
                act = self.get_actions(obs=o,
                                       goal=g,
                                       noise_eps=self.noise_eps if not self.exploit else 0,
                                       random_eps=self.random_eps if not self.exploit else 0)
            # reward will be recomputed while HER sampling
            next_o, _, _, info = self.envs.step(act)
            next_o, _ = self.split_obs(next_o)
            obs.append(o.copy())
            goals.append(g.copy())
            actions.append(act.copy())
            succ = np.reshape([i['is_success'] for i in info], (-1,1))
            successes.append(succ)
            o = next_o.copy()
        obs.append(o.copy())
        
        # num_processes x (T+1) x dim
        obs = np.stack(obs, 1)
        achieved_goals = obs[:, :, -self.goal_dim:]

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

    def append_goal(self, obs, goal):
        return np.concatenate([obs, goal], -1)

    def split_obs(self, obs):
        return np.split(obs, [obs.shape[-1]-self.goal_dim], -1)

    def get_achieved_goal(self, obs):
        return self.split_obs(obs)[-1]

    def preprocess_og(self, obs, goal):
        # subtract achieved goal from desired goal
        if self.relative_goal:
            goal -= self.get_achieved_goal(obs)
        obs = np.clip(obs, -self.clip_obs, self.clip_obs)
        goal = np.clip(goal, -self.clip_obs, self.clip_obs)
        return obs, goal

    def get_actions(self, obs, goal, noise_eps=0, random_eps=0):
        o, g = self.preprocess_og(obs, goal)
        o = self.o_stats.normalize(o)
        g = self.g_stats.normalize(g)
        inp = self.append_goal(o, g)
        inp = torch.from_numpy(inp).float().to(self.device)
        u = self.main_actor(inp)
        u = u.cpu().numpy()

        # Gaussian noise
        noise = noise_eps * self.max_u * np.random.randn(*u.shape)  
        u += noise
        u = np.clip(u, -self.max_u, self.max_u)
        
        # eps-greedy
        # TODO: if using this, need to reduce eps somewhere ? 
        u += np.random.binomial(1, random_eps, u.shape[0]).reshape(-1, 1) * (self._random_action(u.shape[0]) - u)
        # TODO: where is frame skip ?
        return u

    def _random_action(self, n):
        return np.random.uniform(low=-self.max_u, high=self.max_u, size=(n, self.action_dim))

    def store_episode(self, episode_batch, update_stats=True):
        self.buffer.store_episode(episode_batch)
        if update_stats:
            # add transitions to normalizer
            episode_batch['o_2'] = episode_batch['o'][:, 1:, :]
            episode_batch['ag_2'] = episode_batch['ag'][:, 1:, :]
            num_normalizing_transitions = episode_batch['u'].shape[0]*episode_batch['u'].shape[1] 
            transitions = self.her_sampler.sample(episode_batch, num_normalizing_transitions)

            o, g = self.preprocess_og(transitions['o'], transitions['g'])
            # No need to preprocess the o_2 and g_2 since this is only used for stats

            self.o_stats.update(o)
            self.g_stats.update(g)

    def train(self):
        batch_size = 256
        batch = self.buffer.sample(batch_size)
        
        # preprocessing of obs and goal
        o, g = self.preprocess_og(batch['o'], batch['g'])
        o = self.o_stats.normalize(o)
        g = self.g_stats.normalize(g)
        obs = self.append_goal(o, g)
        obs = torch.from_numpy(obs).float().to(self.device)

        o_2, g_2 = self.preprocess_og(batch['o_2'], batch['g']) 
        o_2 = self.o_stats.normalize(o_2)
        g_2 = self.g_stats.normalize(g_2)
        obs_2 = self.append_goal(o_2, g_2)
        obs_2 = torch.from_numpy(obs_2).float().to(self.device)

        rew = torch.from_numpy(batch['r']).float().to(self.device)
        act = torch.from_numpy(batch['u']).float().to(self.device)

        with torch.no_grad():
            act_2 = self.target_actor(obs_2)
            # action is saturating to -1 or 1
            next_qsa = self.target_critic(obs_2, act_2).squeeze()
            target_q = rew + self.gamma * next_qsa

        self.optim_critic.zero_grad()
        main_q = self.main_critic(obs, act).squeeze()
        # with torch.no_grad():
        #     td_error = target_q - q
        cr_loss = self.critic_loss(main_q, target_q)
        # TODO: gradient clipping
        cr_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.main_critic.parameters(), self.max_grad_norm)
        self.optim_critic.step()

        self.optim_actor.zero_grad()
        pred_act = self.main_actor(obs)
        policy_loss = -self.main_critic(obs, pred_act)
        policy_loss = policy_loss.mean()

        # torch.nn.utils.clip_grad_norm_(self.main_actor.parameters(), self.max_grad_norm)
        policy_loss.backward()
        self.optim_actor.step()

        self.update_target_net()
        return cr_loss.item(), policy_loss.item()

    def update_target_net(self):
        # soft update or hard update
        utils.soft_update(self.target_critic, self.main_critic, self.tau)
        utils.soft_update(self.target_actor, self.main_actor, self.tau)

    def clear_history(self):
        self.success_history.clear()

    def logs(self, prefix=''):
        logs = []
        if prefix == 'stats':
            logs += [('stats_o/mean', np.mean(self.o_stats.rms.mean))]
            logs += [('stats_o/std', np.mean(self.o_stats.rms.var**0.5))]
            logs += [('stats_g/mean', np.mean(self.g_stats.rms.mean))]
            logs += [('stats_g/std', np.mean(self.g_stats.rms.var**0.5))]
        else:
            logs += [('success_rate', np.mean(self.success_history))]
            logs += [('episode', self.n_episodes)]

        if prefix in ['train', 'test']:
            return [(prefix + '/' + key, val) for key, val in logs]
        else:
            return logs

            