from model import Actor, Critic
from replay_buffer import ReplayBuffer
from her import HER_sampler
from normalizer import Normalizer
import utils
import torch
import numpy as np
import ipdb


class Policy:
    def __init__(self, params):
        self.relative_goal = params['relative_goal']
        env = params['cached_env']
        self.T = env._max_episode_steps
        
        self.max_u = 1.0 
        self.clip_obs = 200.0
        self.gamma = 1 - 1.0/self.T
        self.tau = 0.05
        self.max_grad_norm = 0.5
        self.clip_pos_returns = True
        self.clip_return = self.T
        self.train_batch_size = 256
        # TODO: learning rate scheduler?? 
        
        self.device = params['device']
        buffer_shapes = utils.get_buffer_shapes(env, self.T)
        self.obs_dim = buffer_shapes['o'][-1]
        self.goal_dim = buffer_shapes['g'][-1]
        input_size = self.obs_dim + self.goal_dim
        self.action_dim = env.action_space.shape[0]

        self.main_actor = Actor(input_size, self.action_dim, self.max_u).to(self.device)
        self.target_actor = Actor(input_size, self.action_dim, self.max_u).to(self.device)
        self.main_critic = Critic(input_size+self.action_dim).to(self.device)
        self.target_critic = Critic(input_size+self.action_dim).to(self.device)
        utils.hard_update(self.target_actor, self.main_actor)
        utils.hard_update(self.target_critic, self.main_critic)

        self.norm_clip = 5.0
        self.o_stats = Normalizer(self.obs_dim, clipob=self.norm_clip)
        self.g_stats = Normalizer(self.goal_dim, clipob=self.norm_clip)

        self.optim_actor = torch.optim.Adam(self.main_actor.parameters(), params['actor_lr'])
        self.optim_critic = torch.optim.Adam(self.main_critic.parameters(), params['critic_lr'])
        self.critic_loss = torch.nn.SmoothL1Loss()
        
        def reward_fun(ag_2, g, info):
            return env.compute_reward(achieved_goal=ag_2, desired_goal=g, info=info)
        
        self.her_sampler = HER_sampler(replay_k=4, reward_fun=reward_fun)
        buffer_size = int(1e6)
        self.buffer = ReplayBuffer(buffer_shapes, buffer_size, self.T, self.her_sampler)
        
    def get_actions(self, obs, ag, goal, noise_eps=0, random_eps=0):
        o, g = self.preprocess_og(obs, ag, goal)
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
        u += np.random.binomial(1, random_eps, u.shape[0]).reshape(-1, 1) * (self._random_action(u.shape[0]) - u)
        # TODO: where is frame skip ?
        return u

    def _random_action(self, n):
        return np.random.uniform(low=-self.max_u, high=self.max_u, size=(n, self.action_dim))

    def append_goal(self, obs, goal):
        return np.concatenate([obs, goal], -1)

    def split_obs(self, obs):
        return np.split(obs, [self.obs_dim, self.obs_dim+self.goal_dim], -1)

    def preprocess_og(self, obs, ag, goal):
        # subtract achieved goal from desired goal
        if self.relative_goal:
            goal = goal - ag
        obs = np.clip(obs, -self.clip_obs, self.clip_obs)
        goal = np.clip(goal, -self.clip_obs, self.clip_obs)
        return obs, goal

    def update_target_net(self):
        # soft update or hard update
        utils.soft_update(self.target_critic, self.main_critic, self.tau)
        utils.soft_update(self.target_actor, self.main_actor, self.tau)

    def store_episode(self, episode_batch, update_stats=True):
        self.buffer.store_episode(episode_batch)
        if update_stats:
            # add transitions to normalizer
            episode_batch['o_2'] = episode_batch['o'][:, 1:, :]
            episode_batch['ag_2'] = episode_batch['ag'][:, 1:, :]
            num_normalizing_transitions = episode_batch['u'].shape[0]*episode_batch['u'].shape[1] 
            transitions = self.her_sampler.sample(episode_batch, num_normalizing_transitions)

            o, g = self.preprocess_og(transitions['o'], transitions['ag'], transitions['g'])
    
            # No need to preprocess the o_2 and g_2 since this is only used for stats
            self.o_stats.update(o)
            self.g_stats.update(g)

            self.o_stats.recompute_stats()
            self.g_stats.recompute_stats()

    def train(self):
        batch = self.buffer.sample(self.train_batch_size)
        
        o, g = self.preprocess_og(batch['o'], batch['ag'], batch['g'])
        o = self.o_stats.normalize(o)
        g = self.g_stats.normalize(g)
        obs = self.append_goal(o, g)
        obs = torch.from_numpy(obs).float().to(self.device)

        o_2, g_2 = self.preprocess_og(batch['o_2'], batch['ag_2'], batch['g']) 
        o_2 = self.o_stats.normalize(o_2)
        g_2 = self.g_stats.normalize(g_2)
        obs_2 = self.append_goal(o_2, g_2)
        obs_2 = torch.from_numpy(obs_2).float().to(self.device)

        rew = torch.from_numpy(batch['r']).float().to(self.device)
        act = torch.from_numpy(batch['u']).float().to(self.device)

        with torch.no_grad():
            act_2 = self.target_actor(obs_2)
            next_qsa = self.target_critic(obs_2, act_2).squeeze()
            target_q = rew + self.gamma * next_qsa
            clip_range = (-self.clip_return, 0. if self.clip_pos_returns else np.inf)
            target_q = np.clip(target_q, *clip_range)

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

        # self.update_target_net()
        return cr_loss.item(), policy_loss.item()

    def logs(self):
        logs = []
        logs += [('stats_o/mean', np.mean(self.o_stats.mean))]
        logs += [('stats_o/std', np.mean(self.o_stats.std))]
        logs += [('stats_g/mean', np.mean(self.g_stats.mean))]
        logs += [('stats_g/std', np.mean(self.g_stats.std))]
        return logs

    def set_train_mode(self):
        self.main_actor.train()
        self.main_critic.train()
        self.target_actor.train()
        self.target_critic.train()

    def set_eval_mode(self):
        self.main_actor.eval()
        self.main_critic.eval()
        self.target_actor.eval()
        self.target_critic.eval()
        