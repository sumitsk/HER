import os
import shutil
import sys
import gym_vecenv
import gym


def check_logdir(log_dir):
    # raise warning if log directory already exists 
    if os.path.exists(log_dir):
        print('\nLog directory exists already! Enter')
        ch = input('c (rename the existing directory with _old and continue)\ns (stop)!\ndel (delete existing dir): ')
        if ch == 's':
            sys.exit(0)
        elif ch == 'c':
            os.rename(log_dir, log_dir+'_old')
        elif ch == 'del':
            shutil.rmtree(log_dir)
        else:
            raise NotImplementedError('Unknown input')
    os.makedirs(log_dir)


def make_robotics_env(env_name, seed):
    def _thunk():
        env = gym.make(env_name)
        # flatten for subprocvec support
        env = gym.wrappers.FlattenDictWrapper(env, ['observation', 'achieved_goal', 'desired_goal'])
        env.seed(seed)
        return env
    return _thunk


def make_parallel_envs(env_name, seed, num_processes):
    envs = [make_robotics_env(env_name=env_name, seed=seed+i*1000000) for i in range(num_processes)]

    if num_processes > 1:
        envs = gym_vecenv.SubprocVecEnv(envs, no_reset=True)
    else:
        envs = gym_vecenv.DummyVecEnv(envs, no_reset=True)
    
    # do not normalize envs
    return envs


def get_cached_env(env_name):
    return gym.make(env_name)

    
def get_buffer_shapes(env, T):
    env.reset()
    obs, _, _, info = env.step(env.action_space.sample())
    shapes = {
        'o': (T+1, obs['observation'].shape[0]),
        'ag': (T+1, obs['desired_goal'].shape[0]),
        'u': (T, env.action_space.shape[0]),
        'g': (T, obs['desired_goal'].shape[0]),
        'info_is_success': (T, 1)
    }
    return shapes


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0-tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)
