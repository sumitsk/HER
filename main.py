import torch
import os
import numpy as np
import datetime
from copy import deepcopy

import utils
from arguments import get_args
from learner import Learner
from policy import Policy
from baselines import logger

import ipdb


if __name__ == '__main__':
    args = get_args()
    logid = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f") if args.logid is None else str(args.logid)
    logdir = os.path.join('save', logid)
    logger.configure(logdir)
    
    params = vars(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    params['device'] = device
    params['cached_env'] = utils.get_cached_env(params['env_name'])

    policy = Policy(params)

    train_envs = utils.make_parallel_envs(params['env_name'], params['seed'], params['num_processes'])
    trainer_params = deepcopy(params)
    trainer_params['envs'] = train_envs
    trainer_params['exploit'] = False
    trainer = Learner(policy, trainer_params)

    eval_seed = np.random.randint(0, 100)
    eval_num_processes = params['num_processes']
    eval_envs = utils.make_parallel_envs(params['env_name'], eval_seed, eval_num_processes)
    evaluator_params = deepcopy(params)
    evaluator_params['envs'] = eval_envs
    evaluator_params['exploit'] = True
    evaluator = Learner(policy, evaluator_params)

    n_test_rollouts = 10
    for epoch in range(args.n_epochs):
        trainer.clear_history()
        for _ in range(args.n_cycles):
            episode = trainer.generate_rollouts()
            policy.store_episode(episode)

            for _ in range(args.n_batches):
                critic_loss, policy_loss = policy.train()
            print(critic_loss, policy_loss)

            # policy.update_target_net()

        evaluator.clear_history()
        for _ in range(n_test_rollouts):
            evaluator.generate_rollouts()

        # log stuffs
        ipdb.set_trace()
        logger.record_tabular('epoch', epoch)
        for key, val in evaluator.logs('test'):
            logger.record_tabular(key, val)
        for key, val in trainer.logs('train'):
            logger.record_tabular(key, val)
        for key, val in policy.logs():
            logger.record_tabular(key, val)

        logger.dump_tabular()


