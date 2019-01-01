import torch
import os
import numpy as np
import datetime
from copy import deepcopy
import utils
from arguments import get_args
from learner import Learner
from policy import Policy
import logger
from tensorboard_logger import configure, log_value
import ipdb


if __name__ == '__main__':
    args = get_args()
    logid = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f") if args.logid is None else str(args.logid)
    logdir = os.path.join('save', logid)
    utils.check_logdir(logdir)
    logger.configure(logdir)
    configure(logdir)

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
    best_success_rate = -1
    latest_policy_path = os.path.join(logger.get_dir(), 'policy_latest.pt')
    best_policy_path = os.path.join(logger.get_dir(), 'policy_best.pt')
    periodic_policy_path = os.path.join(logger.get_dir(), 'policy_{}.pt')

    for epoch in range(params['n_epochs']):
        trainer.clear_history()
        policy.set_train_mode()
        for i in range(params['n_cycles']):
            episode = trainer.generate_rollouts()
            policy.store_episode(episode)

            for _ in range(params['n_batches']):
                critic_loss, policy_loss = policy.train()
            step = epoch+i*params['n_cycles']
            log_value('critic_loss', critic_loss, step)
            log_value('policy_loss', policy_loss, step)

            policy.update_target_net()

        evaluator.clear_history()
        policy.set_eval_mode()
        for _ in range(n_test_rollouts):
            evaluator.generate_rollouts()

        # log statistics
        logger.record_tabular('epoch', epoch)
        test_stats = evaluator.logs()
        for key, val in test_stats.items():
            logger.record_tabular('test/'+key, val)
        train_stats = trainer.logs()
        for key, val in train_stats.items():
            logger.record_tabular('train/'+key, val)
        for key, val in policy.logs():
            logger.record_tabular(key, val)

        logger.dump_tabular()
        log_value('train_success_rate', train_stats['success_rate'], epoch)
        log_value('test_success_rate', test_stats['success_rate'], epoch)

        success_rate = test_stats['success_rate']
        if success_rate >= best_success_rate:
            best_success_rate = success_rate
            logger.info('New best success rate: {}. Saving policy to {} ...'.format(best_success_rate, best_policy_path))
            policy.save(epoch, best_policy_path)
            policy.save(epoch, latest_policy_path)
        if epoch % params['save_every'] == 0:
            policy_path = periodic_policy_path.format(epoch)
            logger.info('Saving periodic policy to {} ...'.format(policy_path))
            policy.save(epoch, policy_path)

