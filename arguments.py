import argparse

def get_args():
    parser = argparse.ArgumentParser(description='RL')
    
    # environment
    parser.add_argument('--env-name', default='FetchReach-v1', help='environment to train on (default: simple_spread)')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--relative-goal', action='store_true')

    # training 
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num-processes', type=int, default=2)
    parser.add_argument('--n-epochs', type=int, default=50)
    parser.add_argument('--n-cycles', type=int, default=10)
    parser.add_argument('--n-batches', type=int, default=40)
    parser.add_argument('--actor-lr', default=1e-3, type=float)
    parser.add_argument('--critic-lr', default=1e-3, type=float)

    # logging
    parser.add_argument('--logid', default=None, type=int, help='unique id for each run (default: date_time)')
    
    # Miscellaneous
    parser.add_argument('--test', action='store_true')
    
    args = parser.parse_args()
            
    return args
