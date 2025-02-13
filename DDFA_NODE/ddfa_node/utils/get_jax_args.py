import argparse
def get_args():
    parser = argparse.ArgumentParser(description='Train Neural ODE with configurable parameters')
    parser.add_argument('--timesteps_per_trial', type=int, default=200,
                      help='Number of timesteps per trial')
    parser.add_argument('--width_size', type=int, default=128,
                      help='Width size of the network')
    parser.add_argument('--hidden_size', type=int, default=256,
                      help='Hidden size of the network')
    parser.add_argument('--depth', type=int, default=3,
                      help='Depth of the network')
    parser.add_argument('--batch_size', type=int, default=2**10,
                      help='Batch size for training')
    parser.add_argument('--seed', type=int, default=6970,
                      help='Random seed')
    parser.add_argument('--lr', type=float, default=1e-3,
                      help='Learning rate')
    parser.add_argument('--steps', type=int, default=20000,
                      help='Number of training steps')
    parser.add_argument('--length', type=float, default=1.0,
                      help='Length parameter')
    parser.add_argument('--skip', type=int, default=10,
                      help='Skip parameter')
    parser.add_argument('--seeding', type=float, default=1,
                      help='Seeding parameter')
    parser.add_argument('--k', type=int, default=1,
                      help='k parameter')
    parser.add_argument('--augment_dims', type=int, default=0,
                      help='Number of augmented dimensions')
    parser.add_argument('--use_recurrence', action='store_true',
                      help='Use recurrence in the model')
    parser.add_argument('--use_linear', action='store_true', default=True,
                      help='Use linear layer in the model')
    parser.add_argument('--only_linear', action='store_true',
                      help='Use only linear layer')
    parser.add_argument('--optim_type', type=str, default='adabelief',
                      help='Optimizer type')
    parser.add_argument('--lmbda', type=float, default=0.0001,
                      help='Lambda parameter')
    parser.add_argument("--GPU", type=int, default=0, help="GPU to use")
    parser.add_argument("--save_config", action='store_true', help="Save the configuration to a file")
    return parser.parse_args()
