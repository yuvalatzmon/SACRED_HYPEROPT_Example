import math

import hyperopt
from hyperopt import fmin, hp
from hyperopt.mongoexp import MongoTrials

from sacred_wrapper import hyperopt_objective

import warnings
# see https://stackoverflow.com/a/40846742
warnings.filterwarnings("ignore", message="numpy.dtype size changed")

if __name__ == '__main__':
    # algo = 'rand'
    algo = 'grid'
    name = 'mnist_keras'
    version = 110
    trials = MongoTrials('mongo://localhost:27017/hyperopt/jobs',
                         exp_key='%s_v%d' % (name, version))

    if algo == 'rand':
        # Define the search space
        space = {
            'lr': hp.qloguniform('lr', math.log(0.00001), math.log(0.1), 0.00001),
            'fc_dim': hp.qloguniform('fc_dim', math.log(32), math.log(256), 8),
            'dropout_rate': hp.quniform('dropout_rate', 0.1, 1, 0.05),
        }

        print('Pending on workers to connect ..')
        argmin = fmin(fn=hyperopt_objective,
                    space=space,
                    algo=hyperopt.rand.suggest,
                    max_evals=12,
                    trials=trials,
                    verbose=1)
        best_acc = 1-trials.best_trial['result']['loss']

        print('best val acc=', best_acc, 'params:', argmin)

    elif algo == 'grid':
        from hyperopt_grid import hyperopt_grid, to_exp_space
        # Grid search


        # Define the search spac. Ranges are defined in log10 space
        log_ranges = dict(lr=(-5, -1, 0.5), # 1e-5, 3e-5, 1e-4, 3e-4, ...
                          fc_dim=(1.5, 3, 0.25), # 30, 60, 100, 200, 300, 600, 1000
                          dropout_rate=(-1, -0.1, 0.1)) # .1, .2, .3, .4, .5, .6, .8,
        # A finer search space
        log_ranges = dict(lr=(-3.5, -3.5, 0.5), # 1e-5, 3e-5, 1e-4, 3e-4, ...
                          fc_dim=(2., 4, 0.25), # 30, 60, 100, 200, 300, 600, 1000
                          dropout_rate=(-0.4, -0.1, 0.1)) # .4, .5, .6, .8,


        grid = hyperopt_grid(log_ranges)
        print('Pending on workers to connect ..')
        argmin = fmin(fn=hyperopt_objective,
                     space=grid.space,
                    algo=grid.suggest,
                    # max_evals=grid.num_combinations,
                    max_evals=10,
                    trials=trials,
                    verbose=1)
        best_acc = 1-trials.best_trial['result']['loss']

        print('best val acc = ', best_acc, '\nparams = ', to_exp_space(argmin),
              '\nlog10(params) = ', argmin)

