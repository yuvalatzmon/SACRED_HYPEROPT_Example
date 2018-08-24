import math

import hyperopt
from hyperopt import fmin, hp
from hyperopt.mongoexp import MongoTrials
from sacred_wrapper import hyperopt_objective

import warnings
# see https://stackoverflow.com/a/40846742
warnings.filterwarnings("ignore", message="numpy.dtype size changed")

if __name__ == '__main__':
    # algo = hyperopt.tpe.suggest # TPE search
    algo = hyperopt.rand.suggest #random search
    name = 'mnist_keras'
    version = 100

    # Define the search space
    space = {
        'lr': hp.qloguniform('lr', math.log(0.00001), math.log(0.1), 0.00001),
        'fc_dim': hp.qloguniform('fc_dim', math.log(32), math.log(256), 8),
        'dropout_rate': hp.quniform('dropout_rate', 0.1, 1, 0.05),
    }

    # Define a finer search space
    # space = {
    #     'lr': hp.qloguniform('lr', math.log(0.002), math.log(0.002), 0.00001),
    #     'fc_dim': hp.qloguniform('fc_dim', math.log(32), math.log(256), 8),
    #     'dropout_rate': hp.quniform('dropout_rate', 0.3, 1, 0.05),
    # }

    trials = MongoTrials('mongo://localhost:27017/hyperopt/jobs', exp_key='%s_v%d' % (name, version))

    print('Pending on workers to connect ..')
    argmin = fmin(fn=hyperopt_objective,
                space=space,
                algo=algo,
                max_evals=12,
                trials=trials,
                verbose=1)
    best_acc = 1-trials.best_trial['result']['loss']

    print('best val acc=', best_acc, 'params:', argmin