import sys

import mnist_keras
from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds

import warnings
# see https://stackoverflow.com/a/40846742
warnings.filterwarnings("ignore", message="numpy.dtype size changed")

args = mnist_keras.args

ex = Experiment('My_Experiment')
my_url = '127.0.0.1:27017'  # Or <server-static-ip>:<port> if running on server
curr_db_name = 'sacred_mnist'
ex.captured_out_filter = apply_backspaces_and_linefeeds

if args.disable_logging == False:
    ex.observers.append(MongoObserver.create(url=my_url,
                                             db_name=curr_db_name))

# noinspection PyUnusedLocal
@ex.config
def config():
    # Set SACRED configuration and the default values (it will override argparse
    # defaults). Note that we take only the hyper-params. Other arguments like
    # --verbose, or --gpu_memory_fraction are irrelevant for the model.
    seed = 0
    lr = 1e-3
    dropout_rate = 0.5
    fc_dim = 128
    epochs = 10
    batch_size = 32

@ex.main
def main(_run):
    cfg = _run.config
    print(cfg)

    # Override argparse arguments with sacred arguments
    command_line_args = mnist_keras.args # argparse command line arguments
    vars(command_line_args).update(cfg)
    # call main script
    val_acc, test_acc = mnist_keras.main(f_log_metrics=log_metrics)
    _run.info = dict(val_acc=val_acc, test_acc=test_acc)
    return val_acc

@ex.capture
def log_metrics(_run, logs):
    _run.log_scalar("loss", float(logs.get('loss')))
    _run.log_scalar("acc", float(logs.get('acc')))
    _run.log_scalar("val_loss", float(logs.get('val_loss')))
    _run.log_scalar("val_acc", float(logs.get('val_acc')))
    _run.result = float(logs.get('val_acc'))


if __name__ == '__main__':
    argv = sys.argv
    # Remove the argparse arguments
    sacred_argv = [arg for arg in argv if not arg.startswith('--')]
    ex.run_commandline(sacred_argv)

##### HyperOpt support #####################
from hyperopt import STATUS_OK
# noinspection PyUnresolvedReferences
import hyperopt_grid
def hyperopt_objective(params):
    config = {}

    if type(params) == dict:
        params = params.items()

    for (key, value) in params:
        if key in ['fc_dim']:
            value = int(value)
        config[key] = value
    run = ex.run(config_updates=config, )
    err = run.result
    return {'loss': 1 - err, 'status': STATUS_OK}

##### End HyperOpt support #################
