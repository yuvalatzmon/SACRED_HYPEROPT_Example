import warnings
import numpy as np
import json

from hyperopt import pyll, hp
from hyperopt.base import miscs_update_idxs_vals
from hyperopt.pyll import scope

def pow10_and_round(v):
    """
    This calculates the power of 10, and rounds by its decimal precision
    e.g.
    -0.5 --> 10**-0.5 --> 0.316  --> 0.3
    -1.5 --> 10**-1.5 --> 0.0316 --> 0.03
    0.25 --> 10**0.25 --> 1.77   --> 2
    1.25 --> 10**1.25 --> 17.78  --> 20
    """
    return np.round(10**v, -np.int32(np.floor(v)))

def dict_to_sorted_str(d):
    return json.dumps(d, sort_keys=True)

def to_exp_space(log_values):
    if type(log_values) == dict:
        return dict([(k, pow10_and_round(v)) for k, v in log_values.items()])
    elif type(log_values) in [tuple, list]:
        return tuple(pow10_and_round(v) for v in log_values)
    else:
        raise RuntimeError()

@scope.define
def f_grid(*args):
    return [(v[0], pow10_and_round(v[1])) for v in args]

class hyperopt_grid():
    def __init__(self, grid_log_ranges):
        self._grid_log_ranges = grid_log_ranges
        self.space = self._get_grid_space()
        self.num_combinations = self._len_outer_product()
        self.executed_params = set()
        self._cnt = 0
        self._cnt_skip = 0
        print('self.num_combinations ', self.num_combinations)


    def _get_grid_space(self):
        log_space = tuple([(key, hp.quniform(key, *range_) ) for key, range_ in
                           self._grid_log_ranges.items()])
        return scope.f_grid(*log_space)

    def _len_outer_product(self):
        # r[1]+1 because quniform range includes the last item
        ranges_elements = tuple(np.arange(r[0], r[1] + r[2], r[2]) for r in
                                self._grid_log_ranges.values())

        return np.prod([len(range_) for range_ in ranges_elements])

    @staticmethod
    def _convert_neg_zeros_to_zeros(trial_dict):
        """ Helps to avoid counting floating zero twice (as +0.0 and -0.0) """
        for k in trial_dict.keys():
            if trial_dict[k][0] == 0 and not isinstance(trial_dict[k][0], (int, np.integer)):
                trial_dict[k] = [+0.0]
        return trial_dict

    @staticmethod
    def _get_historical_params(trials):
        historical_params = []
        for k, trial in enumerate(trials.trials):
            if trials.statuses()[k] == 'ok':
                current_run_params = hyperopt_grid._convert_neg_zeros_to_zeros(
                    dict(trial['misc']['vals']))

                historical_params.append(
                    dict_to_sorted_str(current_run_params))
        return historical_params

    def suggest(self, new_ids, domain, trials, seed):
        rng = np.random.RandomState(seed)
        rval = []    # print('new_ids', new_ids)
        for ii, new_id in enumerate(new_ids):
            while self._cnt <= self.num_combinations:
                # -- sample new specs, idxs, vals
                idxs, vals = pyll.rec_eval(
                    domain.s_idxs_vals,
                    memo={
                        domain.s_new_ids: [new_id],
                        domain.s_rng: rng,
                    })
                new_result = domain.new_result()
                new_misc = dict(tid=new_id, cmd=domain.cmd, workdir=domain.workdir)
                miscs_update_idxs_vals([new_misc], idxs, vals)
                new_trial = trials.new_trial_docs([new_id],
                            [None], [new_result], [new_misc])
                # Except the `while`, until here, code is copied from rand.suggest

                # new code from here
                self.executed_params = self.executed_params.union(
                    self._get_historical_params(trials))

                # avoid counting floating zero twice (as +0.0 and -0.0)
                this_run_params = hyperopt_grid._convert_neg_zeros_to_zeros(
                    dict(new_misc['vals']))
                # represent the params as a hashed string
                this_run_params_str = dict_to_sorted_str(this_run_params)

                # if these params are seen for the first time, then generate a new
                # trial for them
                if this_run_params_str not in self.executed_params:

                    # add the new trial to returned list
                    rval.extend(new_trial)

                    # log the new trial as executed, in order to avoid duplication
                    self._cnt += 1
                    self.executed_params = \
                        self.executed_params.union([this_run_params_str])
                    print(self._cnt, this_run_params)
                    break
                else:
                    # otherwise (params were seen), skip this trial
                    # update internal counter
                    self._cnt_skip += 1

                # Stopping condition (breaking the hyperopt loop)
                if len(self.executed_params) >= self.num_combinations:
                    # returning an empty list, breaks the hyperopt loop
                    return []


                # "Emergency" stopping condition, breaking the hyperopt loop when
                # loop runs for too long without submitted experiments
                if self._cnt_skip >= 100*self.num_combinations:
                    warnings.warn('Warning: Exited due to too many skips.'
                          ' This can happen if most of the param combinationa have '
                                  'been encountered, and drawing a new '
                                  'unseen combination, involves a very low probablity.')
                    # returning an empty list, breaks the hyperopt loop
                    return []

        return rval
