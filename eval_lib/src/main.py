import numpy as np

from .intervals import *
from .intervals_clustered import *
from .intervals_paired import *
from .utils import flatten_subtask_data

__all__ = ['compute_error_bars']

def compute_error_bars(eval_data, alpha, verbose=False, eval_setting='basic', eval_data_paired=None,):
    '''
    Compute confidence/credible intervals for the given evaluation data under the given setting (basic, subtask, paired)
    using ALL POSSIBLE methods for that seting.

    Parameters
    ----------
    eval_data : np.array or list(np.array)
        The evaluation data for a given LLM.
    alpha : float
        The alpha value for which the error bars should be computed (at a 1-alpha confidence level).
    verbose : bool
        Whether to print the method name and confidence interval for each method.
    eval_setting : str
        The evaluation setting. One of 'basic', 'subtask', and 'paired'.
    eval_data_paired : np.array
        The paired evaluation data (data_B) for the paired evaluation setting.

    Returns
    -------
    dict(str, np.array)
        key: Method name.
        value: Confidence/credible interval for the given evaluation data under the given method.
    '''
    basic_methods = {'bayes': bayes_credible_interval,
                     'freq': freq_confidence_interval,
                     'boot': bootstrap_confidence_interval,
                     'wils': lambda *args: freq_confidence_interval_better(*args, method="wilson"),
                     'clop': lambda *args: freq_confidence_interval_better(*args, method="beta"),
                     }

    subtask_methods = {'IS_subtask': lambda *args: bayes_subtask_credible_interval_IS(*args, num_samples=100_000),
                       'clt_subtask': clt_subtask_confidence_interval,
                       'boot_subtask': bootstrap_subtask_confidence_interval}

    paired_methods = {'bayes_paired': importance_sampled_paired_credible_interval,
                      'clt_paired': clt_paired_confidence_interval,
                      'freq': clt_unpaired_confidence_interval,
                      'boot': bootstrap_paired_confidence_interval}

    if eval_setting == 'subtask':
        assert isinstance(eval_data, list)
        # convert the basic methods into functions that can flatten subtask data
        non_subtask_methods = {name: lambda data_list, *args: func(flatten_subtask_data(data_list), *args)  for name, func in basic_methods.items()}
        methods = {**subtask_methods, **non_subtask_methods}
    elif eval_setting == 'paired':
        assert eval_data_paired is not None
        methods = {name: lambda data_A, *args: func(data_A, eval_data_paired, *args) for name, func in paired_methods.items()}
    else:
        methods = basic_methods
    
    if isinstance(alpha, float):
        alpha = [alpha]
        single_alpha = True
    else:
        alpha = list(alpha)
        single_alpha = False
    
    errors = {name: np.zeros((len(alpha), 2)) for name in methods.keys()}
    for method_name, method in methods.items():
        interval = method(eval_data, alpha)
        # if single_alpha:
        #     interval = interval[0]

        if verbose:
            print(method_name, interval)
        
        errors[method_name] = interval

    return errors