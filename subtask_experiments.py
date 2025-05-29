import numpy as np
import pickle
import time 

import eval_lib as eval
from eval_lib import (
    generate_sample_subtask_collection,
    get_intervals,
    get_intervals_using_subtasks,
    get_coverages_and_widths,
    
    freq_confidence_interval, 
    freq_confidence_interval_better, 
    bootstrap_confidence_interval, 
    bayes_credible_interval,

    clt_subtask_confidence_interval,
    bootstrap_subtask_confidence_interval,
    bayes_subtask_credible_interval_IS,
    bayes_subtask_credible_interval_IS_parallel,
)

def experiment_1_2(N_t, Ts, theta_vals, alphas, repeats=10, results_filename=None, seed=0, num_importance_samples=10_000, num_bootstrap_samples=10_000, verbose=True):
    n_vals = [N_t*t for t in Ts]
    n_per_tasks = [np.array([N_t]*t) for t in Ts]

    assert all(n_vals[i] == n_per_tasks[i].sum() for i in range(len(Ts)))

    # generate the data
    subtask_samples, subtask_samples_flat = generate_sample_subtask_collection(n_vals, theta_vals, n_per_tasks, repeats=repeats, seed=seed)

    # Add dictionary to store method times
    method_times = {}

    if verbose: print("Calculating intervals with bayes...", end=' ')
    start = time.time()
    bayes_intervals = get_intervals(n_vals, theta_vals, subtask_samples_flat, alphas, bayes_credible_interval)
    method_times['bayes'] = time.time() - start
    if verbose: print(f"(took {method_times['bayes']:.2f}s)")

    if verbose: print("Calculating intervals with freq...", end=' ')
    start = time.time()
    freq_intervals = get_intervals(n_vals, theta_vals, subtask_samples_flat, alphas, freq_confidence_interval)
    method_times['freq'] = time.time() - start
    if verbose: print(f"(took {method_times['freq']:.2f}s)")

    if verbose: print("Calculating intervals with wilson...", end=' ')
    start = time.time()
    wils_intervals = get_intervals(n_vals, theta_vals, subtask_samples_flat, alphas, freq_confidence_interval_better, method='wilson')
    method_times['wils'] = time.time() - start
    if verbose: print(f"(took {method_times['wils']:.2f}s)")

    if verbose: print("Calculating intervals with wilson and clopper-pearson...", end=' ')
    start = time.time()
    clop_intervals = get_intervals(n_vals, theta_vals, subtask_samples_flat, alphas, freq_confidence_interval_better, method='beta')
    method_times['clop'] = time.time() - start
    if verbose: print(f"(took {method_times['clop']:.2f}s)")

    if verbose: print("Calculating intervals with basic bootstrap...", end=' ')
    start = time.time()
    boot_intervals = get_intervals(n_vals, theta_vals, subtask_samples_flat, alphas, bootstrap_confidence_interval, K=num_bootstrap_samples)
    method_times['boot'] = time.time() - start
    if verbose: print(f"(took {method_times['boot']:.2f}s)")

    if verbose: print("Calculating intervals with clustered bootstrap...", end=' ')
    start = time.time()
    boot_subtask_intervals = get_intervals_using_subtasks(n_vals, theta_vals, subtask_samples, alphas, bootstrap_subtask_confidence_interval, K=num_bootstrap_samples)
    method_times['boot_subtask'] = time.time() - start
    if verbose: print(f"(took {method_times['boot_subtask']:.2f}s)")

    if verbose: print("Calculating intervals with importance sampling subtask...", end=' ')
    start = time.time()
    # non-parallel version
    IS_subtask_intervals = get_intervals_using_subtasks(n_vals, theta_vals, subtask_samples, alphas, bayes_subtask_credible_interval_IS, num_samples=num_importance_samples)
    
    # parallel version
    # IS_subtask_intervals = bayes_subtask_credible_interval_IS_parallel(subtask_samples, alphas, theta_vals, n_vals, num_samples=num_importance_samples)
    method_times['IS_subtask'] = time.time() - start
    if verbose: print(f"(took {method_times['IS_subtask']:.2f}s)")


    if verbose: print("Calculating intervals with (clustered) clt subtask...", end=' ')
    start = time.time()
    clt_subtask_intervals = get_intervals_using_subtasks(n_vals, theta_vals, subtask_samples, alphas, clt_subtask_confidence_interval)
    method_times['clt_subtask'] = time.time() - start
    if verbose: print(f"(took {method_times['clt_subtask']:.2f}s)")


    if verbose: print("Interval calculation done!")
    
    intervals = {'bayes': bayes_intervals, 'freq': freq_intervals,
                 'wils': wils_intervals, 'clop': clop_intervals, 
                 'boot': boot_intervals,
                 'boot_subtask': boot_subtask_intervals,
                #  'bayes_subtask': bayes_subtask_intervals,
                 'IS_subtask': IS_subtask_intervals,
                 'clt_subtask': clt_subtask_intervals}
    
    # process the results
    coverages, widths = get_coverages_and_widths(intervals, n_vals, theta_vals, alphas, repeats)

    # Save results in specially named variables
    results_dict = {
        # data
        'subtask_samples': subtask_samples,
        'subtask_samples_flat': subtask_samples_flat,

        # raw results
        'intervals': intervals, 

        # processed results
        'coverages': coverages, 
        'widths': widths,

        # hyperparameters 
        'n_vals': n_vals,
        'N_t': N_t,
        'Ts': Ts,
        'theta_vals': theta_vals,
        'alphas': alphas,
        'repeats': repeats,

        # method times
        'method_times': method_times
    }

    if results_filename is not None:
        with open(f"results/{results_filename}.pkl", 'wb') as f:
            pickle.dump(results_dict, f)

    return results_dict




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run subtask error-bar experiments with specified hyperparameters.')
    parser.add_argument('--experiment_name', type=str, default='subtask_small', help='Experiment setup to use')
    parser.add_argument('--seed', type=int, default=0, help='Random seed to use')
    parser.add_argument('--custom_name', type=str, default=None, help='Custom name to save results as')
    parser.add_argument('--num_repeats', type=int, default=200, help='Number of repeats to run')
    parser.add_argument('--num_alphas', type=int, default=100, help='Number of alphas to use')
    parser.add_argument('--num_thetas', type=int, default=100, help='Number of theta values to use')
    parser.add_argument('--num_importance_samples', type=int, default=10_000, help='Number of importance samples to use')
    parser.add_argument('--num_bootstrap_samples', type=int, default=10_000, help='Number of bootstrap samples to use')
    parser.add_argument('--prior_alpha', type=float, default=1, help='Alpha parameter for the prior')
    parser.add_argument('--prior_beta', type=float, default=1, help='Beta parameter for the prior')
    parser.add_argument('--fix_theta', type=float, default=None, help='Fix the value of theta')
    args = parser.parse_args()


    print(f"\nStart time: {time.asctime()}")
    print(args)
    print()

    np.random.seed(args.seed)

    alphas = np.linspace(0.005, 0.2, args.num_alphas)
    # theta_vals = np.linspace(0.01, 0.99, args.num_thetas)
    if args.fix_theta is None:
        theta_vals = np.random.beta(args.prior_alpha, args.prior_beta, size=(args.num_thetas,))
    else:
        assert args.num_thetas == 1
        theta_vals = np.array([args.fix_theta])

    repeats = args.num_repeats

    experiment_names = ['subtask_small', 'subtask_medium', 'subtask_large']
    experiment_N_ts  = {experiment_names[0]: 5, experiment_names[1]: 10, experiment_names[2]: 50}
    experiment_Ts    = {experiment_names[0]: [2,   6, 20,  60], 
                        experiment_names[1]: [3,  10, 30, 100], 
                        experiment_names[2]: [2,   6, 20,  60]}
    
    if args.experiment_name not in experiment_names:
        print(f"Experiment setup {args.experiment_name} not found.")
    else:
        custom_name = args.custom_name if args.custom_name is not None else args.experiment_name

        print(f"Running experiment {args.experiment_name} ({custom_name})...")
        results = experiment_1_2(experiment_N_ts[args.experiment_name], experiment_Ts[args.experiment_name], theta_vals, alphas, repeats, results_filename=custom_name, seed=args.seed, num_importance_samples=args.num_importance_samples, num_bootstrap_samples=args.num_bootstrap_samples)

        from plot_main_experiments import cov_vs_alpha_plot, cov_vs_width_plot, double_plot

        # cov_vs_alpha_plot(results, plot_filename=f"coverage_vs_alpha_{custom_name}")
        # cov_vs_width_plot(results, plot_filename=f"coverage_vs_width_{custom_name}")
        double_plot(results, plot_filename=f"combined_plot_{custom_name}", plot_type='subtask')

        print(f"Finished experiment {args.experiment_name} ({custom_name}).\n")

    print(f"\nEnd time: {time.asctime()}")
    


