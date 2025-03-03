import numpy as np
import pickle

import eval_lib as eval
from eval_lib import (
    generate_sample_collection,
    get_intervals,
    get_intervals_using_subtasks,
    get_coverages_and_widths,
    
    freq_confidence_interval, 
    freq_confidence_interval_better, 
    bootstrap_confidence_interval, 
    bayes_credible_interval,
)

def experiment_1_1(n_vals, theta_vals, alphas, repeats=10, results_filename=None, seed=0, num_bootstrap_samples=10_000):
    # generate the data
    samples = generate_sample_collection(n_vals, theta_vals, repeats=repeats, seed=seed)

    # calculate the intervals
    print("Calculating intervals with bayes and freq...")
    bayes_intervals = get_intervals(n_vals, theta_vals, samples, alphas, bayes_credible_interval)
    freq_intervals = get_intervals(n_vals, theta_vals, samples, alphas, freq_confidence_interval)

    print("Calculating intervals with wilson and clopper-pearson...")
    wils_intervals = get_intervals(n_vals, theta_vals, samples, alphas, freq_confidence_interval_better, method='wilson')
    clop_intervals = get_intervals(n_vals, theta_vals, samples, alphas, freq_confidence_interval_better, method='beta')

    print("Calculating intervals with basic bootstrap...")
    boot_intervals = get_intervals(n_vals, theta_vals, samples, alphas, bootstrap_confidence_interval, K=num_bootstrap_samples)

    print("Interval calculation done!")
    
    intervals = {'bayes': bayes_intervals, 'freq': freq_intervals,
                 'wils': wils_intervals, 'clop': clop_intervals, 
                 'boot': boot_intervals,}
    
    # process the results
    coverages, widths = get_coverages_and_widths(intervals, n_vals, theta_vals, alphas, repeats)

    # Save results in specially named variables
    results_dict = {
        # data
        'samples': samples,

        # raw results
        'intervals': intervals, 

        # processed results
        'coverages': coverages, 
        'widths': widths,

        # hyperparameters 
        'n_vals': n_vals,
        'theta_vals': theta_vals,
        'alphas': alphas,
        'repeats': repeats
    }

    if results_filename is not None:
        with open(f"results/{results_filename}.pkl", 'wb') as f:
            pickle.dump(results_dict, f)

    return results_dict




if __name__ == "__main__":
    import argparse
    import time 
    parser = argparse.ArgumentParser(description='Run subtask error-bar experiments with specified hyperparameters.')
    parser.add_argument('--experiment_name', type=str, default='basic_small', help='Experiment setup to use')
    parser.add_argument('--seed', type=int, default=0, help='Random seed to use')
    parser.add_argument('--custom_name', type=str, default=None, help='Custom name to save results as')
    parser.add_argument('--num_repeats', type=int, default=200, help='Number of repeats to run')
    parser.add_argument('--num_alphas', type=int, default=100, help='Number of alphas to use')
    parser.add_argument('--num_thetas', type=int, default=100, help='Number of theta values to use')
    parser.add_argument('--num_bootstrap_samples', type=int, default=10_000, help='Number of bootstrap samples to use')
    parser.add_argument('--prior_alpha', type=float, default=1, help='Alpha parameter for the prior')
    parser.add_argument('--prior_beta', type=float, default=1, help='Beta parameter for the prior')
    parser.add_argument('--fix_theta', type=float, default=None, help='Fix the value of theta')
    args = parser.parse_args()

    import os 
    os.makedirs('results', exist_ok=True)
    os.makedirs('plots/pdfs', exist_ok=True)
    os.makedirs('plots/pngs', exist_ok=True)

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

    experiment_names = ['basic_small', 'basic_medium']
    experiment_n_vals = {
        'basic_small': [3, 10, 30, 100],
        'basic_medium': [10, 10, 100, 300],
    }
    
    if args.experiment_name not in experiment_names:
        print(f"Experiment setup {args.experiment_name} not found.")
    else:
        custom_name = args.custom_name if args.custom_name is not None else args.experiment_name

        print(f"Running experiment {args.experiment_name} ({custom_name})...")
        results = experiment_1_1(experiment_n_vals[args.experiment_name], theta_vals, alphas, repeats, results_filename=custom_name, seed=args.seed, num_bootstrap_samples=args.num_bootstrap_samples)

        from plot_main_experiments import cov_vs_alpha_plot, cov_vs_width_plot, double_plot

        # cov_vs_alpha_plot(results, plot_filename=f"coverage_vs_alpha_{custom_name}")
        # cov_vs_width_plot(results, plot_filename=f"coverage_vs_width_{custom_name}")
        double_plot(results, plot_filename=f"combined_plot_{custom_name}", plot_type='basic')

        print(f"Finished experiment {args.experiment_name} ({custom_name}).\n")

    print(f"\nEnd time: {time.asctime()}")
    


