import numpy as np
import pickle
import time

import eval_lib as eval
from eval_lib import (
    generate_sample_collection,
    generate_sample_collection_paired_gaussian_prior,
    generate_sample_collection_paired_dirichlet_prior,
    generate_sample_collection_paired_model_prior,
    get_intervals_using_pairing,
    get_coverages_and_widths,
    
    clt_unpaired_confidence_interval,
    clt_paired_confidence_interval,
    bootstrap_paired_confidence_interval,

    importance_sampled_paired_credible_interval,
    bayes_unpaired_credible_interval,
)

def experiment_1_3(n_vals, theta_vals_A, theta_vals_B, alphas, repeats=10, results_filename=None, seed=0, num_importance_samples=10_000, num_bootstrap_samples=10_000, simulation_method='gaussian', rho_min=-1, rho_max=1):
    assert len(theta_vals_A) == len(theta_vals_B)

    # generate the data
    if simulation_method == 'gaussian':
        samples_A, samples_B, rhos = generate_sample_collection_paired_gaussian_prior(n_vals, theta_vals_A, theta_vals_B, repeats=repeats, seed=seed, rho_min=args.rho_min, rho_max=args.rho_max)
    elif simulation_method == 'dirichlet':
        # NOTE: DIRICHLET PRIOR OVERWRITES THETA_VALS
        print("NOTE: Dirichlet prior overwrites theta_vals; fixed theta values won't work.")
        samples_A, samples_B, theta_vals_A, theta_vals_B = generate_sample_collection_paired_dirichlet_prior(n_vals, theta_vals_A, theta_vals_B, repeats=repeats, seed=seed)
        assert theta_vals_A.shape == theta_vals_B.shape
        assert len(theta_vals_A.shape) == 1
    elif simulation_method == 'per-question':
        samples_A, samples_B = generate_sample_collection_paired_model_prior(n_vals, theta_vals_A, theta_vals_B, repeats=repeats, seed=seed)
    elif simulation_method == 'independent':
        samples_A = generate_sample_collection(n_vals, theta_vals_A, repeats=repeats, seed=seed)
        samples_B = generate_sample_collection(n_vals, theta_vals_B, repeats=repeats, seed=seed+1)

    # calculate the intervals
    print("Calculating intervals with freq (unpaired CLT)...", end=' ')
    start = time.time()
    freq_intervals = get_intervals_using_pairing(n_vals, len(theta_vals_A), samples_A, samples_B, alphas, clt_unpaired_confidence_interval)
    print(f"(took {time.time() - start:.2f}s)")

    print("Calculating intervals with paired CLT...", end=' ')
    start = time.time()
    clt_intervals = get_intervals_using_pairing(n_vals, len(theta_vals_A), samples_A, samples_B, alphas, clt_paired_confidence_interval)
    print(f"(took {time.time() - start:.2f}s)")

    print("Calculating intervals with bayes (IS)...", end=' ')
    start = time.time()
    bayes_intervals = get_intervals_using_pairing(n_vals, len(theta_vals_A), samples_A, samples_B, alphas, importance_sampled_paired_credible_interval, num_samples=num_importance_samples)
    print(f"(took {time.time() - start:.2f}s)")

    print("Calculating intervals with basic bootstrap...", end=' ')
    start = time.time()
    boot_intervals = get_intervals_using_pairing(n_vals, len(theta_vals_A), samples_A, samples_B, alphas, bootstrap_paired_confidence_interval, K=num_bootstrap_samples)
    print(f"(took {time.time() - start:.2f}s)")

    print("Calculating intervals with bayes (unpaired model)...", end=' ')
    start = time.time()
    bayes_unpaired_intervals = get_intervals_using_pairing(n_vals, len(theta_vals_A), samples_A, samples_B, alphas, bayes_unpaired_credible_interval, num_samples=num_importance_samples)
    print(f"(took {time.time() - start:.2f}s)")


    print("Interval calculation done!")
    
    intervals = {'freq': freq_intervals,
                 'clt_paired': clt_intervals,
                 'bayes_paired': bayes_intervals, 
                 'bayes_unpaired': bayes_unpaired_intervals,
                 'boot': boot_intervals,
    }
    
    # process the results
    theta_diff_vals = theta_vals_A - theta_vals_B
    coverages, widths = get_coverages_and_widths(intervals, n_vals, theta_diff_vals, alphas, repeats)

    # Save results in specially named variables
    results_dict = {
        # data
        'samples_A': samples_A,
        'samples_B': samples_B,

        # raw results
        'intervals': intervals, 

        # processed results
        'coverages': coverages, 
        'widths': widths,

        # hyperparameters 
        'n_vals': n_vals,
        'theta_vals_A': theta_vals_A,
        'theta_vals_B': theta_vals_B,
        'alphas': alphas,
        'repeats': repeats,

        'rhos': rhos,
    }

    if results_filename is not None:
        with open(f"results/{results_filename}.pkl", 'wb') as f:
            pickle.dump(results_dict, f)

    return results_dict


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run subtask error-bar experiments with specified hyperparameters.')
    parser.add_argument('--experiment_name', type=str, default='paired_small', help='Experiment setup to use')
    parser.add_argument('--seed', type=int, default=0, help='Random seed to use')
    parser.add_argument('--custom_name', type=str, default=None, help='Custom name to save results as')
    parser.add_argument('--num_repeats', type=int, default=200, help='Number of repeats to run')
    parser.add_argument('--num_alphas', type=int, default=100, help='Number of alphas to use')
    parser.add_argument('--num_thetas', type=int, default=100, help='Number of theta values to use')
    parser.add_argument('--num_importance_samples', type=int, default=10_000, help='Number of importance samples to use')
    parser.add_argument('--num_bootstrap_samples', type=int, default=10_000, help='Number of bootstrap samples to use')
    parser.add_argument('--prior_alpha_A', type=float, default=1, help='Alpha parameter for the prior')
    parser.add_argument('--prior_beta_A', type=float, default=1, help='Beta parameter for the prior')
    parser.add_argument('--prior_alpha_B', type=float, default=1, help='Alpha parameter for the prior')
    parser.add_argument('--prior_beta_B', type=float, default=1, help='Beta parameter for the prior')
    parser.add_argument('--simulation_method', type=str, default="gaussian", help='Method to generate eval data for the experiment')
    parser.add_argument('--fix_theta_A', type=float, default=None, help='Fix the value of theta_A')
    parser.add_argument('--fix_theta_B', type=float, default=None, help='Fix the value of theta_B')
    parser.add_argument('--rho_min', type=float, default=-1, help='Minimum value for rho in 2D Gaussian model')
    parser.add_argument('--rho_max', type=float, default=1, help='Maximum value for rho in 2D Gaussian model')
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
    if args.fix_theta_A is None:
        assert args.fix_theta_B is None

        theta_vals_A = np.random.beta(args.prior_alpha_A, args.prior_beta_B, size=(args.num_thetas,))
        theta_vals_B = np.random.beta(args.prior_alpha_B, args.prior_beta_B, size=(args.num_thetas,))
    else:
        assert args.num_thetas == 1
        assert args.fix_theta_B is not None
        
        theta_vals_A = np.array([args.fix_theta_A])
        theta_vals_B = np.array([args.fix_theta_B])
    

    repeats = args.num_repeats

    experiment_names = ['paired_small', 'paired_medium']
    experiment_n_vals = {
        'paired_small': [3, 10, 30, 100],
        'paired_medium': [10, 10, 100, 300],
    }
    
    if args.experiment_name not in experiment_names:
        print(f"Experiment setup {args.experiment_name} not found.")
    else:
        custom_name = args.custom_name if args.custom_name is not None else args.experiment_name

        print(f"Running experiment {args.experiment_name} ({custom_name})...")
        results = experiment_1_3(experiment_n_vals[args.experiment_name], theta_vals_A, theta_vals_B, alphas, repeats, results_filename=custom_name, seed=args.seed, num_importance_samples=args.num_importance_samples, num_bootstrap_samples=args.num_bootstrap_samples, simulation_method=args.simulation_method, rho_min=args.rho_min, rho_max=args.rho_max)

        from plot_main_experiments import cov_vs_alpha_plot, cov_vs_width_plot, double_plot

        # cov_vs_alpha_plot(results, plot_filename=f"coverage_vs_alpha_{custom_name}")
        # cov_vs_width_plot(results, plot_filename=f"coverage_vs_width_{custom_name}")
        double_plot(results, plot_filename=f"combined_plot_{custom_name}", plot_type='paired')

        print(f"Finished experiment {args.experiment_name} ({custom_name}).\n")

    print(f"\nEnd time: {time.asctime()}")
    


