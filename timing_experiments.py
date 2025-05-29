import numpy as np
import pickle
import time
import os
from pathlib import Path

from basic_experiments import experiment_1_1
from subtask_experiments import experiment_1_2
from paired_experiments import experiment_1_3

def run_timing_experiment(experiment_func, experiment_args, experiment_repeats=1000, results_filename=None):
    """
    Run timing experiments for a given experiment function.
    
    Args:
        experiment_func: The experiment function to time
        experiment_args: Dictionary of arguments to pass to the experiment function
        experiment_repeats: Number of times to repeat the timing experiment
        results_filename: Name for saving results
    """
    # Create results directory if it doesn't exist
    Path("results/timing_results").mkdir(parents=True, exist_ok=True)
    
    # Get method names from the first run
    print("Running initial experiment to get method names...")
    initial_results = experiment_func(**experiment_args)
    method_names = list(initial_results['intervals'].keys())
    
    # Get N values
    if 'n_vals' in experiment_args:
        n_vals = experiment_args['n_vals']
    elif 'N_t' in experiment_args and 'Ts' in experiment_args:
        N_t = experiment_args['N_t']
        Ts = experiment_args['Ts']
        n_vals = [N_t*t for t in Ts]
    
    # Initialize timing dictionary
    timing_results = {
        'method_times': {method: np.zeros((len(n_vals), experiment_repeats)) for method in method_names},
        'n_vals': experiment_args.get('n_vals'),
        'N_t': experiment_args.get('N_t'),
        'Ts': experiment_args.get('Ts'),
    }
    
    # Run timing experiments for each N value
    for n_idx, n in enumerate(n_vals):
        print(f"\nTiming experiments for N = {n}")
        
        # Create modified args with single n value
        if 'n_vals' in experiment_args:
            experiment_args_single_n = {**experiment_args, 'n_vals': [n]}
        else:  # subtask case
            experiment_args_single_n = {
                **experiment_args,
                'N_t': N_t,
                'Ts': [Ts[n_idx]]
            }
        
        # Run experiments
        print(f"Running {experiment_repeats} timing experiments...")
        for i in range(experiment_repeats):
            if i % 100 == 0:
                print(f"Progress: {i}/{experiment_repeats}")
                
            # Modify args for single repeat
            experiment_args_copy = experiment_args_single_n.copy()
            experiment_args_copy['repeats'] = 1
            experiment_args_copy['results_filename'] = None
            experiment_args_copy['verbose'] = False
            
            # Run experiment and record times
            results = experiment_func(**experiment_args_copy)
            
            # Extract execution times from the results
            for method in method_names:
                timing_results['method_times'][method][n_idx, i] = results.get('method_times', {}).get(method, 0)
    
    # Save results
    if results_filename:
        with open(f"results/timing_results/{results_filename}.pkl", 'wb') as f:
            pickle.dump(timing_results, f)
    
    return timing_results

def run_all_timing_experiments(experiment_repeats=1000, seed=0):
    """Run timing experiments for all three experiment types."""
    np.random.seed(seed)
    
    # Common parameters
    alphas = np.linspace(0.005, 0.2, 1)  # Reduced number for timing
    theta_vals = np.random.beta(1, 1, size=(1,))  # Reduced number for timing
    
    # Basic experiment timing
    print("\nRunning basic experiment timing...")
    n_vals = [3, 10, 30, 100]
    basic_args = {
        'n_vals': n_vals,
        'theta_vals': theta_vals,
        'alphas': alphas,
        'repeats': 1,
        'seed': seed,
        'num_bootstrap_samples': 10_000,
        'verbose': False,
    }
    basic_timing = run_timing_experiment(
        experiment_1_1, 
        basic_args,
        experiment_repeats=experiment_repeats,
        results_filename='basic_timing'
    )
    
    # Subtask experiment timing
    print("\nRunning subtask experiment timing...")
    N_t = 5
    Ts = [2, 6, 20, 60]
    subtask_args = {
        'N_t': N_t,
        'Ts': Ts,
        'theta_vals': theta_vals,
        'alphas': alphas,
        'repeats': 1,
        'seed': seed,
        'num_importance_samples': 10_000,
        'num_bootstrap_samples': 10_000,
        'verbose': False,
    }
    subtask_timing = run_timing_experiment(
        experiment_1_2,
        subtask_args,
        experiment_repeats=experiment_repeats,
        results_filename='subtask_timing'
    )
    
    # Paired experiment timing
    print("\nRunning paired experiment timing...")
    theta_vals_A = np.random.beta(1, 1, size=(1,))
    theta_vals_B = np.random.beta(1, 1, size=(1,))
    paired_args = {
        'n_vals': n_vals,
        'theta_vals_A': theta_vals_A,
        'theta_vals_B': theta_vals_B,
        'alphas': alphas,
        'repeats': 1,
        'seed': seed,
        'num_importance_samples': 10_000,
        'num_bootstrap_samples': 10_000,
        'simulation_method': 'gaussian',
        'verbose': False,
    }
    paired_timing = run_timing_experiment(
        experiment_1_3,
        paired_args,
        experiment_repeats=experiment_repeats,
        results_filename='paired_timing'
    )
    
    return {
        'basic': basic_timing,
        'subtask': subtask_timing,
        'paired': paired_timing
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run timing experiments for all methods.')
    parser.add_argument('--experiment_repeats', type=int, default=1000, 
                        help='Number of times to repeat each timing experiment')
    parser.add_argument('--seed', type=int, default=0, 
                        help='Random seed to use')
    args = parser.parse_args()
    
    print(f"\nStart time: {time.asctime()}")
    print(args)
    print()
    
    timing_results = run_all_timing_experiments(
        experiment_repeats=args.experiment_repeats,
        seed=args.seed
    )
    
    print(f"\nEnd time: {time.asctime()}") 