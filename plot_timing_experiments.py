import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path

from eval_lib import COLOURS, METHOD_NAMES

# Enable LaTeX
plt.rcParams['text.usetex'] = True
plt.style.use(['default'])

def set_font_sizes():
    """Set font sizes for the plots."""
    plt.rcParams.update({
        'font.size': 8,
        'axes.titlesize': 9,
        'axes.labelsize': 9,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 7,
    })

def plot_timing_results(timing_results, plot_filename=None, plot_type='basic'):
    """
    Plot timing results for a given experiment type.
    
    Args:
        timing_results: Dictionary containing method_times and n_vals/N_t/Ts
        plot_filename: Name for saving the plot
        plot_type: Type of experiment ('basic', 'subtask', or 'paired')
    """
    set_font_sizes()
    
    # Get N values from the results
    if plot_type == 'subtask':
        N_t = timing_results['N_t']
        Ts = timing_results['Ts']
        n_vals = [N_t*t for t in Ts]
    else:
        n_vals = timing_results['n_vals']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Figure out which methods we're plotting
    all_methods = ['IS_subtask', 'bayes', 'clt_subtask', 'freq', 'wils', 'clop', 'boot_subtask', 'boot']
    unclustered_methods = ['bayes', 'freq', 'wils', 'clop', 'boot']
    paired_methods = ['bayes_paired', 'bayes_unpaired', 'clt_paired', 'freq', 'boot']
    
    if plot_type == 'subtask':
        methods = [m for m in all_methods if m in timing_results['method_times']]
    elif plot_type == 'paired':
        methods = [m for m in paired_methods if m in timing_results['method_times']]
    else:
        methods = [m for m in unclustered_methods if m in timing_results['method_times']]
    
    dashed_methods = unclustered_methods + ['bayes_unpaired']
    
    # Plot timing results for each method
    n_vals_str = [str(n) for n in n_vals]
    for method in methods:
        # Get times for each N value
        times = timing_results['method_times'][method]  # shape: (len(n_vals), experiment_repeats)
        mean_times = np.mean(times, axis=1) * 1000  # average over repeats and convert to milliseconds
        std_times = np.std(times, axis=1) * 1000  # std over repeats and convert to milliseconds
        se_times = std_times / np.sqrt(times.shape[1])  # standard error
        
        # Plot with error bars
        ax.errorbar(n_vals_str, mean_times, yerr=se_times,
                   label=METHOD_NAMES[method],
                   color=COLOURS[method],
                   linestyle=':' if method in dashed_methods and plot_type in ('subtask', 'paired') else '-',
                   alpha=0.5 if method in dashed_methods and plot_type in ('subtask', 'paired') else 1,
                   capsize=3)
    
    # Customize plot
    if plot_type == 'subtask':
        ax.set_xlabel(f'$N = N_t \\times T$\n$(N_t = {N_t},\\ T = {",\\ ".join(map(str, Ts))}$)')
    else:
        ax.set_xlabel('$N$')
    ax.set_ylabel('Execution Time (ms)')
    ax.set_xticks(n_vals_str)
    ax.grid(True, alpha=0.3)
    
    # Add legend
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0))
    
    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    
    # Save plot if filename provided
    if plot_filename:
        # Create directories if they don't exist
        Path("plots/pdfs/timing_plots").mkdir(parents=True, exist_ok=True)
        Path("plots/pngs/timing_plots").mkdir(parents=True, exist_ok=True)
        
        plt.savefig(f"plots/pdfs/timing_plots/{plot_filename}.pdf", bbox_inches='tight')
        plt.savefig(f"plots/pngs/timing_plots/{plot_filename}.png", bbox_inches='tight')
        print(f"Saved timing plots to plots/{{pdfs,pngs}}/timing_plots/{plot_filename}.{{pdf,png}}")
    else:
        plt.show() 

def plot_combined_timing_results(timing_results_dict, plot_filename=None):
    """
    Plot timing results for all experiment types in a single figure with three subplots.
    
    Args:
        timing_results_dict: Dictionary mapping experiment types to timing results
        plot_filename: Name for saving the plot
    """
    set_font_sizes()
    
    # Create figure with three subplots in a row
    fig, axs = plt.subplots(1, 3, figsize=(6.75, 3.25))
    
    # Define experiment types and their titles
    experiment_types = ['basic', 'subtask', 'paired']
    subplot_titles = ['IID', 'Clustered', 'Paired']
    
    # Process each experiment type
    for i, (exp_type, title) in enumerate(zip(experiment_types, subplot_titles)):
        timing_results = timing_results_dict[exp_type]
        ax = axs[i]
        
        # Get N values from the results
        if exp_type == 'subtask':
            N_t = timing_results['N_t']
            Ts = timing_results['Ts']
            n_vals = [N_t*t for t in Ts]
        else:
            n_vals = timing_results['n_vals']
        
        # Figure out which methods we're plotting
        all_methods = ['IS_subtask', 'bayes', 'clt_subtask', 'freq', 'wils', 'clop', 'boot_subtask', 'boot']
        unclustered_methods = ['bayes', 'freq', 'wils', 'clop', 'boot']
        paired_methods = ['bayes_paired', 'bayes_unpaired', 'clt_paired', 'freq', 'boot']
        
        if exp_type == 'subtask':
            methods = [m for m in all_methods if m in timing_results['method_times']]
        elif exp_type == 'paired':
            methods = [m for m in paired_methods if m in timing_results['method_times']]
        else:
            methods = [m for m in unclustered_methods if m in timing_results['method_times']]
        
        dashed_methods = unclustered_methods + ['bayes_unpaired']
        
        # Plot timing results for each method
        n_vals_str = [str(n) for n in n_vals]
        for method in methods:
            # Get times for each N value
            times = timing_results['method_times'][method]  # shape: (len(n_vals), experiment_repeats)
            mean_times = np.mean(times, axis=1) * 1000  # average over repeats and convert to milliseconds
            std_times = np.std(times, axis=1) * 1000  # std over repeats and convert to milliseconds
            se_times = std_times / np.sqrt(times.shape[1])  # standard error
            
            # Plot with error bars
            ax.errorbar(n_vals_str, mean_times, yerr=se_times,
                       label=METHOD_NAMES[method],
                       color=COLOURS[method],
                       linestyle=':' if method in dashed_methods and exp_type in ('subtask', 'paired') else '-',
                       alpha=0.5 if method in dashed_methods and exp_type in ('subtask', 'paired') else 1,
                       capsize=3)
        
        # Customize subplot
        if exp_type == 'subtask':
            ax.set_xlabel(f'$N = N_t \\times T$\n$(N_t = {N_t},\\ T = {",\\ ".join(map(str, Ts))}$)')
        else:
            ax.set_xlabel('$N$')
        
        if i == 0:  # Only add y-label to the first subplot
            ax.set_ylabel('Execution Time (ms)')
        
        ax.set_title(title)
        ax.set_xticks(n_vals_str)
        ax.grid(True, alpha=0.3)
        
        # Use log scale for y-axis if there's a large range of values
        if np.max(mean_times) / np.min(mean_times) > 100:
            ax.set_yscale('log')
            
        # Add legend to each subplot
        ax.legend(loc='center left' if exp_type == 'paired' else 'upper left', frameon=True)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot if filename provided
    if plot_filename:
        # Create directories if they don't exist
        Path("plots/pdfs/timing_plots").mkdir(parents=True, exist_ok=True)
        Path("plots/pngs/timing_plots").mkdir(parents=True, exist_ok=True)
        
        plt.savefig(f"plots/pdfs/timing_plots/{plot_filename}.pdf", bbox_inches='tight')
        plt.savefig(f"plots/pngs/timing_plots/{plot_filename}.png", bbox_inches='tight')
        print(f"Saved combined timing plot to plots/{{pdfs,pngs}}/timing_plots/{plot_filename}.{{pdf,png}}")
    else:
        plt.show()

if __name__ == "__main__":
    # Load timing results for all experiment types
    experiment_types = ['basic', 'subtask', 'paired']
    timing_results_dict = {}
    
    for exp_type in experiment_types:
        print(f"Loading {exp_type} timing results...")
        
        # Load timing results
        with open(f"results/timing_results/{exp_type}_timing.pkl", "rb") as f:
            timing_results_dict[exp_type] = pickle.load(f)
    
    # Create combined timing plot
    plot_combined_timing_results(timing_results_dict, plot_filename="combined_timing")
    
    # Also create individual plots for backward compatibility
    for exp_type in experiment_types:
        print(f"\nProcessing {exp_type} timing results...")
        
        # Create timing plot
        plot_timing_results(timing_results_dict[exp_type], 
                          plot_filename=f"{exp_type}_timing",
                          plot_type=exp_type)
