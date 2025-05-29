import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams['text.usetex'] = True
plt.style.use(['default'])

# Set font sizes
plt.rc('font', size=8)          # controls default text sizes
plt.rc('axes', titlesize=10)    # fontsize of the axes title
plt.rc('axes', labelsize=7)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=7)    # fontsize of the tick labels
plt.rc('ytick', labelsize=7)    # fontsize of the tick labels
plt.rc('legend', fontsize=7)    # legend fontsize
plt.rc('figure', titlesize=9)   # fontsize of the figure title

from eval_lib import COLOURS, METHOD_NAMES, compute_error_bars

empirical_means_line_style = "solid"
empirical_mean_style_kwargs = {
    'linewidth': 1,
    'color': 'black',
    'alpha': 1,
    'zorder': 100
}

basic_method_names_ordered = ["bayes", "freq", "wils", "clop", "boot"]
subtask_method_names_ordered = ["IS_subtask", "clt_subtask", "boot_subtask", "bayes", "freq", "boot", "wils", "clop"]
subtask_ONLY_method_names_ordered = ["IS_subtask", "clt_subtask", "boot_subtask"]

method_names_ordered = basic_method_names_ordered

COLOURS['IS_subtask'] = 'darkblue'
COLOURS['clt_subtask'] = 'orangered'
COLOURS['boot_subtask'] = '#653700'

langchain_nice_llm_names_all = {
    'claude-2.1': 'Claude-2.1',
    'gpt-4-1106-preview (functions)': r'GPT-4$^\dagger$',
    'mixtral-8x7b-instruct': 'Mixtral-8x7B',
    'gpt-3.5-turbo-0613-openai (functions)': r'GPT-3.5$^\dagger$',
    'gpt-4-0613 (functions)': r'GPT-4$^\ddagger$',
    'gpt-3.5-turbo-1106 (functions)': r'GPT-3.5$^\ddagger$', 
    'llama-v2-70b-chat': 'Llama-2-70B',
    'mistral-7b-instruct': 'Mistral-7B',
    'llama-v2-13b-chat': 'Llama-2-13B',
}

langchain_nice_llm_names_subset = {
    'claude-2.1': 'Claude-2.1',
    'gpt-4-1106-preview (functions)': 'GPT-4',
    'mixtral-8x7b-instruct': 'Mixtral 8x7B',
    'gpt-3.5-turbo-0613-openai (functions)': 'GPT-3.5',
    # 'gpt-4-0613 (functions)': r'GPT-4$^\ddagger$',
    # 'gpt-3.5-turbo-1106 (functions)': r'GPT-3.5$^\ddagger$', 
    'llama-v2-70b-chat': 'Llama-2-70B',
    'mistral-7b-instruct': 'Mistral-7B',
    # 'llama-v2-13b-chat': 'Llama-2-13B',
}

# Matharena AIME II
matharena_nice_llm_names_all = {
    'o3-mini (high)': 'o3-mini (high)',
    'DeepSeek-R1': 'DeepSeek-R1',
    'o3-mini (medium)': 'o3-mini (medium)',
    'o1 (medium)': 'o1 (medium)',
    'QwQ-32B*': 'QwQ-32B',
    'DeepSeek-R1-Distill-32B': 'DeepSeek-R1-Distill-32B',
    'DeepSeek-R1-Distill-70B': 'DeepSeek-R1-Distill-70B',
    'gemini-2.0-flash-thinking': 'gemini-2.0-flash-thinking',
    'DeepSeek-R1-Distill-14B': 'DeepSeek-R1-Distill-14B',
    'DeepSeek-V3-03-24*': 'DeepSeek-V3-03-24',
    'Claude-3.7-Sonnet (Thinking)*': 'Claude-3.7-Sonnet (Thinking)',
    'gemini-2.0-pro': 'gemini-2.0-pro',
    'QwQ-32B-Preview': 'QwQ-32B-Preview',
    'o3-mini (low)': 'o3-mini (low)',
    'gemini-2.0-flash': 'gemini-2.0-flash',
    'DeepSeek-V3': 'DeepSeek-V3',
    'DeepSeek-R1-Distill-1.5B': 'DeepSeek-R1-Distill-1.5B',
    'gpt-4o': 'gpt-4o',
    'Claude-3.5-Sonnet': 'Claude-3.5-Sonnet'
}
matharena_nice_llm_names_all_original_order = {
    'o3-mini (high)': 'o3-mini (high)',
    'o3-mini (medium)': 'o3-mini (medium)',
    'o1 (medium)': 'o1 (medium)',
    'DeepSeek-R1': 'DeepSeek-R1',
    'QwQ-32B*': 'QwQ-32B',
    'DeepSeek-R1-Distill-32B': 'DeepSeek-R1-Distill-32B',
    'DeepSeek-R1-Distill-70B': 'DeepSeek-R1-Distill-70B',
    'gemini-2.0-flash-thinking': 'gemini-2.0-flash-thinking',
    'Claude-3.7-Sonnet (Thinking)*': 'Claude-3.7-Sonnet (Thinking)',
    'DeepSeek-R1-Distill-14B': 'DeepSeek-R1-Distill-14B',
    'DeepSeek-V3-03-24*': 'DeepSeek-V3-03-24',
    'o3-mini (low)': 'o3-mini (low)',
    'QwQ-32B-Preview': 'QwQ-32B-Preview',
    'gemini-2.0-pro': 'gemini-2.0-pro',
    'gemini-2.0-flash': 'gemini-2.0-flash',
    'DeepSeek-V3': 'DeepSeek-V3',
    'DeepSeek-R1-Distill-1.5B': 'DeepSeek-R1-Distill-1.5B',
    'gpt-4o': 'gpt-4o',
    'Claude-3.5-Sonnet': 'Claude-3.5-Sonnet'
}

matharena_nice_llm_names_subset = {
    'o3-mini (high)': 'o3-mini (high)',
    # 'o3-mini (medium)': 'o3-mini (medium)',
    # 'o1 (medium)': 'o1 (medium)',
    'DeepSeek-R1': 'DeepSeek-R1',
    'QwQ-32B*': 'QwQ-32B*',
    # 'DeepSeek-R1-Distill-32B': 'DeepSeek-R1-Distill-32B',
    # 'DeepSeek-R1-Distill-70B': 'DeepSeek-R1-Distill-70B',
    # 'gemini-2.0-flash-thinking': 'gemini-2.0-flash-thinking',
    # 'Claude-3.7-Sonnet (Thinking)*': 'Claude-3.7-Sonnet (Thinking)*',
    # 'DeepSeek-R1-Distill-14B': 'DeepSeek-R1-Distill-14B',
    # 'DeepSeek-V3-03-24*': 'DeepSeek-V3-03-24*',
    'o3-mini (low)': 'o3-mini (low)',
    # 'QwQ-32B-Preview': 'QwQ-32B-Preview',
    'gemini-2.0-pro': 'gemini-2.0-pro',
    # 'gemini-2.0-flash': 'gemini-2.0-flash',
    # 'DeepSeek-V3': 'DeepSeek-V3',
    # 'DeepSeek-R1-Distill-1.5B': 'DeepSeek-R1-Distill-1.5B',
    'gpt-4o': 'gpt-4o',
    'Claude-3.5-Sonnet': 'Claude-3.5-Sonnet'
}

def create_error_bar_plot(plot_name, nice_llm_names, data_location, fig_size=(6.75, 3), eval_setting='basic', N_t=None):
    if eval_setting == 'subtask':
        method_names_ordered = subtask_method_names_ordered
    elif eval_setting == 'subtask_ONLY':
        method_names_ordered = subtask_ONLY_method_names_ordered
    else:
        method_names_ordered = basic_method_names_ordered

    # Load data
    data = pd.read_csv(data_location)
    if "matharena" in plot_name and eval_setting == 'basic':
        # only use first attempt at each problem
        data = data.iloc[::4]
    
    # Get LLMs to plot
    llms = list(nice_llm_names.keys())
    
    # Calculate empirical means
    empirical_means = {llm: data[llm].mean() for llm in llms}
    
    # Calculate error bars
    error_bars_0_05 = {}
    error_bars_0_5 = {}
    method_means = {}
    for llm in llms:
        error_bars_0_05[llm] = {}
        error_bars_0_5[llm] = {}
        method_means[llm] = {}

        eval_data = data[llm].to_numpy()
        if eval_setting in ['subtask', 'subtask_ONLY']:
            assert N_t is not None
            # split data into N/N_t tasks
            eval_data = [eval_data[..., (i*N_t):((i+1)*N_t)] for i in range(len(eval_data)//N_t)]

        error_bars = compute_error_bars(eval_data, [0.05, 0.5, 1], eval_setting=eval_setting if eval_setting != 'subtask_ONLY' else 'subtask')

        for method_name, error in error_bars.items():
            error_bars_0_05[llm][method_name] = error[0]
            error_bars_0_5[llm][method_name] = error[1]

            if 'bayes' in method_name or 'boot' in method_name:
                method_means[llm][method_name] = error[2].mean()
            elif method_name == 'wils':
                method_means[llm][method_name] = error[0].mean()
            else:
                method_means[llm][method_name] = empirical_means[llm]
            
        method_names = list(error_bars.keys())
    
    # Filter method names to match the ordered list
    method_names = [method for method in method_names_ordered if method in method_names]

    # Construct a dataframe with an llm_name column, an empirical_mean column, and three columns for each method (lower, mean, upper)
    df = pd.DataFrame(index=llms)
    df['empirical_mean'] = empirical_means
    per_llm_method_means = {method: {llm: method_means[llm][method] for llm in llms} for method in method_names}
    per_llm_error_bars_lower = {method: {llm: error_bars_0_05[llm][method][0] for llm in llms} for method in method_names}
    per_llm_error_bars_upper = {method: {llm: error_bars_0_05[llm][method][1] for llm in llms} for method in method_names}

    for method in method_names:
        df[f'{method}_lower'] = per_llm_error_bars_lower[method]
        # df[f'{method}_mean'] = per_llm_method_means[method]
        df[f'{method}_upper'] = per_llm_error_bars_upper[method]

    # Turn the llm index into a column
    df = df.reset_index()
    df = df.rename(columns={'index': 'llm'})

    # Save the dataframe
    Path("results/real_world_error_bars").mkdir(parents=True, exist_ok=True)
    df.to_csv(f'results/real_world_error_bars/{plot_name}_df.csv', index=False)

    # Create figure and axes
    fig, axs = plt.subplots(1, 1, figsize=fig_size)
    axs.grid(True, alpha=0.3)
    # Plot faint horizontal lines at accuracy = 0 and = 1
    axs.axhline(0, color='black', alpha=0.3, linestyle="-")
    axs.axhline(1, color='black', alpha=0.3, linestyle="-")

    # Plot the error bars
    box_width = 0.2 if eval_setting in ['basic', 'subtask_ONLY'] else 0.25
    box_gap = 0.15 if eval_setting in ['basic', 'subtask_ONLY'] else 0.25
    width_per_llm = 2 if eval_setting in ['basic', 'subtask_ONLY'] else 4.5
    for method_idx, method in enumerate(method_names):
        if (box_width+box_gap)*len(method_names) > width_per_llm:
            print("Warning: the width of the box plots are too large for the number of LLMs. Consider reducing the boxplot widths/gaps or increasing the width_per_llm.")
        
        if eval_setting == 'subtask':
            positions = np.arange(len(llms)) * width_per_llm + (method_idx - (len(method_names) % 2) - 1)* (box_width+box_gap)  # Increase the distance between each LLM
        else:
            positions = np.arange(len(llms)) * width_per_llm + method_idx * (box_width+box_gap)  # Increase the distance between each LLM

        for llm_idx, llm in enumerate(llms): 
            colour = COLOURS[method]
            line_style = "solid" #if not (eval_setting == 'subtask' and method not in ['IS_subtask', 'clt_subtask', 'boot_subtask']) else "dashed"
            alpha = 1 #if not (eval_setting == 'subtask' and method not in ['IS_subtask', 'clt_subtask', 'boot_subtask']) else 0.5
            axs.bxp([{
                        # 'med': error_bars_0_5[llm][method_idx].mean(),
                        'med': method_means[llm][method],
                        'q1': method_means[llm][method],#error_bars_0_5[llm][method][0],
                        'q3': method_means[llm][method],#error_bars_0_5[llm][method][1],
                        'whislo': error_bars_0_05[llm][method][0],
                        'whishi': error_bars_0_05[llm][method][1],
                        'caps': error_bars_0_05[llm][method],
                        'fliers': [],
                        'mean': []
                    }], 
                    positions=[positions[llm_idx]], 
                    widths=box_width, 
                    showfliers=False, 
                    boxprops=dict(color=colour, linestyle=line_style, alpha=alpha), 
                    medianprops=dict(color=colour, linewidth=0, alpha=alpha), 
                    whiskerprops=dict(color=colour, linestyle=line_style, zorder=50, alpha=alpha), 
                    capprops=dict(color=colour, alpha=alpha))

    # Plot empirical means
    for llm_idx, llm in enumerate(llms):
        llm_boxes_total_width = len(method_names) * (box_width+box_gap) - box_gap
        positions = np.arange(len(llms)) * width_per_llm + (box_width+box_gap)*2

        empirical_means_xs = [positions[llm_idx] - llm_boxes_total_width/2, positions[llm_idx] + llm_boxes_total_width/2]
        if eval_setting == 'subtask':
            empirical_means_xs[1] += (1-len(method_names)%2)*(box_width+box_gap)/2
        empirical_means_ys = [empirical_means[llm], empirical_means[llm]]
        
        axs.plot(empirical_means_xs, empirical_means_ys, linestyle=empirical_means_line_style, **empirical_mean_style_kwargs)
            
    # Create legend
    handles = [plt.Line2D([0], [0],
                          color=COLOURS[method],
                          alpha=1,# if not (eval_setting == 'subtask' and method not in ['IS_subtask', 'clt_subtask', 'boot_subtask']) else 0.5,
                          lw=2)
               for method in method_names]

    handles.append(plt.Line2D([0], [0], linestyle=empirical_means_line_style, **empirical_mean_style_kwargs))
    labels = [METHOD_NAMES[method] for method in method_names]
    labels.append('Empirical Mean')
    axs.set_xticks(np.arange(len(llms)) * width_per_llm + (box_width+box_gap)*2)
    axs.set_xticklabels([nice_llm_names[llm] for llm in llms], rotation=30, ha='right')
    axs.set_ylabel('Accuracy')

    fig.tight_layout()

    # Shrink current axis's height by 10% on the top
    box = axs.get_position()
    axs.set_position([box.x0, box.y0,
                    box.width, box.height * 0.85])
    
    fig.legend(handles, labels,
                loc='upper center',
                fancybox=True, shadow=True, ncol=3, bbox_to_anchor=(0.5, 1.))

    # Save figures
    plt.savefig(f"plots/pngs/real_data_{plot_name}.png", dpi=300)
    plt.savefig(f"plots/pdfs/real_data_{plot_name}.pdf")
    plt.savefig(f"PLOTS_FINAL/pngs/real_data_{plot_name}.png", dpi=300)
    plt.savefig(f"PLOTS_FINAL/pdfs/real_data_{plot_name}.pdf")

    print(f"Saved plots/pngs/real_data_{plot_name}.png and plots/pdfs/real_data_{plot_name}.pdf")
    print(f"Saved PLOTS_FINAL/pngs/real_data_{plot_name}.png and PLOTS_FINAL/pdfs/real_data_{plot_name}.pdf")


if __name__ == "__main__":
    np.random.seed(0)

    plot_names_to_llm_collections_and_data_locations = {
        'langchain_full': (langchain_nice_llm_names_all, 'data/langchain_data_tool_use_FULL.csv'),
        'langchain_subset': (langchain_nice_llm_names_subset, 'data/langchain_data_tool_use_FULL.csv'),
        'matharena_aime_II_full': (matharena_nice_llm_names_all, 'data/matharena_aime_II.csv'),
        'matharena_aime_II_subset': (matharena_nice_llm_names_subset, 'data/matharena_aime_II.csv'),
    }
        
    for plot_name, (nice_llm_names, data_location) in plot_names_to_llm_collections_and_data_locations.items():
        create_error_bar_plot(plot_name, nice_llm_names, data_location, fig_size=(12, 5) if plot_name == 'matharena_aime_II' else ((6.75, 3) if 'full' in plot_name else (3.25, 3)))
        print()

    # also do subtask plots for matharena AIME II
    create_error_bar_plot('matharena_aime_II_full_subtask', matharena_nice_llm_names_all, 'data/matharena_aime_II.csv', fig_size=(6.75, 3), eval_setting='subtask', N_t=4)
    print()

    create_error_bar_plot('matharena_aime_II_full_subtask_ONLY', matharena_nice_llm_names_all, 'data/matharena_aime_II.csv', fig_size=(6.75, 3), eval_setting='subtask_ONLY', N_t=4)