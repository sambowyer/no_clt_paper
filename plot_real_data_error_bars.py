import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

method_names_ordered = ["bayes", "freq", "wils", "clop", "boot"]

nice_llm_names_all = {
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

nice_llm_names_subset = {
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


if __name__ == "__main__":
    np.random.seed(0)

    plot_names_to_llm_collections = {
        'full': nice_llm_names_all,
        'subset': nice_llm_names_subset
    }
    
    langchain_data = pd.read_csv('data/langchain_data_tool_use_FULL.csv')

    for plot_name, nice_llm_names in plot_names_to_llm_collections.items():

        langchain_llms = langchain_data.columns[2:]

        # assert all([llm in langchain_llms for llm in nice_llm_names.keys()])
        langchain_llms = (nice_llm_names.keys())

        # first the empirical means
        empirical_means = {llm: langchain_data[llm].mean() for llm in langchain_llms}

        # Then the error bars
        error_bars_0_05 = {}
        error_bars_0_5 = {}
        method_means = {}
        for llm in langchain_llms:
            error_bars_0_05[llm] = {}
            error_bars_0_5[llm] = {}
            method_means[llm] = {}

            error_bars = compute_error_bars(langchain_data[llm].to_numpy(), [0.05, 0.5, 1])

            for method_name, error in error_bars.items():
                error_bars_0_05[llm][method_name] = error[0]
                error_bars_0_5[llm][method_name] = error[1]

                if method_name in ('bayes', 'boot'):
                    method_means[llm][method_name] = error[2].mean()
                elif method_name == 'wils':
                    method_means[llm][method_name] = error[0].mean()
                else:
                    method_means[llm][method_name] = empirical_means[llm]
                

            method_names = list(error_bars.keys())

        # print(method_means)

        method_names = [method for method in method_names_ordered if method in method_names]
        
        fig, axs = plt.subplots(1, 1, figsize=(6.75, 3) if plot_name == 'full' else (3.25, 3))
        axs.grid(True, alpha=0.3)
        # first plot faint horizontal lines at accuracy = 0 and = 1
        axs.axhline(0, color='black', alpha=0.3, linestyle="-")
        axs.axhline(1, color='black', alpha=0.3, linestyle="-")

        # Now we can plot the error bars
        for method_idx, method in enumerate(method_names):
            box_width = 0.2
            box_gap = 0.15
            width_per_llm = 2
            if (box_width+box_gap)*len(method_names) > width_per_llm:
                print("Warning: the width of the box plots are too large for the number of LLMs. Consider reducing the boxplot widths/gaps or increasing the width_per_llm.")
            positions = np.arange(len(langchain_llms)) * width_per_llm + method_idx * (box_width+box_gap)  # Increase the distance between each LLM

            for llm_idx, llm in enumerate(langchain_llms): 
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
                        boxprops=dict(color='#444'), 
                        medianprops=dict(color=COLOURS[method], linewidth=0), 
                        whiskerprops=dict(color=COLOURS[method], zorder=50), 
                        capprops=dict(color=COLOURS[method]))

        for llm_idx, llm in enumerate(langchain_llms):
            llm_boxes_total_width = len(method_names) * (box_width+box_gap) - box_gap
            positions = np.arange(len(langchain_llms)) * width_per_llm + (box_width+box_gap)*2

            empirical_means_xs = [positions[llm_idx] - llm_boxes_total_width/2, positions[llm_idx] + llm_boxes_total_width/2]
            empirical_means_ys = [empirical_means[llm], empirical_means[llm]]
            
            axs.plot(empirical_means_xs, empirical_means_ys, linestyle=empirical_means_line_style, **empirical_mean_style_kwargs)
                
        handles = [plt.Line2D([0], [0], color=COLOURS[method], lw=2) for method in method_names]
        handles.append(plt.Line2D([0], [0], linestyle=empirical_means_line_style, **empirical_mean_style_kwargs))
        labels = [METHOD_NAMES[method] for method in method_names]
        labels.append('Empirical Mean')
        axs.set_xticks(np.arange(len(langchain_llms)) * width_per_llm + (box_width+box_gap)*2)
        axs.set_xticklabels([nice_llm_names[llm] for llm in langchain_llms], rotation=30, ha='right')
        axs.set_ylabel('Accuracy')

        fig.tight_layout()

        # Shrink current axis's height by 10% on the top
        box = axs.get_position()
        axs.set_position([box.x0, box.y0,
                        box.width, box.height * 0.85])

        # fig.tight_layout()
        
        fig.legend(handles, labels,
                    loc='upper center',
                    fancybox=True, shadow=True, ncol=3, bbox_to_anchor=(0.5, 1.))
        # fig.tight_layout()

        plt.savefig(f"plots/pngs/real_data_langchain_{plot_name}.png")
        plt.savefig(f"plots/pdfs/real_data_langchain_{plot_name}.pdf")


        plt.savefig(f"PLOTS_FINAL/pngs/real_data_langchain_{plot_name}.png")
        plt.savefig(f"PLOTS_FINAL/pdfs/real_data_langchain_{plot_name}.pdf")