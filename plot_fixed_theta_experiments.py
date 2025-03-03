import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True
plt.style.use(['default'])

# Set font sizes
plt.rc('font', size=8)          # controls default text sizes
plt.rc('axes', titlesize=10)    # fontsize of the axes title
plt.rc('axes', labelsize=9)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=7)    # fontsize of the tick labels
plt.rc('ytick', labelsize=7)    # fontsize of the tick labels
plt.rc('legend', fontsize=7)    # legend fontsize
plt.rc('figure', titlesize=9)   # fontsize of the figure title

from eval_lib import COLOURS, METHOD_NAMES

def load_results_dict(filename):
    with open(f"results/{filename}.pkl", "rb") as f:
        results_dict = pickle.load(f)

    # rename 'mill_subtask' and 'mill_paired' to 'clt_subtask' and 'clt_paired'
    # in results_dict['widths'] and results_dict['coverages']
    for metric in ['widths', 'coverages']:
        for method in ['mill_subtask', 'mill_paired']:
            if method in results_dict[metric]:
                results_dict[metric][method.replace('mill', 'clt')] = results_dict[metric].pop(method)
                
    return results_dict

def fixed_theta_plot(results_collection, plot_filename=None, folder='plots', plot_type='subtask'):
    '''
    Parameters
    ----------
    results_collection : dict
        A dictionary of results dictionaries, where the keys are the y-axis labels and the values are the results dictionaries.
    '''
    n_vals_collection = [results['n_vals'] for results in results_collection.values()]
    assert all(n_vals == n_vals_collection[0] for n_vals in n_vals_collection), "All results dictionaries must have same n_vals"
    n_vals = n_vals_collection[0]

    fig, axs = plt.subplots(len(results_collection), len(n_vals), figsize=(6.4, 4.8), sharex=True, sharey=True)

    for j, (y_label, results) in enumerate(results_collection.items()):
        coverages = results['coverages']
        widths = results['widths']
        alphas = results['alphas']
        n_vals = results['n_vals']

        all_methods = ['IS_subtask', 'bayes', 'clt_subtask', 'freq', 'wils', 'clop', 'boot_subtask', 'boot']
        unclustered_methods = ['bayes', 'freq', 'wils', 'clop', 'boot']
        paired_methods = ['bayes_paired', 'bayes_unpaired', 'clt_paired', 'freq', 'boot']

        dashed_methods = unclustered_methods + ['bayes_unpaired']

        if plot_type == 'subtask':
            n_per_task = results['n_per_task'] if 'n_per_task' in results else results['N_t']
            Ts = results['Ts']

            methods = [m for m in all_methods if m in coverages]
        elif plot_type == 'paired':
            methods = [m for m in paired_methods if m in coverages]
        else:
            methods = [m for m in unclustered_methods if m in coverages]
        
        # plot coverage vs alpha
        for i, ax in enumerate(axs[j,:]):
            for method in methods:
                ax.plot(1-alphas,
                        coverages[method][i],
                        label=METHOD_NAMES[method],
                        color=COLOURS[method], 
                        linestyle=':' if method in dashed_methods and plot_type in ('subtask', 'paired') else '-',
                        alpha=0.5 if method in dashed_methods and plot_type in ('subtask', 'paired') else 1
                )

            # plot y=1-alpha
            ax.plot(1-alphas, 1-alphas, color='grey', linestyle='--', label=r"$1-\alpha$")

            ax.grid(True, alpha=0.3)

        axs[j,0].set_ylabel('Coverage')

        axs[j,-1].yaxis.set_label_position("right")
        axs[j,-1].set_ylabel(f"{y_label}", rotation=270, labelpad=15)        
    
    for i, ax in enumerate(axs[0,:]):
        if plot_type == 'subtask':
            ax.set_title(f'$N$ = {n_vals[i]}\n($T$={Ts[i]}, $N_t$={n_per_task})')
        else:
            ax.set_title(f'$N$ = {n_vals[i]}')

    # Set a single ylabel for all rows
    # fig.text(0.04, 0.5, 'Coverage', va='center', rotation='vertical')
    # axs[1,0].set_ylabel('Coverage')

    plt.tight_layout()

    # Shrink current axis's height by 10% on the bottom
    for i, ax in enumerate(axs.flatten()):
        box = ax.get_position()
        y_offset = box.height * 0.1 * (1,2.25,3.5)[i//len(n_vals)]
        ax.set_position([box.x0, box.y0 + y_offset,
                         box.width, box.height * 0.9])
        
    # Shift the x-axis labels to be horizontally aligned with the center of the overall plot
    axs[-1,1].set_xlabel(r"Confidence level, $1-\alpha$", va='top', ha='left', labelpad=0.5)

    # First find the center of the plot
    fig_center = axs[0,0].get_position().x0 + (axs[0,-1].get_position().x0 + axs[0,-1].get_position().width - axs[0,0].get_position().x0) / 2
    
    # Then (manually) shift the x-axis labels to be horizontally aligned with the center of the plot
    axs[-1,1].xaxis.set_label_coords(fig_center - .0, -0.2)

    # Put a legend below current axis
    if plot_type == 'subtask':
        fig.legend([METHOD_NAMES[m] for m in methods] + [r"$1-\alpha$"],
            loc='lower center',# bbox_to_anchor=(0.5, -0.05),
            fancybox=True, shadow=True, ncol=5)
    else:
        fig.legend([METHOD_NAMES[m] for m in methods] + [r"$1-\alpha$"],
            loc='lower center',
            fancybox=True, shadow=True, ncol=6, bbox_to_anchor=(0.5, 0.025))

    # save the plot
    if plot_filename is not None:
        plt.savefig(f"{folder}/pngs/{plot_filename}.png")
        plt.savefig(f"{folder}/pdfs/{plot_filename}.pdf")

        print(f"Saved plot to {folder}/pngs/{plot_filename}.png and {folder}/pdfs/{plot_filename}.pdf")
    else:
        plt.show()


if __name__ == "__main__":
    # load some results that we know exist
    import pickle

    ### FULL EXPERIMENTS

    basic_exps = {
        'exp4-1_fixed_theta': {r'$\theta = 0.5$':  load_results_dict("basic_small_FIXED_50"),#_BIGREPEATS"),
                                r'$\theta = 0.8$':  load_results_dict("basic_small_FIXED_80"),#_BIGREPEATS"),
                                r'$\theta = 0.95$': load_results_dict("basic_small_FIXED_95"),#_BIGREPEATS"),}
        }
    }

    subtask_exps = {
        'exp4-2_fixed_theta': {r'$\theta = 0.5$':  load_results_dict("subtask_small_FIXED_50"),#_BIGREPEATS"),
                                r'$\theta = 0.8$':  load_results_dict("subtask_small_FIXED_80"),#_BIGREPEATS"),
                                r'$\theta = 0.95$': load_results_dict("subtask_small_FIXED_95"),#_BIGREPEATS"),}
        }
    }

    paired_exps = {
        'exp4-4_fixed_thetaA50': {r'$\theta_B = 0.5$':  load_results_dict("paired_small_FIXED_A50_B50"),#_BIGREPEATS"),
                                   r'$\theta_B = 0.47$': load_results_dict("paired_small_FIXED_A50_B47"),#_BIGREPEATS"),
                                   r'$\theta_B = 0.42$': load_results_dict("paired_small_FIXED_A50_B42"),#_BIGREPEATS"),},
        },

        'exp4-4_fixed_thetaA80': {r'$\theta_B = 0.8$':  load_results_dict("paired_small_FIXED_A80_B80"),#_BIGREPEATS"),
                                   r'$\theta_B = 0.77$': load_results_dict("paired_small_FIXED_A80_B77"),#_BIGREPEATS"),
                                   r'$\theta_B = 0.72$': load_results_dict("paired_small_FIXED_A80_B72"),#_BIGREPEATS"),},
        },

        'exp4-4_fixed_thetaA95': {r'$\theta_B = 0.95$': load_results_dict("paired_small_FIXED_A95_B95"),#_BIGREPEATS"),
                                   r'$\theta_B = 0.92$': load_results_dict("paired_small_FIXED_A95_B92"),#_BIGREPEATS"),
                                   r'$\theta_B = 0.87$': load_results_dict("paired_small_FIXED_A95_B87"),#_BIGREPEATS"),}
        }
    }

    exps = {'basic': basic_exps, 'subtask': subtask_exps, 'paired': paired_exps}
    
    for exp_type, exp_dicts in exps.items():
        for exp_filename, results_collection in exp_dicts.items():

            fixed_theta_plot(results_collection, plot_filename=exp_filename, plot_type=exp_type, folder='PLOTS_FINAL')
