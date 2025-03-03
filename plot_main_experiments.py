import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True
plt.style.use(['default'])

# Set font sizes
def set_font_sizes(small=False):
    if not small:
        plt.rc('font', size=8)          # controls default text sizes
        plt.rc('axes', titlesize=10)    # fontsize of the axes title
        plt.rc('axes', labelsize=9)     # fontsize of the x and y labels
        plt.rc('xtick', labelsize=7)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=7)    # fontsize of the tick labels
        plt.rc('legend', fontsize=7)    # legend fontsize
        plt.rc('figure', titlesize=9)   # fontsize of the figure title
    else:
        plt.rc('font', size=8)          # controls default text sizes
        plt.rc('axes', titlesize=10)    # fontsize of the axes title
        plt.rc('axes', labelsize=7)     # fontsize of the x and y labels
        plt.rc('xtick', labelsize=7)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=7)    # fontsize of the tick labels
        plt.rc('legend', fontsize=7)    # legend fontsize
        plt.rc('figure', titlesize=9)   # fontsize of the figure title


from eval_lib import COLOURS, METHOD_NAMES

# plot coverage vs width
def cov_vs_width_plot(results_dict, plot_filename=None):
    set_font_sizes()

    coverages = results_dict['coverages']
    widths = results_dict['widths']
    alphas = results_dict['alphas']
    n_vals = results_dict['n_vals']
    n_per_task = results_dict['n_per_task'] if 'n_per_task' in results_dict else results_dict['N_t']
    Ts = results_dict['Ts']
    
    fig, axs = plt.subplots(1, len(n_vals), figsize=(6.5, 1.75), sharey=True)
    for i, ax in enumerate(axs):
        ax.plot(widths['bayes'][i], coverages['bayes'][i], label='Bayes', color=COLOURS['bayes'])
        ax.plot(widths['freq'][i],  coverages['freq'][i],  label='Freq (CLT)', color=COLOURS['freq'])
        ax.plot(widths['wils'][i],  coverages['wils'][i],  label='Wilson', color=COLOURS['wils'])
        ax.plot(widths['boot'][i],  coverages['boot'][i],  label='Bootstrap', color=COLOURS['boot'])

        ax.plot(widths['IS_subtask'][i], coverages['IS_subtask'][i], label='Bayes Clustered (IS)', color=COLOURS['IS_subtask'])
        ax.plot(widths['clt_subtask'][i], coverages['clt_subtask'][i], label='CLT Clustered', color=COLOURS['clt_subtask'])

        ax.plot(widths['clop'][i],  coverages['clop'][i],  label='Clopper-Pearson', color=COLOURS['clop'])

        # # plot y=width
        # ax.plot([0,1], [0,1], color='grey', linestyle='--', label="y=x")

        ax.set_title(f'N = {n_vals[i]}')
        ax.set_xlabel('Interval Width')

    axs[0].set_ylabel('Coverage')
    axs[-1].legend(loc='lower right', fontsize=7, ncol=2)

    # add a title
    fig.suptitle(f'Coverage vs Width for Bayesian and Frequentist Methods\n($N_t=${n_per_task}, $T=${Ts})')
    plt.tight_layout()

    if plot_filename is not None:
        plt.savefig(f"plots/{plot_filename}.png")
        plt.savefig(f"plots/{plot_filename}.pdf")
    else:
        plt.show()

# plot coverage vs alpha
def cov_vs_alpha_plot(results_dict, plot_filename=None, folder='plots', plot_type='subtask'):
    set_font_sizes()

    coverages = results_dict['coverages']
    widths = results_dict['widths']
    alphas = results_dict['alphas']
    n_vals = results_dict['n_vals']

    all_methods = ['IS_subtask', 'bayes', 'clt_subtask', 'freq', 'wils', 'clop', 'boot_subtask', 'boot']
    unclustered_methods = ['bayes', 'freq', 'wils', 'clop', 'boot']
    paired_methods = ['bayes_paired', 'clt_paired', 'freq', 'boot']

    if plot_type == 'subtask':
        n_per_task = results_dict['n_per_task'] if 'n_per_task' in results_dict else results_dict['N_t']
        Ts = results_dict['Ts']

        methods = [m for m in all_methods if m in coverages]
    elif plot_type == 'paired':
        methods = [m for m in paired_methods if m in coverages]
    else:
        methods = [m for m in unclustered_methods if m in coverages]
    
    figsize = (6.75, 2.25) if plot_type == 'paired' else (6.75, 2.5)
    fig, axs = plt.subplots(1, len(n_vals), figsize=figsize, sharey=True)
    
    # plot coverage vs alpha
    for i, ax in enumerate(axs):
        for method in methods:
            ax.plot(1-alphas,
                    coverages[method][i],
                    label=METHOD_NAMES[method],
                    color=COLOURS[method], 
                    linestyle=':' if method in unclustered_methods and plot_type in ('subtask', 'paired') else '-',
                    alpha=0.5 if method in unclustered_methods and plot_type in ('subtask', 'paired') else 1
            )

        # plot y=1-alpha
        ax.plot(1-alphas, 1-alphas, color='grey', linestyle='--', label=r"$1-\alpha$")

        if plot_type == 'subtask':
            ax.set_title(f'$N$ = {n_vals[i]}\n($T$={Ts[i]}, $N_t$={n_per_task})')
        else:
            ax.set_title(f'$N$ = {n_vals[i]}')

        ax.grid(True, alpha=0.3)

    axs[1].set_xlabel(r"Confidence level, $1-\alpha$", va='top', ha='left', labelpad=0.5)
    axs[0].set_ylabel('Coverage')

    plt.tight_layout()

    # Shrink current axis's height by 10% on the bottom
    shink_factor = 0.18 if plot_type == 'paired' else 0.2
    for i, ax in enumerate(axs.flatten()):
        box = ax.get_position()
        y_offset = box.height * shink_factor
        ax.set_position([box.x0, box.y0 + y_offset,
                         box.width, box.height * (1-shink_factor)])
        
    # Shift the x-axis labels to be horizontally aligned with the center of the overall plot
    # First find the center of the plot
    fig_center = axs[0].get_position().x0 + (axs[-1].get_position().x0 + axs[-1].get_position().width - axs[0].get_position().x0) / 2
    
    # Then (manually) shift the x-axis labels to be horizontally aligned with the center of the plot
    axs[1].xaxis.set_label_coords(fig_center + 0.05, -0.18)

    # Put a legend below current axis
    if plot_type == 'subtask':
        fig.legend([METHOD_NAMES[m] for m in methods] + [r"$1-\alpha$"],
            loc='lower center',# bbox_to_anchor=(0.5, -0.05),
            fancybox=True, shadow=True, ncol=5)
    else:
        fig.legend([METHOD_NAMES[m] for m in methods] + [r"$1-\alpha$"],
            loc='lower center',
            fancybox=True, shadow=True, ncol=6, bbox_to_anchor=(0.5, 0.025))

    # fig.tight_layout()

    # save the plot
    if plot_filename is not None:
        plt.savefig(f"{folder}/pngs/{plot_filename}.png")
        plt.savefig(f"{folder}/pdfs/{plot_filename}.pdf")

        print(f"Saved plot to {folder}/pngs/{plot_filename}.png and {folder}/pdfs/{plot_filename}.pdf")
    else:
        plt.show()

def double_plot(results_dict, plot_filename=None, folder='plots', plot_type='subtask'):
    set_font_sizes(small = (plot_type == 'basic'))
    coverages = results_dict['coverages']
    widths = results_dict['widths']
    alphas = results_dict['alphas']
    n_vals = results_dict['n_vals']

    all_methods = ['IS_subtask', 'bayes', 'clt_subtask', 'freq', 'wils', 'clop', 'boot_subtask', 'boot']
    unclustered_methods = ['bayes', 'freq', 'wils', 'clop', 'boot']
    paired_methods = ['bayes_paired', 'clt_paired', 'freq', 'boot']

    if plot_type == 'subtask':
        n_per_task = results_dict['n_per_task'] if 'n_per_task' in results_dict else results_dict['N_t']
        Ts = results_dict['Ts']

        methods = [m for m in all_methods if m in coverages]
    elif plot_type == 'paired':
        methods = [m for m in paired_methods if m in coverages]
    else:
        methods = [m for m in unclustered_methods if m in coverages]
    
    fig, axs = plt.subplots(2, len(n_vals), figsize=(6.75, 4) if plot_type != 'basic' else (6, 3.05), sharey=True)
    
    # plot coverage vs alpha
    for i, ax in enumerate(axs[0]):
        for method in methods:
            ax.plot(1-alphas,
                    coverages[method][i],
                    label=METHOD_NAMES[method],
                    color=COLOURS[method], 
                    linestyle=':' if method in unclustered_methods and plot_type in ('subtask', 'paired') else '-',
                    alpha=0.5 if method in unclustered_methods and plot_type in ('subtask', 'paired') else 1
            )

        # plot y=1-alpha
        ax.plot(1-alphas, 1-alphas, color='grey', linestyle='--', label=r"$1-\alpha$")

        if plot_type == 'subtask':
            ax.set_title(f'$N$ = {n_vals[i]}\n($T$={Ts[i]}, $N_t$={n_per_task})')
        else:
            ax.set_title(f'$N$ = {n_vals[i]}')

        ax.grid(True, alpha=0.3)

        if plot_type == "basic":
            # fix xticks to [0.8, 0.9, 1.0]
            ax.set_xticks([0.8, 0.9, 1.0])

    axs[0,1].set_xlabel(r"Confidence level, $1-\alpha$", va='top', ha='left', labelpad=0.5)
    axs[0,0].set_ylabel('Coverage')
    # axs[0,-1].legend(loc='lower right', fontsize=7, ncol=2)

    # plot coverage vs width
    for i, ax in enumerate(axs[1]):
        for method in methods:
            ax.plot(widths[method][i], 
                    coverages[method][i], 
                    label=METHOD_NAMES[method], 
                    color=COLOURS[method],
                    linestyle=':' if method in unclustered_methods and plot_type in ('subtask', 'paired') else '-',
                    alpha=0.9 if method in unclustered_methods and plot_type in ('subtask', 'paired') else 1
            )

        ax.grid(True, alpha=0.3)

    # fig.supxlabel('Interval Width', y=0.1)
    axs[1,1].set_xlabel('Interval Width', va='top', ha='left', labelpad=0.5)
    axs[1,0].set_ylabel('Coverage')

    plt.tight_layout()

    # Shrink current axis's height by 10% on the bottom
    # and extend on x-axis by 5% on the right
    for i, ax in enumerate(axs.flatten()):
        box = ax.get_position()
        y_shrink = 0.1 if plot_type != 'basic' else 0.05
        x_stretch = 0#0.05
        y_offset = box.height * y_shrink *2.25 if i >= len(n_vals) else box.height * y_shrink
        if plot_type == 'basic':
            y_offset *= 1.75
        x_offset = 0#box.width * x_stretch * (i%4)
        ax.set_position([box.x0 + x_offset, box.y0 + y_offset,
                        box.width * (1+x_stretch), box.height * (1-y_shrink)])
        
    # Shift the x-axis labels to be horizontally aligned with the center of the overall plot
    # First find the center of the plot
    fig_center = axs[0,0].get_position().x0 + (axs[0,-1].get_position().x0 + axs[0,-1].get_position().width - axs[0,0].get_position().x0) / 2
    
    # Then (manually) shift the x-axis labels to be horizontally aligned with the center of the plot
    axs[0,1].xaxis.set_label_coords(fig_center + 0.05, -0.18 if plot_type != 'basic' else -0.23)
    axs[1,1].xaxis.set_label_coords(fig_center + 0.27, -0.18 if plot_type != 'basic' else -0.23)


    # Put a legend below current axis
    if plot_type == 'subtask':
        fig.legend([METHOD_NAMES[m] for m in methods] + [r"$1-\alpha$"],
            loc='lower center',# bbox_to_anchor=(0.5, -0.05),
            fancybox=True, shadow=True, ncol=5)
    elif plot_type == 'basic':
        fig.legend([METHOD_NAMES[m] for m in methods] + [r"$1-\alpha$"],
            loc='lower center',
            fancybox=True, shadow=True, ncol=6, bbox_to_anchor=(0.5, 0.01))
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
        ### IID questions
        'basic_small': 'exp4-1',

        ## IID questions w/ prior mismatch
        'basic_small_beta_10_10': 'exp4-1_beta-10-10_mismatch',
        'basic_small_beta_100_20': 'exp4-1_beta-100-20_mismatch',
        'basic_small_beta_20_100': 'exp4-1_beta-20-100_mismatch',
    }

    subtask_exps = {
        ### Subtask/Clustered
        # 'small_5E_IS': 'exp4-2',
        'subtask_small': 'exp4-2',

        ## Subtask w/ prior mismatch
        'small_beta_10_10': 'exp4-2_beta-10-10_mismatch',
        'small_beta_100_20': 'exp4-2_beta-100-20_mismatch',
        'small_beta_20_100': 'exp4-2_beta-20-100_mismatch',
    }

    paired_exps = {
        ### Paired
        'true_paired_small': 'exp4-4',
        
        ## Paired w/ prior mismatch
        # Uniform theta_A
        'true_paired_small_beta_10_10': 'exp4-4_beta-10-10_mismatch',
        'true_paired_small_beta_100_20': 'exp4-4_beta-100-20_mismatch',
        'true_paired_small_beta_20_100': 'exp4-4_beta-20-100_mismatch',

        # Non-uniform theta_A
        'true_paired_A10020_B10020_small': 'exp4-4_both_beta-100-20_mismatch',
        'true_paired_A10020_B20100_small': 'exp4-4_A_beta-100-20_B_beta-20-100_mismatch',
    }

    ### MINIMAL EXPERIMENTS
    # basic_exps = ['basic_small']
    # subtask_exps = ['small_5E_IS']
    # paired_exps = ['true_paired_small']
    
    exps = {'basic': basic_exps, 'subtask': subtask_exps, 'paired': paired_exps}

    make_cov_alpha_only_plot_exps = ['subtask_small', 'true_paired_small']
    
    for exp_type, exp_names in exps.items():
        for exp_name, exp_filename in exp_names.items():
            with open(f"results/{exp_name}.pkl", "rb") as f:
                results_dict = pickle.load(f)

                # rename 'mill_subtask' and 'mill_paired' to 'clt_subtask' and 'clt_paired'
                # in results_dict['widths'] and results_dict['coverages']
                for metric in ['widths', 'coverages']:
                    for method in ['mill_subtask', 'mill_paired']:
                        if method in results_dict[metric]:
                            results_dict[metric][method.replace('mill', 'clt')] = results_dict[metric].pop(method)

            double_plot(results_dict, plot_filename=exp_filename, plot_type=exp_type, folder='PLOTS_FINAL')

            if exp_name in make_cov_alpha_only_plot_exps:
                cov_vs_alpha_plot(results_dict, plot_filename=f"{exp_filename}_alpha_only", plot_type=exp_type, folder='PLOTS_FINAL')
                