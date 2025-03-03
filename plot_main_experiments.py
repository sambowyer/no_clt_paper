import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams['text.usetex'] = True
plt.style.use(['default'])

# Set font sizes
def set_font_sizes(small=False):
    if not small:
        plt.rc('font', size=8)          # controls default text sizes
        plt.rc('axes', titlesize=9)    # fontsize of the axes title
        plt.rc('axes', labelsize=9)     # fontsize of the x and y labels
        plt.rc('xtick', labelsize=7)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=7)    # fontsize of the tick labels
        plt.rc('legend', fontsize=7)    # legend fontsize
        plt.rc('figure', titlesize=9)   # fontsize of the figure title
    else:
        plt.rc('font', size=8)          # controls default text sizes
        plt.rc('axes', titlesize=9)    # fontsize of the axes title
        plt.rc('axes', labelsize=7)     # fontsize of the x and y labels
        plt.rc('xtick', labelsize=7)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=7)    # fontsize of the tick labels
        plt.rc('legend', fontsize=7)    # legend fontsize
        plt.rc('figure', titlesize=9)   # fontsize of the figure title


from eval_lib import COLOURS, METHOD_NAMES, get_mean_abs_distance_from_coverage

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
def cov_vs_alpha_plot(results_dict, plot_filename=None, folder='plots', plot_type='subtask', restrict_bayes_methods=False):
    set_font_sizes()

    # get the data to plot
    coverages = results_dict['coverages']
    widths = results_dict['widths']
    alphas = results_dict['alphas']
    n_vals = results_dict['n_vals']

    nomimal_err = get_mean_abs_distance_from_coverage(coverages, n_vals, alphas)

    # figure out which methods we're plotting
    all_methods = ['IS_subtask', 'bayes', 'clt_subtask', 'freq', 'wils', 'clop', 'boot_subtask', 'boot']
    unclustered_methods = ['bayes', 'freq', 'wils', 'clop', 'boot']

    paired_methods = ['bayes_paired', 'bayes_unpaired', 'clt_paired', 'freq', 'boot']
    if not restrict_bayes_methods:
        paired_methods += ['bayes_paired_dirichlet', 'bayes_paired_per_question']

    if plot_type == 'subtask':
        n_per_task = results_dict['n_per_task'] if 'n_per_task' in results_dict else results_dict['N_t']
        Ts = results_dict['Ts']

        methods = [m for m in all_methods if m in coverages]
    elif plot_type == 'paired':
        methods = [m for m in paired_methods if m in coverages]
    else:
        methods = [m for m in unclustered_methods if m in coverages]

    dashed_methods = unclustered_methods + ['bayes_unpaired']
    
    # set up the figure
    # figsize = (6.75, 2.25) if plot_type == 'paired' else (6.75, 2.5)
    figsize = (6.75, 1.75) if plot_type == 'paired' else (6.75, 1.9)
    fig = plt.figure(constrained_layout=True, figsize=figsize)
    # fig, axs = plt.subplots(1, len(n_vals)+1, figsize=figsize)#, sharey=True)
    axs = []

    gs1 = gridspec.GridSpec(1, len(n_vals)+1, width_ratios=[1]*len(n_vals) + [1.2], figure=fig)
    gs1.update(wspace=0.2, hspace=1.5, left=0.08, right=0.96) # set the spacing between axes
    
    # plot coverage vs alpha
    # for i, ax in enumerate(axs[:len(n_vals)]):
    for i in range(len(n_vals)):
        ax = fig.add_subplot(gs1[i])
        axs.append(ax)
        if i > 0:
            ax.sharey(axs[0])
            ax.yaxis.set_tick_params(which='both', labelleft=False, labelright=False)

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

        if plot_type == 'subtask':
            ax.set_title(f'$N$ = {n_vals[i]} ($T$={Ts[i]})')#, $N_t$={n_per_task})')
        else:
            ax.set_title(f'$N$ = {n_vals[i]}')

        ax.grid(True, alpha=0.3)

    axs[1].set_xlabel(r"Confidence level, $1-\alpha$", va='top', ha='left', labelpad=0.5)
    axs[0].set_ylabel('Coverage')

    # plot coverage error vs N
    gs2 = gridspec.GridSpec(1, 1)#len(n_vals)+1)#, figure=fig)
    gs2.update(wspace=0.2, hspace=1.5, left=0.84, right=0.98) # set the spacing between axes
    axs.append(fig.add_subplot(gs2[-1]))
    n_vals_str = [str(n) for n in n_vals]
    for method in methods:
        axs[-1].plot(n_vals_str, nomimal_err[method], 
                     label=METHOD_NAMES[method],
                     color=COLOURS[method], 
                     linestyle=':' if method in dashed_methods and plot_type in ('subtask', 'paired') else '-',
                     alpha=0.5 if method in dashed_methods and plot_type in ('subtask', 'paired') else 1
        )

    axs[-1].plot(n_vals_str, [0]*len(n_vals), color='grey', linestyle='--', label="y=0")
    axs[-1].set_ylabel('Coverage Error')

    axs[-1].set_xlabel(r'$N$')
    axs[-1].set_xticks(n_vals_str)

    axs[-1].grid(True, alpha=0.3)

    # plt.tight_layout()

    # Shrink current axis's height by 10% on the bottom
    # shink_factor = 0.18 if plot_type == 'paired' else 0.2
    shink_factor = 0.35 if plot_type == 'paired' else 0.4
    for i, ax in enumerate(axs):
        box = ax.get_position()
        y_offset = box.height * shink_factor
        ax.set_position([box.x0, box.y0 + y_offset,
                         box.width, box.height * (1-shink_factor)])
        
    # Shift the x-axis labels to be horizontally aligned with the center of the overall plot
    # First find the center of the plot
    fig_center = axs[0].get_position().x0 + (axs[-2].get_position().x0 + axs[-2].get_position().width - axs[0].get_position().x0) / 2
    
    # Then (manually) shift the x-axis labels to be horizontally aligned with the center of the plot
    axs[1].xaxis.set_label_coords(fig_center + 0.05, -0.22)

    end_box = axs[-1].get_position()
    axs[-1].xaxis.set_label_coords(end_box.x0 - 3*end_box.width, -0.22)

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

def double_plot(results_dict, plot_filename=None, folder='plots', plot_type='subtask', restrict_bayes_methods=False):
    set_font_sizes()#small = (plot_type == 'basic'))

    # get the data to plot
    coverages = results_dict['coverages']
    widths = results_dict['widths']
    alphas = results_dict['alphas']
    n_vals = results_dict['n_vals']

    nomimal_err = get_mean_abs_distance_from_coverage(coverages, n_vals, alphas)

    # figure out which methods we're plotting
    all_methods = ['IS_subtask', 'bayes', 'clt_subtask', 'freq', 'wils', 'clop', 'boot_subtask', 'boot']
    unclustered_methods = ['bayes', 'freq', 'wils', 'clop', 'boot']

    paired_methods = ['bayes_paired', 'bayes_unpaired', 'clt_paired', 'freq', 'boot']
    if not restrict_bayes_methods:
        paired_methods += ['bayes_paired_dirichlet', 'bayes_paired_per_question']

    if plot_type == 'subtask':
        n_per_task = results_dict['n_per_task'] if 'n_per_task' in results_dict else results_dict['N_t']
        Ts = results_dict['Ts']

        methods = [m for m in all_methods if m in coverages]
    elif plot_type == 'paired':
        methods = [m for m in paired_methods if m in coverages]
    else:
        methods = [m for m in unclustered_methods if m in coverages]

    dashed_methods = unclustered_methods + ['bayes_unpaired']
    
    # set up the figure
    figsize = (6, 3.05)
    fig = plt.figure(constrained_layout=True, figsize=figsize)
    gs = gridspec.GridSpec(3, len(n_vals) + 1, figure=fig, width_ratios=[1]*len(n_vals) + [1.1], height_ratios=[1, 1, 0.5 if plot_type == 'subtask' else 0.4])
    gs.update(wspace=0.1, hspace=0.0) # set the spacing between axes
    
    axs = []
    for i in range(2):
        row_axs = []
        for j in range(len(n_vals)):
            ax = fig.add_subplot(gs[i, j])
            row_axs.append(ax)
            if j > 0:
                # make the y-axes share the same scale (except for the nominal error plot)
                ax.sharey(row_axs[0])
                ax.yaxis.set_tick_params(which='both', labelleft=False, labelright=False)
        axs.append(row_axs)
    
    # add the extra plot for nominal error
    ax_nominal_err = fig.add_subplot(gs[0, -1])
    
    # plot coverage vs alpha
    for i, ax in enumerate(axs[0]):
        for method_idx, method in enumerate(methods):
            ax.plot(1-alphas,
                    coverages[method][i],
                    label=METHOD_NAMES[method],
                    color=COLOURS[method], 
                    linestyle=':' if method in dashed_methods and plot_type in ('subtask', 'paired') else '-',
                    alpha=0.5 if method in dashed_methods and plot_type in ('subtask', 'paired') else 1,
                    zorder=len(methods) - method_idx
            )

        # plot y=1-alpha
        ax.plot(1-alphas, 1-alphas, color='grey', linestyle='--', label=r"$1-\alpha$", zorder=-1)

        if plot_type == 'subtask':
            ax.set_title(f'$N$ = {n_vals[i]}\n($T$={Ts[i]}, $N_t$={n_per_task})')
            # ax.set_title(f'$(T, N_t) =$ ({Ts[i]},{n_per_task})')#, $N_t$={n_per_task})')
        else:
            ax.set_title(f'$N$ = {n_vals[i]}')

        ax.grid(True, alpha=0.3)
        ax.set_xticks([0.8, 0.9, 1.0])

    axs[0][1].set_xlabel(r"Confidence level, $1-\alpha$", va='top', labelpad=0.5)
    axs[0][1].xaxis.set_label_coords(1.15, -0.25)
    axs[0][0].set_ylabel('Coverage')
    # axs[0,-1].legend(loc='lower right', fontsize=7, ncol=2)

    # plot coverage vs width
    for i, ax in enumerate(axs[1]):
        for method in methods:
            ax.plot(widths[method][i], 
                    coverages[method][i], 
                    label=METHOD_NAMES[method], 
                    color=COLOURS[method],
                    linestyle=':' if method in dashed_methods and plot_type in ('subtask', 'paired') else '-',
                    alpha=0.5 if method in dashed_methods and plot_type in ('subtask', 'paired') else 1
            )

        ax.grid(True, alpha=0.3)

    # fig.supxlabel('Interval Width', y=0.1)
    axs[1][1].set_xlabel('Interval Width', va='top', ha='left', labelpad=0.5) # don't use ha='right' (or = anything) as it will mess up subplot spacing
    axs[1][0].set_ylabel('Coverage')

    # Plot nominal error
    n_vals_str = [str(n) for n in n_vals]
    for method in methods:
        ax_nominal_err.plot(n_vals_str, nomimal_err[method], 
                            label=METHOD_NAMES[method],
                            color=COLOURS[method], 
                            linestyle=':' if method in dashed_methods and plot_type in ('subtask', 'paired') else '-',
                            alpha=0.5 if method in dashed_methods and plot_type in ('subtask', 'paired') else 1
        )

    ax_nominal_err.plot(n_vals_str, [0]*len(n_vals), color='grey', linestyle='--', label="y=0")
    ax_nominal_err.set_ylabel('Coverage Error')
    ax_nominal_err.set_xlabel(r'$N$')
    ax_nominal_err.set_xticks(n_vals_str)
    ax_nominal_err.grid(True, alpha=0.3)

    # plt.tight_layout()

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
    import pickle

    ### FULL EXPERIMENTS

    basic_exps = {
        ### IID questions
        'basic_small': 'exp4-1',

        # ## IID questions w/ prior mismatch
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

        ## Paired w/ ablated correlation coefficients
        'true_paired_small_neg_rho': 'exp4-4_neg_rho',
        'true_paired_small_zero_rho': 'exp4-4_zero_rho',
        'true_paired_small_pos_rho': 'exp4-4_pos_rho',

        ## Paired w/ high correlation coefficient and large theta_A and theta_B
        'true_paired_small_pos_rho_A10020_B10020': 'exp4-4_both_beta-100-20_large_rho',
        
        ## Paired w/ prior mismatch
        # Uniform theta_A
        'true_paired_small_beta_10_10': 'exp4-4_beta-10-10_mismatch',
        'true_paired_small_beta_100_20': 'exp4-4_beta-100-20_mismatch',
        'true_paired_small_beta_20_100': 'exp4-4_beta-20-100_mismatch',

        # # Non-uniform theta_A
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

            double_plot(results_dict, plot_filename=exp_filename, plot_type=exp_type, folder='PLOTS_FINAL', restrict_bayes_methods=True)

            if exp_name in make_cov_alpha_only_plot_exps:
                cov_vs_alpha_plot(results_dict, plot_filename=f"{exp_filename}_alpha_only", plot_type=exp_type, folder='PLOTS_FINAL', restrict_bayes_methods=True)
                