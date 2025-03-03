import numpy as np
from numpy.random import dirichlet
from scipy.stats import bootstrap
import arviz as az
from tqdm import tqdm
import pickle

import matplotlib.pyplot as plt


def calculate_f1(confusion_array):
    """Calculate F1 score from an array containing [TP, FP, FN, TN]"""
    # confusion_array is [4, n_sim, sth]
    tp, fp, fn, _ = confusion_array  # [n_sim, sth]
    precision = np.divide(
        tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) != 0
    )
    recall = np.divide(
        tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) != 0
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision, dtype=float),
        where=(precision + recall) != 0,
    )
    return f1


def _stat_f1(experiment_arr, axis):
    """Statistic function for bootstrap"""
    # experiment_arr is [4, BOOT_SIM, n_sim, N] -- this is how it is passed
    # from scipy's bootstrap
    confusion_arr = experiment_arr.sum(-1)
    return calculate_f1(confusion_arr)  # [n_sim, BOOT_SIM]


def simulate_f1_intervals(
    n_sim=1000,
    Ns=[3, 10, 30, 100],
    conf_levels=np.array([0.8, 0.9, 0.95]),
    save_to_pkl="f1_coverage_results.pkl",
):
    """
    Simulate coverage of F1 score intervals using:
      1) Bayesian HDI based on Dirichlet posterior
      2) Bootstrap CI

    For each N in Ns, and each confidence level in conf_levels:
      1. Sample true parameters from Dirichlet(1,1,1,1)
      2. Generate data from Multinomial(N, theta)
      3. Compute intervals and check if they contain the true F1

    Returns: dict with the coverage data
    """
    coverage_data = {}

    # Generate true parameters for all simulations
    true_thetas = dirichlet(alpha=[1, 1, 1, 1], size=n_sim)  # [n_sim, 4]
    true_f1s = calculate_f1(true_thetas.T)  # [n_sim]
    coverage_data["true_f1s"] = true_f1s

    for N in Ns:
        print(f"Computing N={N}")

        # Pre-generate all data
        data = np.zeros((N, 4, n_sim))  # [N, 4, n_sim]
        for i in range(n_sim):
            data[..., i] = np.random.multinomial(
                n=1, pvals=true_thetas[i], size=N
            )  # [N, 4]

        agg_data = data.sum(0)  # [N, 4, n_sim] -> [4, n_sim]
        coverage_bayes = np.zeros_like(conf_levels, dtype=float)
        coverage_boot = np.zeros_like(conf_levels, dtype=float)
        size_bayes = np.zeros_like(conf_levels, dtype=float)
        size_boot = np.zeros_like(conf_levels, dtype=float)
        nan_boot = np.zeros_like(conf_levels, dtype=float)

        for i, conf in enumerate(tqdm(conf_levels)):
            in_bayes = 0
            in_boot = 0
            sum_bayes = 0
            sum_boot = 0

            # --- Bayesian HDI using Dirichlet posterior ---
            # Posterior parameters = prior params + counts
            post_params = np.ones((4, 1)) + agg_data  # [4, n_sim]
            # Draw samples from posterior
            post_samples = np.array(
                [
                    dirichlet(post_params[:, i], size=2000)
                    for i in range(post_params.shape[1])
                ]
            )  # [n_sim, 2000, 4]
            f1_samples = calculate_f1(post_samples.transpose(2, 1, 0))  # [2000, n_sim]
            # bayes_hdi = az.hdi(f1_samples, hdi_prob=conf)  # [n_sim, 2]
            # QBI
            bayes_hdi = np.percentile(
                f1_samples, [100 * (1 - conf) / 2, 100 - 100 * (1 - conf) / 2], axis=0
            ).T  # [2, n_sim] - >[n_sim, 2]
            assert all(bayes_hdi[:, 1] - bayes_hdi[:, 0] >= 0)
            sum_bayes = (bayes_hdi[:, 1] - bayes_hdi[:, 0]).sum()
            in_bayes = (
                (bayes_hdi[:, 0] <= true_f1s) & (true_f1s <= bayes_hdi[:, 1])
            ).sum()

            # --- Bootstrap CI ---
            boot = bootstrap(
                (data,),  # [N, 4, n_sim]
                statistic=_stat_f1,
                confidence_level=conf,
                n_resamples=10000,
                method="BCa",
                vectorized=True,
            )
            ci_boot = boot.confidence_interval
            ci_boot_high_minus_low = ci_boot.high - ci_boot.low
            ci_boot_nans = np.isnan(ci_boot_high_minus_low)
            sum_boot = ci_boot_high_minus_low[~ci_boot_nans].sum()
            in_boot = (
                (ci_boot.low[~ci_boot_nans] <= true_f1s[~ci_boot_nans])
                & (true_f1s[~ci_boot_nans] <= ci_boot.high[~ci_boot_nans])
            ).sum()

            # Store coverage and interval sizes
            coverage_bayes[i] = in_bayes / n_sim
            coverage_boot[i] = in_boot / n_sim
            size_bayes[i] = sum_bayes / n_sim
            size_boot[i] = sum_boot / n_sim
            nan_boot[i] = ci_boot_nans.sum() / n_sim

        coverage_data[N] = {
            "confidence_levels": conf_levels.copy(),
            "coverage_bayes": coverage_bayes,
            "coverage_boot": coverage_boot,
            "size_bayes": size_bayes,
            "size_boot": size_boot,
            "nan_boot": nan_boot,
        }

    # Save results
    with open(save_to_pkl, "wb") as f:
        pickle.dump(coverage_data, f)

    return coverage_data


def plot_coverage_f1(
    N_list,
    coverage_results: dict,
    save_to_pdf: str = "PLOTS_FINAL/pdfs/f_scores.pdf",
    figsize=(6.8, 2.1),
):
    fig, axes = plt.subplots(1, len(N_list), figsize=figsize, sharey=True)
    colors = {"bayes": "blue", "boot": "brown", "nominal": "gray"}

    for idx, N in enumerate(N_list):
        confs = coverage_results[N]["confidence_levels"]
        ax = axes[idx]
        ax.plot(
            confs,
            coverage_results[N]["coverage_bayes"],
            # marker=markers["bayes_diff"],
            color=colors["bayes"],
            linestyle="-",
            label=f"Bayes" if N == N_list[0] else None,
        )
        ax.plot(
            confs,
            coverage_results[N]["coverage_boot"],
            # marker=markers["clt_diff"],
            color=colors["boot"],
            linestyle="-",
            label=f"Bootstrap" if N == N_list[0] else None,
        )
        # Diagonal line
        ax.plot(
            confs,
            confs,
            "--",
            color=colors["nominal"],
            label=r"$1-\alpha$" if N == N_list[0] else None,
            linewidth=2,
        )
        ax.set_title(rf"$N = {N}$")
        # ax.set_xlabel(r"Confidence level, $1-\alpha$")
        # if idx == 0:
        #     ax.set_ylabel("Coverage")
        ax.set_xlim(0.78, 1.01)
        ax.set_xticks([0.8, 0.9, 1.00])

        ax.grid(True, alpha=0.3)
    # ax.legend(loc="lower center")
    fig.supxlabel(r"Confidence level, $1-\alpha$", y=0.1)
    fig.supylabel("Coverage", x=0.02, y=0.6)
    fig.legend(
        loc="lower center",
        fancybox=True,
        shadow=True,
        ncol=7,
        bbox_to_anchor=(0.5, -0.04),
    )

    plt.tight_layout()
    plt.savefig(save_to_pdf, bbox_inches="tight", pad_inches=0.2)
    plt.show()


def plot_size_f1(
    N_list,
    coverage_results: dict,
    save_to_pdf: str = "PLOTS_FINAL/pdfs/f_scores_sizes_qbi.pdf",
    figsize=(6.8, 2.1),
):
    fig, axes = plt.subplots(1, len(N_list), figsize=figsize, sharey=True)
    colors = {"bayes": "blue", "boot": "brown", "nominal": "gray"}

    for idx, N in enumerate(N_list):
        confs = coverage_results[N]["confidence_levels"]
        ax = axes[idx]
        ax.plot(
            coverage_results[N].get(f"size_bayes", []),
            coverage_results[N]["coverage_bayes"],
            # marker=markers["bayes_diff"],
            color=colors["bayes"],
            linestyle="-",
            label=f"Bayes" if N == N_list[0] else None,
        )
        ax.plot(
            coverage_results[N].get(f"size_boot", []),
            coverage_results[N]["coverage_boot"],
            # marker=markers["clt_diff"],
            color=colors["boot"],
            linestyle="-",
            label=f"Bootstrap" if N == N_list[0] else None,
        )
        ax.set_title(rf"$N = {N}$")

        ax.grid(True, alpha=0.3)
    # ax.legend(loc="lower center")
    fig.supxlabel("Interval width", y=0.1)
    fig.supylabel("Coverage", x=0.02, y=0.6)
    fig.legend(
        loc="lower center",
        fancybox=True,
        shadow=True,
        ncol=7,
        bbox_to_anchor=(0.5, -0.04),
    )

    plt.tight_layout()
    plt.savefig(save_to_pdf, bbox_inches="tight", pad_inches=0.2)
    plt.show()


if __name__ == "__main__":
    np.random.seed(20250123)

    n_sim = 10000
    N_list = [3, 10, 30, 100]
    conf_levels = np.linspace(0.8, 0.995, 100)

    true_thetas_list = [
        (0.5, 0.5),
        (0.5, 0.47),
        (0.5, 0.42),
        (0.5, 0.35),
        (0.8, 0.8),
        (0.8, 0.77),
        (0.8, 0.72),
        (0.8, 0.65),
        (0.95, 0.95),
        (0.95, 0.92),
        (0.95, 0.87),
        (0.95, 0.8),
    ]
    coverage_results = simulate_f1_intervals(
        n_sim,
        N_list,
        conf_levels,
        save_to_pkl=f"results/coverage_results_f1score_qbi.pkl",
    )
    if False:
        n_sim = 3000
        for true_thetas in true_thetas_list:
            coverage_results = simulate_f1_intervals(
                n_sim,
                N_list,
                conf_levels,
                true_thetas=true_thetas,
                save_to_pkl=f"results/coverage_results_fscores_qbi_thetaA_{true_thetas[0]}_thetaB_{true_thetas[1]}.pkl",
            )

    plt.rc("font", size=8)  # controls default text sizes
    plt.rc("axes", titlesize=10)  # fontsize of the axes title
    plt.rc("axes", labelsize=9)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=7)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=7)  # fontsize of the tick labels
    plt.rc("legend", fontsize=7)  # legend fontsize
    plt.rc("figure", labelsize=8)  # size of figure-level labels eg supxlabel
    plt.rc("figure", titlesize=9)  # fontsize of the figure title

    plot_coverage_f1(
        N_list,
        coverage_results,
        save_to_pdf=f"PLOTS_FINAL/pdfs/f_scores_qbi.pdf",
        # figsize=(3.4, 2.0),
    )
    plot_size_f1(
        N_list,  # [1:3],
        coverage_results,
        save_to_pdf=f"PLOTS_FINAL/pdfs/f_scores_size_qbi.pdf",
    )
