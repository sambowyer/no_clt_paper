import math
import numpy as np
import arviz as az
from scipy.stats import norm, bootstrap
from scipy.stats.contingency import odds_ratio
from tqdm import tqdm
import pickle
import os
from collections import defaultdict

import matplotlib.pyplot as plt


def _stat_diff(x, y, axis):
    # print(x.shape)
    diff = x.mean(axis=-1) - y.mean(axis=-1)
    # print("diff shape", diff.shape)
    return diff


def _compute_or(arr, eps=1e-9):
    """
    Compute odds = p / (1 - p) from a 0/1 array.
    Clamps extreme values slightly to avoid division by zero or inf.
    """
    p = arr.mean(axis=-1)
    p = np.clip(p, eps, 1 - eps)
    return p / (1 - p)


def _stat_or(x, y, axis):
    return _compute_or(x) / _compute_or(y)


def simulate_two_proportions(
    n_sim,
    Ns,
    conf_levels,
    true_thetas: tuple[float, float] | None = None,
    betadist_A: tuple[float, float] = (1, 1),
    betadist_B: tuple[float, float] = (1, 1),
    save_to_pkl: str = "coverage_results_model_comp_qbi.pkl",
):
    """
    Simulate coverage of two-proportion intervals for:
      1) CLT-based CI on (theta_A - theta_B)
      2) Bayesian HDI on (theta_A - theta_B)
      3) Fisher exact CI on odds ratio
      4) Bayesian HDI on odds ratio

    For each N in Ns, and each confidence level in conf_levels:
      1. Sample theta_A, theta_B ~ Uniform(0,1)
      2. Generate X_A ~ Binomial(N, theta_A), X_B ~ Binomial(N, theta_B)
      3. Compute intervals and check if they contain the true difference or true OR.

    Returns: dict with the coverage data
    """

    coverage_data = {}
    is0_in_clt_diff = {}
    is0_in_bayes_diff = {}
    is0_in_boot_diff = {}
    is1_in_fisher_or = {}
    is1_in_bayes_or = {}
    is1_in_boot_or = {}

    if true_thetas is not None:
        theta_As = np.array(true_thetas[0]).repeat(n_sim)  # [n_sim]
        theta_Bs = np.array(true_thetas[1]).repeat(n_sim)  # [n_sim]
    else:
        # theta_As = np.random.rand(n_sim)  # [n_sim]
        # theta_Bs = np.random.rand(n_sim)  # [n_sim]
        theta_As = np.random.beta(*betadist_A, size=n_sim)  # [n_sim]
        theta_Bs = np.random.beta(*betadist_B, size=n_sim)  # [n_sim]

    coverage_data["theta_diff"] = theta_As - theta_Bs
    coverage_data["theta_OR"] = (theta_As / (1 - theta_As)) / (
        theta_Bs / (1 - theta_Bs)
    )

    for N in Ns:
        print(f"Computing N={N}")

        # pre-generate data: use the same data at different significance levels
        dataAs = np.random.binomial(1, p=theta_As, size=(N, n_sim))  # [N, n_sim]
        dataBs = np.random.binomial(1, p=theta_Bs, size=(N, n_sim))  # [N, n_sim]

        coverage_clt_diff = np.zeros_like(conf_levels, dtype=float)
        coverage_bayes_diff = np.zeros_like(conf_levels, dtype=float)
        coverage_fisher_or = np.zeros_like(conf_levels, dtype=float)
        coverage_bayes_or = np.zeros_like(conf_levels, dtype=float)
        coverage_boot_diff = np.zeros_like(conf_levels, dtype=float)
        coverage_boot_or = np.zeros_like(conf_levels, dtype=float)

        size_clt_diff = np.zeros_like(conf_levels, dtype=float)
        size_bayes_diff = np.zeros_like(conf_levels, dtype=float)
        size_fisher_or = np.zeros_like(conf_levels, dtype=float)
        size_bayes_or = np.zeros_like(conf_levels, dtype=float)
        size_boot_diff = np.zeros_like(conf_levels, dtype=float)
        size_boot_or = np.zeros_like(conf_levels, dtype=float)

        fisher_infs = np.zeros_like(conf_levels, dtype=int)

        # can it discriminate the difference?
        is0_in_clt_diff[N] = {}
        is0_in_bayes_diff[N] = {}
        is0_in_boot_diff[N] = {}
        is1_in_fisher_or[N] = {}
        is1_in_bayes_or[N] = {}
        is1_in_boot_or[N] = {}

        for i, conf in enumerate(tqdm(conf_levels)):
            z_val = norm.ppf(1 - (1 - conf) / 2)

            in_clt_diff = 0
            in_bayes_diff = 0
            in_fisher_or = 0
            in_bayes_or = 0
            in_boot_diff = 0
            in_boot_or = 0

            sum_clt_diff = 0
            sum_bayes_diff = 0
            sum_fisher_or = 0
            sum_bayes_or = 0
            sum_boot_diff = 0
            sum_boot_or = 0

            fisher_inf = 0

            ###
            is0_in_clt_diff[N][conf] = []
            is0_in_bayes_diff[N][conf] = []
            is0_in_boot_diff[N][conf] = []
            is1_in_fisher_or[N][conf] = []
            is1_in_bayes_or[N][conf] = []
            is1_in_boot_or[N][conf] = []

            for nnn in range(n_sim):
                # sample true parameters
                theta_A = theta_As[nnn]  # float
                theta_B = theta_Bs[nnn]  # float
                dataA = dataAs[:, nnn]  # [N]
                dataB = dataBs[:, nnn]  # [N]

                # true difference
                true_diff = theta_A - theta_B
                # true odds ratio, taking care of potential inf
                true_or = (theta_A / (1 - theta_A)) / (theta_B / (1 - theta_B))

                X_A = dataA.sum()
                X_B = dataB.sum()
                # X_A = np.random.binomial(n=N, p=theta_A)
                # X_B = np.random.binomial(n=N, p=theta_B)
                p_A = X_A / N
                p_B = X_B / N

                # --- CLT-based (Wald) interval on difference (p_A - p_B) ---
                diff_hat = p_A - p_B
                se_diff = np.sqrt(p_A * (1 - p_A) / N + p_B * (1 - p_B) / N)
                hw = z_val * se_diff
                clt_lower = diff_hat - hw
                clt_upper = diff_hat + hw
                assert clt_upper - clt_lower >= 0
                sum_clt_diff += clt_upper - clt_lower
                if clt_lower <= true_diff <= clt_upper:
                    in_clt_diff += 1
                if clt_lower <= 0 <= clt_upper:
                    is0_in_clt_diff[N][conf].append(1)
                else:
                    is0_in_clt_diff[N][conf].append(0)

                # --- Bayesian HDI on difference ---
                # Posterior draws for theta_A, theta_B
                drawsA = np.random.beta(1 + X_A, 1 + (N - X_A), size=2000)
                drawsB = np.random.beta(1 + X_B, 1 + (N - X_B), size=2000)
                diff_samples = drawsA - drawsB
                assert (diff_samples <= 1).all()
                assert (diff_samples >= -1).all()

                # diff_hdi = az.hdi(diff_samples, hdi_prob=conf) # (2,)
                diff_hdi = np.percentile(
                    diff_samples,
                    [100 * (1 - conf) / 2, 100 - 100 * (1 - conf) / 2],
                    axis=0,
                )
                assert diff_hdi[1] - diff_hdi[0] >= 0
                assert diff_hdi[1] - diff_hdi[0] <= 2
                sum_bayes_diff += diff_hdi[1] - diff_hdi[0]
                if diff_hdi[0] <= true_diff <= diff_hdi[1]:
                    in_bayes_diff += 1
                if diff_hdi[0] <= 0 <= diff_hdi[1]:
                    is0_in_bayes_diff[N][conf].append(1)
                else:
                    is0_in_bayes_diff[N][conf].append(0)
                # prob_A_is_better

                # --- Fisher exact CI on odds ratio ---
                # Build 2x2 table: [ [ X_A, N - X_A], [ X_B, N - X_B ] ]
                table = np.array([[X_A, N - X_A], [X_B, N - X_B]])
                fisher_result = odds_ratio(
                    table, kind="conditional"  # --> fisher exact
                )
                # fisher_result has .confidence_interval(confidence_level=...) method
                ci_or = fisher_result.confidence_interval(
                    confidence_level=conf, alternative="two-sided"
                )

                assert ci_or.high - ci_or.low >= 0
                if math.isinf(ci_or.high) or math.isinf(ci_or.low):
                    fisher_inf += 1
                else:
                    sum_fisher_or += ci_or.high - ci_or.low
                if ci_or.low <= true_or <= ci_or.high:
                    in_fisher_or += 1
                if ci_or.low <= 1 <= ci_or.high:
                    is1_in_fisher_or[N][conf].append(1)
                else:
                    is1_in_fisher_or[N][conf].append(0)

                # ---  Bayesian HDI on odds ratio ---
                or_samples = (drawsA / (1 - drawsA)) / (drawsB / (1 - drawsB))
                # or_hdi = az.hdi(or_samples, hdi_prob=conf)
                # qbi
                or_hdi = np.percentile(
                    or_samples,
                    [100 * (1 - conf) / 2, 100 - 100 * (1 - conf) / 2],
                    axis=0,
                )
                assert or_hdi[1] - or_hdi[0] >= 0
                sum_bayes_or += or_hdi[1] - or_hdi[0]
                if or_hdi[0] <= true_or <= or_hdi[1]:
                    in_bayes_or += 1
                if or_hdi[0] <= 1 <= or_hdi[1]:
                    is1_in_bayes_or[N][conf].append(1)
                else:
                    is1_in_bayes_or[N][conf].append(0)

            # run bootstrap vectorised over n_sim -> scipy.stats.bootstrap
            # --- bootstrap on diff ----
            boot_diff = bootstrap(
                (dataAs, dataBs),
                statistic=_stat_diff,
                confidence_level=conf,
                n_resamples=10000,
                method="BCa",
                vectorized=True,
            )
            ci_boot_diff = boot_diff.confidence_interval
            ci_boot_high_minus_low = ci_boot_diff.high - ci_boot_diff.low
            ci_boot_diff_num_nas = np.isnan(ci_boot_high_minus_low).sum()
            sum_boot_diff = ci_boot_high_minus_low[
                ~np.isnan(ci_boot_high_minus_low)
            ].sum()
            in_boot_diff = (
                (ci_boot_diff.low <= coverage_data["theta_diff"])
                & (coverage_data["theta_diff"] <= ci_boot_diff.high)
            ).sum()
            is0_in_boot_diff[N][conf] = list(
                ((ci_boot_diff.low <= 0) & (0 <= ci_boot_diff.high)) * 1.0
            )

            # --- bootstrap on OR ----
            boot_or = bootstrap(
                (dataAs, dataBs),
                statistic=_stat_or,
                confidence_level=conf,
                n_resamples=10000,
                method="BCa",
                vectorized=True,
            )
            ci_boot_or = boot_or.confidence_interval
            ci_boot_or_high_minus_low = ci_boot_or.high - ci_boot_or.low
            ci_boot_or_num_nas = np.isnan(ci_boot_high_minus_low).sum()
            sum_boot_or = ci_boot_or_high_minus_low[
                ~np.isnan(ci_boot_or_high_minus_low)
            ].sum()
            in_boot_or = (
                (ci_boot_or.low <= coverage_data["theta_OR"])
                & (coverage_data["theta_OR"] <= ci_boot_or.high)
            ).sum()
            is1_in_boot_or[N][conf] = list(
                ((ci_boot_or.low <= 1) & (1 <= ci_boot_or.high)) * 1.0
            )

            # especially for small N theres a lot of inf-s
            # print(f"conf level = {conf}, num inf = {fisher_inf}")

            # Store coverage (fraction of times the interval contained truth)
            coverage_clt_diff[i] = in_clt_diff / n_sim
            coverage_bayes_diff[i] = in_bayes_diff / n_sim
            coverage_fisher_or[i] = in_fisher_or / n_sim
            coverage_bayes_or[i] = in_bayes_or / n_sim
            coverage_boot_diff[i] = in_boot_diff / n_sim
            coverage_boot_or[i] = in_boot_or / n_sim

            size_clt_diff[i] = sum_clt_diff / n_sim
            size_bayes_diff[i] = sum_bayes_diff / n_sim
            size_fisher_or[i] = sum_fisher_or / n_sim
            size_bayes_or[i] = sum_bayes_or / n_sim
            size_boot_diff[i] = sum_boot_diff / n_sim
            size_boot_or[i] = sum_boot_or / n_sim

            fisher_infs[i] = fisher_inf

        coverage_data[N] = {
            "confidence_levels": conf_levels.copy(),
            "fisher_infs": fisher_infs,
            "coverage_clt_diff": coverage_clt_diff,
            "coverage_bayes_diff": coverage_bayes_diff,
            "coverage_fisher_or": coverage_fisher_or,
            "coverage_bayes_or": coverage_bayes_or,
            "coverage_boot_diff": coverage_boot_diff,
            "coverage_boot_or": coverage_boot_or,
            "size_clt_diff": size_clt_diff,
            "size_bayes_diff": size_bayes_diff,
            "size_fisher_or": size_fisher_or,
            "size_bayes_or": size_bayes_or,
            "size_boot_diff": size_boot_diff,
            "size_boot_or": size_boot_or,
        }

    coverage_data["is0_in_clt_diff"] = is0_in_clt_diff
    coverage_data["is0_in_bayes_diff"] = is0_in_bayes_diff
    coverage_data["is0_in_boot_diff"] = is0_in_boot_diff
    coverage_data["is1_in_fisher_or"] = is1_in_fisher_or
    coverage_data["is1_in_bayes_or"] = is1_in_bayes_or
    coverage_data["is1_in_boot_or"] = is1_in_boot_or

    with open(save_to_pkl, "wb") as f:
        pickle.dump(coverage_data, f)

    return coverage_data


def plot_coverage_comparison(
    N_list, coverage_results: dict, save_to_pdf: str = "PLOTS_FINAL/pdfs/indep_model_comparison.pdf"
):
    fig, axes = plt.subplots(1, len(N_list), figsize=(6.8, 2.1), sharey=True)

    colors = {
        "clt_diff": "darkorange",
        "bayes_diff": "blue",
        "bayes_or": "green",
        "fisher_or": "purple",
        "boot_diff": "brown",
        "boot_or": "darkred",
        "nominal": "gray",
    }

    for idx, N in enumerate(N_list):
        ax = axes[idx]

        confs = coverage_results[N]["confidence_levels"]
        ax.plot(
            confs,
            coverage_results[N]["coverage_bayes_diff"],
            # marker=markers["bayes_diff"],
            color=colors["bayes_diff"],
            linestyle="-",
            label=f"Bayes Diff" if N == N_list[0] else None,
        )
        ax.plot(
            confs,
            coverage_results[N]["coverage_clt_diff"],
            # marker=markers["clt_diff"],
            color=colors["clt_diff"],
            linestyle="-",
            label=f"CLT Diff" if N == N_list[0] else None,
        )
        ax.plot(
            confs,
            coverage_results[N]["coverage_boot_diff"],
            # marker=markers["boot_diff"],
            color=colors["boot_diff"],
            linestyle="-",
            label="Bootstrap Diff" if N == N_list[0] else None,
        )
        ax.plot(
            confs,
            coverage_results[N]["coverage_bayes_or"],
            # marker=markers["bayes_or"],
            color=colors["bayes_diff"],
            linestyle="dotted",
            label=f"Bayes OR" if N == N_list[0] else None,
        )
        ax.plot(
            confs,
            coverage_results[N]["coverage_fisher_or"],
            # marker=markers["fisher_or"],
            color=colors["clt_diff"],
            linestyle="dotted",
            label=f"Fisher OR" if N == N_list[0] else None,
        )
        ax.plot(
            confs,
            coverage_results[N]["coverage_boot_or"],
            # marker=markers["boot_or"],
            color=colors["boot_diff"],
            linestyle="dotted",
            label="Bootstrap OR" if N == N_list[0] else None,
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


def plot_size_comparison(
    N_list, coverage_results, save_to_pdf: str = "PLOTS_FINAL/pdfs/indep_model_size_comparison.pdf"
):
    # Define colors for different coverage metrics (consistent with coverage plots)
    colors = {
        "clt_diff": "darkorange",
        "bayes_diff": "blue",
        "boot_diff": "brown",
        "fisher_or": "darkorange",
        "bayes_or": "blue",
        "boot_or": "brown",
    }

    # Initialize the figure with 2 rows (Diff and OR) and len(N_list) columns
    fig, axes = plt.subplots(
        2,
        len(N_list),
        figsize=(6.8, 4),
        sharey=False,  # Different y-scales for Diff and OR
        sharex=False,
    )

    # Ensure axes is a 2D array even if len(N_list) == 1
    if len(N_list) == 1:
        axes = axes.reshape(2, 1)

    # Define metric groups
    metric_groups = {
        "Diff": ["clt_diff", "bayes_diff", "boot_diff"],
        "OR": ["fisher_or", "bayes_or", "boot_or"],
    }

    # Iterate over each N and plot metrics
    for col, N in enumerate(N_list):
        # Plot "Diff" metrics in the first row
        ax_diff = axes[0, col]
        for metric in metric_groups["Diff"]:
            size = coverage_results[N].get(f"size_{metric}", [])
            coverage = coverage_results[N].get(f"coverage_{metric}", [])
            # if size and coverage:
            label = (
                metric.replace("_", " ").title().replace("Clt", "CLT")
                if col == 0
                else None
            )
            ax_diff.plot(
                size,
                coverage,
                # marker=markers[metric],
                color=colors[metric],
                linestyle="-",
                label=label,
            )
        # Plot nominal line for "Diff"
        confs_diff = coverage_results[N].get("confidence_levels", [])
        ax_diff.set_title(rf"$N = {N}$")
        ax_diff.grid(True, alpha=0.3)

        # Plot "OR" metrics in the second row
        ax_or = axes[1, col]
        for metric in metric_groups["OR"]:
            coverage = coverage_results[N].get(f"coverage_{metric}", [])
            if metric == "boot_or":
                total_clipped = 0
                boot_size = coverage_results[N].get("size_boot_or", [])
                max_or_value = max(
                    max(coverage_results[N]["size_fisher_or"]),
                    max(coverage_results[N]["size_bayes_or"]),
                )
                print("max_or_value", max_or_value)
                size = [min(x, max_or_value) for x in boot_size]
                clipped_points = sum(1 for x, y in zip(boot_size, coverage) if x != y)
                total_clipped += clipped_points
                print(
                    f"Total points clipped for N={N} in Bootstrap OR: {total_clipped}"
                )
            else:
                size = coverage_results[N].get(f"size_{metric}", [])

            # if size and coverage:
            label = metric.replace("_", " ").title() if col == 0 else None
            ax_or.plot(
                size,
                coverage,
                color=colors[metric],
                linestyle="dotted",
                label=label,
            )
        # Plot nominal line for "OR"
        confs_or = coverage_results[N].get("confidence_levels", [])
        ax_or.set_title(rf"$N = {N}$")
        ax_or.grid(True, alpha=0.3)

    # Adjust layout to make room for super labels and legend
    plt.subplots_adjust(
        top=0.88, bottom=0.20, left=0.07, right=0.95, hspace=0.35, wspace=0.3
    )

    # Create a single legend for all subplots
    handles = [
        plt.Line2D(
            [0],
            [0],
            color=colors["bayes_diff"],
            linestyle="-",
            label="Bayes Diff",
        ),
        plt.Line2D(
            [0],
            [0],
            color=colors["clt_diff"],
            linestyle="-",
            label="CLT Diff",
        ),
        plt.Line2D(
            [0],
            [0],
            color=colors["boot_diff"],
            linestyle="-",
            label="Bootstrap Diff",
        ),
        plt.Line2D(
            [0],
            [0],
            color=colors["bayes_or"],
            linestyle="dotted",
            label="Bayes OR",
        ),
        plt.Line2D(
            [0],
            [0],
            color=colors["fisher_or"],
            linestyle="dotted",
            label="Fisher OR",
        ),
        plt.Line2D(
            [0],
            [0],
            color=colors["boot_or"],
            linestyle="dotted",
            label="Bootstrap OR",
        ),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=7, bbox_to_anchor=(0.5, 0.02))

    fig.supxlabel("Interval width", y=0.1)
    fig.supylabel("Coverage", x=0.00, y=0.5)

    # Save and display the figure
    plt.savefig(save_to_pdf, bbox_inches="tight", pad_inches=0.2)
    plt.show()


if __name__ == "__main__":
    np.random.seed(20250117)

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
    if True:
        n_sim = 3000
        for true_thetas in true_thetas_list:
            coverage_results = simulate_two_proportions(
                n_sim,
                N_list,
                conf_levels,
                true_thetas=true_thetas,
                save_to_pkl=f"results/coverage_results_model_comp_qbi_thetaA_{true_thetas[0]}_thetaB_{true_thetas[1]}_qbi.pkl",
            )

        plt.rc("font", size=8)  # controls default text sizes
        plt.rc("axes", titlesize=10)  # fontsize of the axes title
        plt.rc("axes", labelsize=9)  # fontsize of the x and y labels
        plt.rc("xtick", labelsize=7)  # fontsize of the tick labels
        plt.rc("ytick", labelsize=7)  # fontsize of the tick labels
        plt.rc("legend", fontsize=7)  # legend fontsize
        plt.rc("figure", labelsize=8)  # size of figure-level labels eg supxlabel
        plt.rc("figure", titlesize=9)  # fontsize of the figure title

        plot_coverage_comparison(
            N_list,
            coverage_results,
            save_to_pdf=f"PLOTS_FINAL/pdfs/indep_model_comparison_qbi.pdf",
        )
    plot_size_comparison(
        N_list, coverage_results, save_to_pdf="PLOTS_FINAL/pdfs/indep_model_size_comparison_qbi.pdf"
    )
