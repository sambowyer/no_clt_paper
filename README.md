# Code for "Position: Don’t use the CLT in LLM evals with fewer than a few hundred datapoints"

**ICML 2025 Spotlight**

Paper: https://arxiv.org/abs/2503.01747.

Note that simple, user-friendly versions of the Bayesian eval methods used in this paper are available as a standalone package at https://github.com/sambowyer/bayes_evals.

--------

Find the paper plots in `PLOTS_FINAL/` and langchain eval data (used in Figure 1) in `data`.

### Environment setup
To install the required packages and activate the conda environment, run:
```bash
conda env create -f env.yml
conda activate no_clt_evals
```

### Sections 3.1, 3.2, and 3.4 (IID, clustered and paired settings)
To run the experiments for sections 3.1, 3.2 and 3.4, you need to install the `eval_lib` package:

```bash
pip install -e eval_lib
```

Section 3.1 uses `basic_experiment.py`, section 3.2 uses `subtask_experiment.py` and section 3.4 uses `paired_experiment.py`.

To run all the experiments neccessary for these sections, run the files in `experiment_scripts/`:
- `sh experiment_scripts/run_basic_experiments.sh`
- `sh experiment_scripts/run_subtask_experiments.sh`
- `sh experiment_scripts/run_paired_experiments.sh`

(Or just pick out the experiments you want to run from these files.)
These will by default create plots in `plots/` and save the results in `results/` (which are currently empty).

Then generate the plots (into `FINAL_PLOTS/`) as follows:
- `python plot_real_data_error_bars`: generates Figure 1 and Figure 8 (error bars on real langchain eval data)
- `python plot_beta_pdfs.py`: generates Figure 13 (densities of beta distributions)
- `python plot_main_experiments.py`: generates all other Figures for sections 3.1, 3.2 and 3.4 (and their corresponding appendix figures -- except for the fixed-theta appendix figures)
- `python plot_fixed_theta_experiments.py`: generates the fixed-theta figures for the appendix ablation subsections corresponding to sections 3.1, 3.2 and 3.4 (Figures 17, 21, 33, 34, and 35)

### Sections 3.3, and 3.5 (independent model comparison and F1 scores)
To obtain the Figures related to the independent model comparison and F1 score settings, use the scripts: `indep_model_comparison.py` and `f_score_comparison.py` respectively. These scripts will generate the plots in `FINAL_PLOTS/pdfs/` and save the results in `results/`.

# Citation
```bibtex
@inproceedings{bowyer2025positiondontuseclt,
      title={Position: Don't use the CLT in LLM evals with fewer than a few hundred datapoints}, 
      author={Sam Bowyer and Laurence Aitchison and Desi R. Ivanova},
      year={2025},
      booktitle={Forty-second International Conference on Machine Learning Position Paper Track},
      url={https://arxiv.org/abs/2503.01747}, 
}
```
