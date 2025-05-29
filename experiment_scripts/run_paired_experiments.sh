####################################################################################################
## Baseline Experiment

python paired_experiments.py --experiment_name paired_small --custom_name true_paired_small --positive_rho


####################################################################################################
## Mismatched Priors

### theta_B ~ Beta(10,10)
python paired_experiments.py --experiment_name paired_small --custom_name true_paired_small_beta_10_10 --prior_alpha_B 10 --prior_beta_B 10 --positive_rho

### theta_B ~ Beta(100,20)
python paired_experiments.py --experiment_name paired_small --custom_name true_paired_small_beta_100_20 --prior_alpha_B 100 --prior_beta_B 20 --positive_rho

### theta_B ~ Beta(20,100)
python paired_experiments.py --experiment_name paired_small --custom_name true_paired_small_beta_20_100 --prior_alpha_B 20 --prior_beta_B 100 --positive_rho

### theta_A ~ Beta(100,20) AND theta_B ~ Beta(100,20)
python paired_experiments.py --experiment_name paired_small --custom_name true_paired_A10020_B10020_small --prior_alpha_A 100 --prior_beta_A 20 --prior_alpha_B 100 --prior_beta_B 20 --positive_rho

### theta_A ~ Beta(100,20) AND theta_B ~ Beta(20,100)
python paired_experiments.py --experiment_name paired_small --custom_name true_paired_A10020_B20100_small --prior_alpha_A 100 --prior_beta_A 20 --prior_alpha_B 20 --prior_beta_B 100 --positive_rho


####################################################################################################
## Mismatched Priors with FIXED theta_A and theta_B 

### theta_A = 0.5, theta_B = 0.5
python paired_experiments.py --experiment_name paired_small --custom_name paired_small_FIXED_A50_B50 --num_thetas 1 --fix_theta_A 0.50 --fix_theta_B 0.50 --positive_rho

### theta_A = 0.5, theta_B = 0.47
python paired_experiments.py --experiment_name paired_small --custom_name paired_small_FIXED_A50_B47 --num_thetas 1 --fix_theta_A 0.50 --fix_theta_B 0.47 --positive_rho

### theta_A = 0.5, theta_B = 0.42
python paired_experiments.py --experiment_name paired_small --custom_name paired_small_FIXED_A50_B42 --num_thetas 1 --fix_theta_A 0.50 --fix_theta_B 0.42 --positive_rho

#########################

### theta_A = 0.8, theta_B = 0.8
python paired_experiments.py --experiment_name paired_small --custom_name paired_small_FIXED_A80_B80 --num_thetas 1 --fix_theta_A 0.80 --fix_theta_B 0.80 --positive_rho

### theta_A = 0.8, theta_B = 0.77
python paired_experiments.py --experiment_name paired_small --custom_name paired_small_FIXED_A80_B77 --num_thetas 1 --fix_theta_A 0.80 --fix_theta_B 0.77 --positive_rho

### theta_A = 0.8, theta_B = 0.72
python paired_experiments.py --experiment_name paired_small --custom_name paired_small_FIXED_A80_B72 --num_thetas 1 --fix_theta_A 0.80 --fix_theta_B 0.72 --positive_rho

#########################

### theta_A = 0.95, theta_B = 0.95
python paired_experiments.py --experiment_name paired_small --custom_name paired_small_FIXED_A95_B95 --num_thetas 1 --fix_theta_A 0.95 --fix_theta_B 0.95 --positive_rho

### theta_A = 0.95, theta_B = 0.92
python paired_experiments.py --experiment_name paired_small --custom_name paired_small_FIXED_A95_B92 --num_thetas 1 --fix_theta_A 0.95 --fix_theta_B 0.92 --positive_rho

### theta_A = 0.95, theta_B = 0.87
python paired_experiments.py --experiment_name paired_small --custom_name paired_small_FIXED_A95_B87 --num_thetas 1 --fix_theta_A 0.95 --fix_theta_B 0.87 --positive_rho
