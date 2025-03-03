####################################################################################################
## Baseline Experiment

python basic_experiments.py --experiment_name basic_small


####################################################################################################
## Mismatched Priors

### Beta(10,10)
python basic_experiments.py --experiment_name basic_small --custom_name basic_small_beta_10_10 --prior_alpha 10 --prior_beta 10

### Beta(100,20)
python basic_experiments.py --experiment_name basic_small --custom_name basic_small_beta_100_20 --prior_alpha 100 --prior_beta 20

### Beta(20,100)
python basic_experiments.py --experiment_name basic_small --custom_name basic_small_beta_20_100 --prior_alpha 20 --prior_beta 100


####################################################################################################
## Fixed thetas: 0.5, 0.8, 0.95 with BIGREPEATS (3000 repeats)
NUM_BIGREPEATS=3000

### theta = 0.5
python basic_experiments.py --experiment_name basic_small --custom_name basic_small_FIXED_50_BIGREPEATS --num_thetas 1 --fix_theta 0.5 --num_repeats $NUM_BIGREPEATS

### theta = 0.8
python basic_experiments.py --experiment_name basic_small --custom_name basic_small_FIXED_80_BIGREPEATS --num_thetas 1 --fix_theta 0.8 --num_repeats $NUM_BIGREPEATS

### theta = 0.95
python basic_experiments.py --experiment_name basic_small --custom_name basic_small_FIXED_95_BIGREPEATS --num_thetas 1 --fix_theta 0.95 --num_repeats $NUM_BIGREPEATS
