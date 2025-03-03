import numpy as np
import scipy.stats as stats
import torch as t

from .intervals import bootstrap_confidence_interval, get_bayes_posterior
from .binorm import binorm_cdf

__all__ = [
    'clt_unpaired_confidence_interval',
    'clt_paired_confidence_interval',
    'bootstrap_paired_confidence_interval',
    
    'importance_sampled_paired_credible_interval',
    'bayes_unpaired_credible_interval',
]

################################################################
## Frequentist (Confidence Intervals)

### CLT (ignoring pairing)
def clt_unpaired_confidence_interval(data_A, data_B, alpha):
    '''
    Compute confidence intervals for theta_A - theta_B ignoring question-pairing (and any clustering).
    
    Parameters
    ----------
    data_A, data_B: np.ndarray
        Binary data, batch dimension is the last dimension
    alpha: float, or list of floats
        Significance levels

    Returns
    -------
    confidence_interval: tuple
        Lower and upper bounds of the confidence interval
    '''
    if isinstance(alpha, float):
        alpha = np.array([alpha])
    else:
        alpha = np.array(alpha)

    # get SE_CLT for each dataset
    N_A = data_A.shape[-1]
    N_B = data_B.shape[-1]
    X_bar_A = data_A.mean(-1)
    X_bar_B = data_B.mean(-1)
    SE_CLT_A = np.sqrt(X_bar_A * (1 - X_bar_A) / N_A)
    SE_CLT_B = np.sqrt(X_bar_B * (1 - X_bar_B) / N_B)

    # get the difference in means
    diff = X_bar_A - X_bar_B

    # get the standard error of the difference
    SE_diff = np.sqrt(SE_CLT_A**2 + SE_CLT_B**2)

    # return the confidence interval
    z = stats.norm.ppf(1 - alpha / 2)

    delta = z[:,None] * SE_diff[None,...] # shape == (len(alpha), *data.shape[:-1])

    output = np.array([diff[None,:] - delta, diff[None,:] + delta]).transpose(-2, -1, 0) # shape == (len(alpha), *data.shape[:-1], 2)

    if len(alpha) == 1:
        return output[0]
    return output


### Paired CLT
def clt_paired_confidence_interval(data_A, data_B, alpha):
    '''
    Compute confidence intervals in the subtask setting according to paired standard error formulation.
    (See Miller, 2024: https://arxiv.org/abs/2411.00640 for details.)
    
    Parameters
    ----------
    data_A, data_B: np.ndarray
        Binary data, batch dimension is the last dimension
    alpha: float, or list of floats
        Significance levels

    Returns
    -------
    confidence_interval: tuple
        Lower and upper bounds of the confidence interval
    '''
    if isinstance(alpha, float):
        alpha = np.array([alpha])
    else:
        alpha = np.array(alpha)

    assert data_A.shape == data_B.shape

    data_diff = data_A - data_B  # i.e. {s_{A-B,i}} for i in 1...N
    N = data_diff.shape[-1]

    data_diff_mean = data_diff.mean(-1) # i.e. {s_bar_{A-B}}

    SE_paired = np.sqrt(np.sum((data_diff - data_diff_mean[...,None])**2, axis=-1)/(N*(N-1))) # shape == (*data.shape[:-1])

    # return the confidence interval
    z = stats.norm.ppf(1 - alpha / 2)

    delta = z[:,None] * SE_paired[None,...] # shape == (len(alpha), *data.shape[:-1])

    output = np.array([data_diff_mean[None,:] - delta, data_diff_mean[None,:] + delta]).transpose(-2, -1, 0) # shape == (len(alpha), *data.shape[:-1], 2)

    if len(alpha) == 1:
        return output[0]
    return output


### Bootstrap (ignoring pairing)
def bootstrap_paired_confidence_interval(data_A, data_B, alpha, K=100, sample_length=None):
    '''
    Compute confidence intervals for theta_A - theta_B ignoring question-pairing (and any clustering) via bootstrapping.
    
    Parameters
    ----------
    data_A, data_B: np.ndarray
        Binary data, batch dimension is the last dimension
    alpha: float, or list of floats
        Significance levels
    K: int
        Number of bootstrap samples
    sample_length: int
        Number of samples in each bootstrap sample. If None, defaults to the length of the
        original data.

    Returns
    -------
    confidence_interval: tuple
        Lower and upper bounds of the confidence interval
    '''
    
    return bootstrap_confidence_interval(data_A - data_B, alpha, K=K, sample_length=sample_length)



################################################################
## Bayesian (Credible Intervals)

### Importance sampling - 2D Gaussian model
def importance_sampled_paired_credible_interval(data_A, data_B, alpha, num_samples=10_000, max_parallel_repeats=50, rho_proposal_range=(-1 + 1e-20, 1 - 1e-20), num_monte_carlo_samples=100_000, output_cpu=True):
    '''
    data_A, data_B: np.ndarray
        Binary data, batch dimension is the last dimension
    alpha: float
        Significance level
    num_samples: int
        Number of importance samples
    max_parallel_repeats: int
        Maximum number of repeats to run in parallel
    rho_proposal_range: tuple
        Range of rho values to sample from (uniform proposal)

    Returns
    -------
    credible_interval: np.ndarray
        Lower and upper bounds of the credible interval
    '''
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')

    if isinstance(alpha, float):
        alpha = t.tensor([alpha], dtype=t.float32).to(device)
    elif isinstance(alpha, list) or isinstance(alpha, np.ndarray):
        alpha = t.tensor(alpha, dtype=t.float32).to(device)

    if len(data_A.shape) == 1:
        data_A = data_A[None,:]

    if len(data_B.shape) == 1:
        data_B = data_B[None,:]

    repeats = data_A.shape[0]
    N = data_A.shape[-1]

    if isinstance(data_A, np.ndarray):
        data_A = t.tensor(data_A).to(device)
    if isinstance(data_B, np.ndarray):
        data_B = t.tensor(data_B).to(device)

    assert data_A.shape == data_B.shape
    assert len(data_A.shape) == 2

    if repeats > max_parallel_repeats:
        interval = t.zeros((len(alpha), repeats, 2), device=device)

        for i in range(repeats // max_parallel_repeats):
            interval[:, i*max_parallel_repeats:(i+1)*max_parallel_repeats, :] = importance_sampled_paired_credible_interval(
                data_A[i*max_parallel_repeats:(i+1)*max_parallel_repeats, :], 
                data_B[i*max_parallel_repeats:(i+1)*max_parallel_repeats, :], 
                alpha, num_samples, max_parallel_repeats, rho_proposal_range, num_monte_carlo_samples,
                output_cpu=False
            )

    else:
        # sample a bunch of theta_As, theta_Bs and rhos
        theta_As = t.distributions.Beta(1, 1).sample((repeats, num_samples)).to(device)
        theta_Bs = t.distributions.Beta(1, 1).sample((repeats, num_samples)).to(device)
        rhos     = t.distributions.Uniform(rho_proposal_range[0], rho_proposal_range[1]).sample((repeats, num_samples)).to(device)

        diff = theta_As - theta_Bs

        # calculate the mus for the 2D Gaussian
        mu_A = t.distributions.Normal(0, 1).icdf(theta_As)
        mu_B = t.distributions.Normal(0, 1).icdf(theta_Bs)
        assert mu_A.shape == mu_B.shape == (repeats, num_samples,)

        # stack the mus
        mu = t.stack([mu_A, mu_B], dim=-1)
        assert mu.shape == (repeats, num_samples, 2)

        # create the 2x2 contingency table (flattened)
        S = (data_A * data_B).sum(-1, keepdims=True)             # S = A correct,   B correct
        T = (data_A * (1 - data_B)).sum(-1, keepdims=True)       # T = A correct,   B incorrect
        U = ((1 - data_A) * data_B).sum(-1, keepdims=True)       # U = A incorrect, B correct
        V = ((1 - data_A) * (1 - data_B)).sum(-1, keepdims=True) # V = A incorrect, B incorrect

        assert S.shape == T.shape == U.shape == V.shape == (repeats,1)

        # calculate the probabilities of each case
        theta_V = binorm_cdf(0, 0, mu_A, mu_B, 1, 1, rhos)
        theta_S = theta_As + theta_Bs + theta_V - 1
        theta_T = 1 - theta_Bs - theta_V
        theta_U = 1 - theta_As - theta_V

        log_likelihoods = S * t.log(theta_S) + T * t.log(theta_T) + U * t.log(theta_U) + V * t.log(theta_V)
        assert log_likelihoods.shape == (repeats, num_samples)

        log_prior_prob = t.log(t.tensor(1.0, device=device) * t.tensor(1.0, device=device) * t.tensor(1/2, device=device))
        log_proposal_prob = t.log(t.tensor(1.0, device=device) * t.tensor(1.0, device=device) * t.tensor(1/(rho_proposal_range[1]-rho_proposal_range[0]), device=device))

        log_weights = log_likelihoods + log_prior_prob - log_proposal_prob

        # max_log_weights = np.nanmax(log_weights, dim=-1, keepdim=True)
        max_log_weights = t.max(t.nan_to_num(log_weights, nan=-t.inf), dim=-1, keepdim=True).values
        weights = t.exp(log_weights - max_log_weights)
        weights[t.isnan(weights)] = 0
        weights /= weights.sum(dim=-1, keepdim=True)
        assert weights.shape == (repeats, num_samples)

        diff_post = t.zeros((repeats, num_samples), device=device)
        
        for r in range(repeats):
            diff_post[r] = diff[r, t.multinomial(weights[r], num_samples, replacement=True)]

        assert diff_post.shape == (repeats, num_samples)
        
        percentiles = t.stack([100 * alpha / 2, 100 * (1 - alpha / 2)]).transpose(0,1)
        assert percentiles.shape == (len(alpha), 2)

        quantiles = percentiles / 100

        # torch quantile = numpy percentile except it only accepts 1D quantile tensors
        interval = t.zeros((len(alpha), repeats, 2), device=device)
        # breakpoint()
        interval[:, :, 0] = t.quantile(diff_post, quantiles[:,0], dim=-1)
        interval[:, :, 1] = t.quantile(diff_post, quantiles[:,1], dim=-1)
        # assert interval.shape == (len(alpha), *data_A.shape[:-1], 2)
        
    if len(alpha) == 1:
        out = interval[0]
    else:
        out = interval 
        
    return out.cpu().numpy() if output_cpu else out


### Unpaired Bayes (Beta-Binomial model)
def bayes_unpaired_credible_interval(data_A, data_B, alpha, num_samples=10_000):
    '''
    data_A, data_B: np.ndarray
        Binary data, batch dimension is the last dimension
    alpha: float
        Significance level
    num_samples: int
        Number of importance samples

    Returns
    -------
    credible_interval: np.ndarray
        Lower and upper bounds of the credible interval
    '''
    if isinstance(alpha, float):
        alpha = np.array([alpha])
    else:
        alpha = np.array(alpha)

    if len(data_A.shape) == 1:
        data_A = data_A[None,:]

    if len(data_B.shape) == 1:
        data_B = data_B[None,:]

    repeats = data_A.shape[0]
    N = data_A.shape[-1]

    assert data_A.shape == data_B.shape
    assert len(data_A.shape) == 2

    # get posteriors for each model
    post_A = get_bayes_posterior(data_A)
    post_B = get_bayes_posterior(data_B)

    # sample from the posteriors
    theta_As_post = post_A.rvs(size=(num_samples, repeats)).transpose(1,0)
    theta_Bs_post = post_B.rvs(size=(num_samples, repeats)).transpose(1,0)
    assert theta_As_post.shape == theta_Bs_post.shape == (repeats, num_samples)

    diff_post = theta_As_post - theta_Bs_post

    percentiles = np.array([100 * alpha / 2, 100 * (1 - alpha / 2)]).transpose(1,0) # shape == (len(alpha), 2)S

    interval = np.percentile(diff_post, percentiles, axis=-1).transpose(0,2,1) # shape == (len(alpha), *data.shape[:-1], 2)
    
    if len(alpha) == 1:
        return interval[0]
    return interval


################################################################
## Testing

if __name__ == "__main__":
    from utils import sample_bernoulli

    repeats = 3
    data_A = sample_bernoulli(5, 0.5, shape=(repeats,))
    data_B = sample_bernoulli(5, 0.8, shape=(repeats,))

    print("data_A, data_B:")
    print(data_A, data_A.shape)
    print(data_B, data_B.shape)

    method_name_2_func = {
        'clt_unpaired': clt_unpaired_confidence_interval,
        'clt_paired': clt_paired_confidence_interval,
        'bootstrap_paired': bootstrap_paired_confidence_interval,
        'bayes_paired': importance_sampled_paired_credible_interval,
        'dirichlet_bayes_paired': dirichlet_paired_credible_interval,
    }

    print("Running tests with alphas=0.05")

    for method in ['clt_unpaired', 'clt_paired', 'bootstrap_paired', 'bayes_paired', 'dirichlet_bayes_paired']:
        print(f"Running {method} with alpha=0.05")
        results = method_name_2_func[method](data_A, data_B, 0.05)
        print(results, results.shape)

        print(f"Running {method} with alpha=[0.05,0.1]")
        results = method_name_2_func[method](data_A, data_B, [0.05,0.1])
        print(results, results.shape)

        print()

    ## check lambda_min and 
    print()
    print("Checking lambda_min and lambda_max:")
    theta_As = np.array([0.5, 0.6, 0.8])
    theta_Bs = np.array([0.5, 0.4, 0.9])

    print("theta_As, theta_Bs:")
    print(theta_As, theta_As.shape)
    print(theta_Bs, theta_Bs.shape)

    lambdas_min_ = lambda_min(theta_As, theta_Bs)
    lambdas_max_ = lambda_max(theta_As, theta_Bs)

    lambdas = np.random.uniform(low=lambdas_min_, high=lambdas_max_, size=(3,))
    print("lambdas_min, lambdas_max:")
    print(lambdas_min_, lambdas_min_.shape)
    print(lambdas_max_, lambdas_max_.shape)

    # the following should all be between 0 and 1
    print("Inequality checks (should all be between 0 and 1):")
    print("theta_Bs - lambdas*theta_As:")
    print(theta_Bs - lambdas*theta_As)

    print("theta_Bs + lambdas*(1-theta_As):")
    print(theta_Bs + lambdas*(1-theta_As))