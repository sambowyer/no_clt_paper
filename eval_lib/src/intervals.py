import numpy as np
import scipy.stats as stats
import statsmodels.stats.proportion as sm_proportion

__all__ = ['freq_confidence_interval', 
           'freq_confidence_interval_better', 

           'bootstrap_confidence_interval', 
           
           'get_bayes_posterior', 
           'bayes_credible_interval']

################################################################
## Frequentist (Confidence Intervals)

### CLT
def freq_confidence_interval(data, alpha):
    '''
    Compute the confidence interval for the parameter of a Bernoulli distribution. 

    Parameters
    ----------
    data: np.ndarray
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

    no_repeat_dim = len(data.shape) == 1
    if no_repeat_dim:
        data = data[None,]

    n = data.shape[-1]
    theta_hat = data.mean(axis=-1)     # shape == data.shape[:-1]
    z = stats.norm.ppf(1 - alpha / 2)  # shape == (len(alpha),)

    delta = z[:,None] * (np.sqrt(theta_hat * (1 - theta_hat) / n))[None,...] # shape == (len(alpha), *data.shape[:-1])
    
    output = np.array([theta_hat[None,:] - delta, theta_hat[None,:] + delta]).transpose(-2, -1, 0) # shape == (len(alpha), *data.shape[:-1], 2)

    if len(alpha) == 1:
        if no_repeat_dim:
            output = output[0]
        output = output[0]
    elif no_repeat_dim:
        output = output[:,0]
    return output

### Wilson & Clopper-Pearson ('beta')
def freq_confidence_interval_better(data, alpha, method='wilson'):
    '''
    Compute the confidence interval for the parameter of a Bernoulli distribution. 

    Parameters
    ----------
    data: np.ndarray
        Binary data, batch dimension is the last dimension
    alpha: float, or list of floats
        Significance level
    method: str
        Method for computing the confidence interval ('wilson' or 'beta')

    Returns
    -------
    confidence_interval: tuple
        Lower and upper bounds of the confidence interval
    '''
    if isinstance(alpha, float):
        alpha = np.array([alpha])
    else:
        alpha = np.array(alpha)

    n = data.shape[-1]
    n_s = data.sum(axis=-1).astype(int)

    output = np.zeros(alpha.shape + data.shape[:-1] + (2,))
    for i, a in enumerate(alpha):
        interval = sm_proportion.proportion_confint(count=n_s, nobs=n, alpha=a, method=method)
        output[i, ..., :] = np.column_stack(interval)
        
    if len(alpha) == 1:
        return output[0]
    return output


### Bootstrap
def bootstrap_confidence_interval(data, alpha, K=10_000, sample_length=None, max_parallel_repeats=1000):
    '''
    Compute the confidence interval for the parameter of a Bernoulli distribution using the bootstrap. 

    Parameters
    ----------
    data: np.ndarray
        Binary data, batch dimension is the last dimension
    alpha: float, or list of floats
        Significance level
    K: int
        Number of bootstrap samples

    Returns
    -------
    confidence_interval: tuple
        Lower and upper bounds of the confidence interval
    '''
    if isinstance(alpha, float):
        alpha = np.array([alpha])
    else:
        alpha = np.array(alpha)

    no_repeat_dim = len(data.shape) == 1
    if no_repeat_dim:
        data = data[None,:]
    num_repeats = data.shape[0]

    n = data.shape[-1]
    if not isinstance(sample_length, int):
        sample_length = n

    interval = np.zeros(alpha.shape + data.shape[:-1] + (2,))
    if num_repeats > max_parallel_repeats:
        for i in range(num_repeats // max_parallel_repeats):
            interval[:, i*max_parallel_repeats:(i+1)*max_parallel_repeats, :] = bootstrap_confidence_interval(data[ i*max_parallel_repeats:(i+1)*max_parallel_repeats, :], alpha, K, sample_length, max_parallel_repeats)

    else:
    
        indices = np.random.choice(n, (sample_length, K), replace=True)
        theta_bootstrapped = data[..., indices]
        theta_hats = theta_bootstrapped.mean(axis=-2)  # shape == data.shape[:-1] + (K,)

        percentiles = np.array([100 * alpha / 2, 100 * (1 - alpha / 2)]).transpose(1,0) # shape == (len(alpha), 2)S
        
        interval = np.percentile(theta_hats, percentiles, axis=-1).transpose(0,2,1) # shape == (len(alpha), *data.shape[:-1], 2)
        
    if len(alpha) == 1:
        if no_repeat_dim:
            interval = interval[0]
        interval = interval[0]
    elif no_repeat_dim:
        interval = interval[:,0]
    
    return interval


################################################################
## Bayes (Credible Intervals)

def get_bayes_posterior(data, prior=(1, 1)):
    '''
    Compute the posterior distribution of the parameter of a Bernoulli distribution. 

    Parameters
    ----------
    data: np.ndarray
        Binary data, batch dimension is the last dimension
    prior: tuple
        Beta distribution parameters (default: (1, 1) i.e. uniform)

    Returns
    -------
    posterior: scipy.stats.beta
        Posterior distribution
    '''
    n = data.shape[-1]
    a = data.sum(axis=-1)
    b = n - a
    return stats.beta(a + prior[0], b + prior[1])

def bayes_credible_interval(data, alpha, prior=(1, 1)):
    '''
    Compute the credible interval for the parameter of a Bernoulli distribution. 

    Parameters
    ----------
    data: np.ndarray
        Binary data, batch dimension is the last dimension
    alpha: float, or list of floats
        Significance levels
    prior: tuple
        Beta distribution parameters (default: (1, 1) i.e. uniform)

    Returns
    -------
    credible_interval: tuple
        Lower and upper bounds of the credible interval (with extra batch dimensions and potentially a significance level dimension prepended)
    '''
    if isinstance(alpha, float):
        alpha = np.array([alpha])
    else:
        alpha = np.array(alpha)
    
    output = np.zeros(alpha.shape + data.shape[:-1] + (2,))
    for i, a in enumerate(alpha):
        posterior = get_bayes_posterior(data, prior)
        interval = posterior.interval(1 - a)
        output[i, ..., :] = np.column_stack(interval)
    
    
    if len(alpha) == 1:
        return output[0]
    return output


################################################################
## Testing

if __name__ == "__main__":
    from utils import sample_bernoulli

    data = sample_bernoulli(10, 0.5)
    print("Data:")
    print(data, data.shape)

    print("Confidence Intervals with alpha=0.05")

    print(f"CLT: \n{freq_confidence_interval(data, 0.05)}")
    print(f"Wilson: \n{freq_confidence_interval_better(data, 0.05, method='wilson')}")
    print(f"Clopper-Pearson: \n{freq_confidence_interval_better(data, 0.05, method='beta')}")
    print(f"Bootstrap: \n{bootstrap_confidence_interval(data, 0.05)}")
    print(f"Bayes: \n{bayes_credible_interval(data, 0.05)}")

    print("Confidence Intervals with alpha=[0.05, 0.1]")

    print(f"CLT: \n{freq_confidence_interval(data, [0.05, 0.1])}")
    print(f"Wilson: \n{freq_confidence_interval_better(data, [0.05, 0.1], method='wilson')}")
    print(f"Clopper-Pearson: \n{freq_confidence_interval_better(data, [0.05, 0.1], method='beta')}")
    print(f"Bootstrap: {bootstrap_confidence_interval(data, [0.05, 0.1])}")
    print(f"Bayes: \n{bayes_credible_interval(data, [0.05, 0.1])}")

    print()

    data = sample_bernoulli(10, 0.5, (3,))
    print("Data:")
    print(data, data.shape)

    print("Confidence Intervals with alpha=[0.05, 0.1]")

    print(f"CLT: \n{freq_confidence_interval(data, [0.05, 0.1])}")
    print(f"Wilson: \n{freq_confidence_interval_better(data, [0.05, 0.1], method='wilson')}")
    print(f"Clopper-Pearson: \n{freq_confidence_interval_better(data, [0.05, 0.1], method='beta')}")
    print(f"Bootstrap: {bootstrap_confidence_interval(data, [0.05, 0.1])}")
    print(f"Bayes: \n{bayes_credible_interval(data, [0.05, 0.1])}")

