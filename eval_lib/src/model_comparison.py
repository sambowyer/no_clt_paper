import numpy as np
import scipy.stats as stats

from .intervals import freq_confidence_interval, get_bayes_posterior

__all__ = ['ztest', 
           'freq_model_comparison', 
           'bayes_model_comparison']

################################################################
## Frequentist (Null Hypothesis Testing)
def ztest(data_A, data_B, alternative='greater'):
    '''
    Compare two Bernoulli distributions using a z-test.
    data_A, data_B: np.ndarray
        Binary data, batch dimension is the last dimension
    alternative: str
        The alternative hypothesis (default: 'greater')

    Returns
    -------
    pvalue: float
        The p-value of the test
    '''
    assert data_A.shape == data_B.shape
    
    interval = freq_confidence_interval(data_A - data_B, alpha=0.05)

    mean = interval.mean(-1)

    std_err = ((interval[:, 1] - interval[:, 0]) / 2) / 1.96

    gaussian = stats.norm(loc=0, scale=std_err)

    if alternative == 'greater':
        pvalue = 1-gaussian.cdf(mean)
    elif alternative == 'less':
        pvalue = gaussian.cdf(mean)
    elif alternative == 'two-sided':
        pvalue = 2 * gaussian.cdf(-np.abs(mean))

    # replace any nans with 0 if mean(data_A - data_B) == 0 else 1
    if np.isnan(pvalue).any():
        if alternative == 'greater':
            pvalue[np.isnan(pvalue)] = 1 - ((data_A - data_B).mean(-1) > 0).astype(int)[np.isnan(pvalue)]
        elif alternative == 'less':
            pvalue[np.isnan(pvalue)] = ((data_A - data_B).mean(-1) < 0).astype(int)[np.isnan(pvalue)]
        elif alternative == 'two-sided':
            pvalue[np.isnan(pvalue)] = ((data_A - data_B).mean(-1) != 0).astype(int)[np.isnan(pvalue)]

    return pvalue


def freq_model_comparison(data_A, data_B, method='fisher'):
    '''
    Compare two Bernoulli distributions using a frequentist approach: run a hypothesis test where
    the null hypothesis is that the two distributions are the same and the alternative hypothesis
    is that the parameter of A is greater than that of B.

    Parameters
    ----------
    data_A, data_B: np.ndarray
        Binary data, batch dimension is the last dimension
    method: str
        Method for comparing the distributions ('fisher', 'barnard', 'boschloo', 'welch', 'z-test')

    Returns
    -------
    pvalue: np.ndarray
        The p-value of the test per repeat
    '''

    data_shape = data_A.shape
    assert data_A.shape == data_B.shape

    if method == 'z-test':    
        pvalue = ztest(data_A, data_B, alternative='greater')
    elif len(data_shape) == 1:
        contigency_table = np.array([[data_A.sum(),               data_B.sum()],
                                     [data_A.size - data_A.sum(), data_B.size - data_B.sum()]])

        if method == 'fisher':
            pvalue = stats.fisher_exact(contigency_table, alternative='greater').pvalue
        elif method == 'barnard':
            pvalue = stats.barnard_exact(contigency_table, alternative='greater').pvalue
        elif method == 'boschloo':
            pvalue = stats.boschloo_exact(contigency_table, alternative='greater').pvalue
        elif method == 'welch':
            pvalue = stats.ttest_ind(data_A, data_B,  equal_var=False, alternative='greater').pvalue
    else:
        # batch dimension is at end, so we can iterate over all but the last dimension
        output_shape = data_shape[:-1]
        pvalue = np.zeros(output_shape)

        for idx in np.ndindex(output_shape):
            pvalue[idx] = freq_model_comparison(data_A[idx], data_B[idx], method)

    return pvalue


################################################################
## Bayesian Model Comparison
def bayes_model_comparison(data_A, data_B, prior=(1, 1), num_post_samples=10_000):
    '''
    Compare two Bernoulli distributions using a Bayesian approach.

    Parameters
    ----------
    data_A, data_B: np.ndarray
        Binary data, batch dimension is the last dimension
    prior: tuple
        Beta distribution parameters (default: (1, 1) i.e. uniform)
    num_post_samples: int
        Number of samples to draw from the posterior

    Returns
    -------
    prob_A_greater: float
        The probability that the parameter of A is greater than that of B
    '''
    assert data_A.shape == data_B.shape  # either (n,) or (repeats, n)
    data_shape = data_A.shape

    assert len(data_shape) <= 2
    if len(data_shape) == 1:
        data_shape = (1, 1) # (n,) -> (1, n)

    # get the posterior for each dataset (this 'uses up' the n dimension)
    post_A = get_bayes_posterior(data_A, prior)
    post_B = get_bayes_posterior(data_B, prior)

    # get a bunch of samples from the posterior
    samples_A = post_A.rvs(size=(num_post_samples, data_shape[0],))
    samples_B = post_B.rvs(size=(num_post_samples, data_shape[0],))

    assert samples_A.shape == samples_B.shape 
    assert samples_A.shape == (num_post_samples, data_shape[0],)

    # calculate the probability that A is greater than B (averaged over num_post_samples AND batch dimension (n,))
    prob_A_greater = np.mean(samples_A > samples_B, axis=0)
    assert prob_A_greater.shape == (data_shape[0],)

    return prob_A_greater
