import numpy as np
import scipy.stats as stats

__all__ = [
    'COLOURS_COLORBREWER',
    'COLOURS_BASIC',
    'COLOURS',
    'METHOD_NAMES',

    'sample_bernoulli', 
    'generate_sample_collection', 
    'generate_sample_collection_paired_gaussian_prior',
    'generate_sample_collection_paired_dirichlet_prior',
    'generate_sample_collection_paired_model_prior',

    'lambda_min',
    'lambda_max',

    'sample_subtasks', 
    'flatten_subtask_data', 
    'generate_sample_subtask_collection', 
    
    'get_intervals', 
    'get_intervals_using_subtasks', 
    'get_intervals_using_pairing',
    'get_coverages_and_widths',
    'get_mean_abs_distance_from_coverage',
]

################################################################
## Useful Constants

COLOURS_COLORBREWER = {
    'bayes': '#377eb8',
    'freq': '#e41a1c',
    'clop': '#f781bf',
    'wils': '#984ea3',
    'boot': '#a65628',
    'boot_subtask': '#000000',

    'bayes_subtask': '#4daf4a',
    'clt_subtask': '#ff7f00',

    'bayes_paired': '#4daf4a',
    'bayes_paired_dirichlet': '#1f451e',
    'clt_paired': '#ff7f00'
}

COLOURS_BASIC = {
    'bayes': 'blue',
    'freq': 'darkorange',
    'clop': 'limegreen',
    'wils': 'darkgreen',
    'boot': 'brown',

    'boot_subtask': 'brown',
    'bayes_subtask': 'deepskyblue',
    # 'clt_subtask': 'orange',
    'clt_subtask': 'darkorange',
    # 'IS_subtask': 'purple',
    'IS_subtask': 'blue',

    'clt_paired': 'darkorange',
    'bayes_paired': 'blue',
    'bayes_paired_per_question': 'cyan',
    'bayes_paired_dirichlet': 'purple',
    'bayes_unpaired': 'blue'
}

COLOURS = COLOURS_BASIC
# COLOURS = COLOURS_COLORBREWER

METHOD_NAMES = {
    'bayes': 'Bayes',
    'freq': 'CLT',
    'wils': 'Wilson',
    'clop': 'Clopper-Pearson',
    'boot': 'Bootstrap',

    'boot_subtask': 'Clustered Bootstrap',
    'clt_subtask': 'Clustered CLT',
    'IS_subtask': 'Clustered Bayes (IS)',

    'clt_paired': 'Paired CLT',
    'bayes_paired': 'Paired Bayes (IS)',
    'bayes_paired_per_question': 'Old Bayes (IS)',
    'bayes_paired_dirichlet': 'Dirichlet Bayes',
    'bayes_unpaired': 'Unpaired Bayes'
}

################################################################
## Simple Bernoulli Experiments

def sample_bernoulli(n, theta, shape=None, seed=None):
    '''
    Sample n Bernoulli random variables with probability theta.

    Parameters
    ----------
    n: int
        Number of samples
    theta: float
        Probability of success
    shape: tuple
        Extra dimensions for the output

    Returns
    -------
    samples: np.ndarray
        shape == (shape[0], shape[1], ..., shape[-1], n)
    '''
    if seed is not None:
        np.random.seed(seed)

    if shape is None:
        shape = n
    else:
        shape = shape + (n,)
    return np.random.binomial(1, theta, shape)

### Independent Bernoulli Simulation
def generate_sample_collection(n_vals, theta_vals, repeats=1, seed=None):
    '''
    Generate a collection of samples from Bernoulli distributions with different parameters (theta) and sample sizes (n).

    Parameters
    ----------
    n_vals: list
        Sample sizes
    theta_vals: list
        Probabilities
    repeats: int
        Number of samples for each (n, theta) pair
    seed: int
        Random seed

    Returns
    -------
    samples: dict
        Dictionary with keys n and values numpy arrays of shape (len(theta_vals), repeats, n)
    '''
    if seed is not None:
        np.random.seed(seed)

    samples = {n: np.array([sample_bernoulli(n, theta, (repeats,)) for theta in theta_vals]) for n in n_vals}

    for n in n_vals:
        assert samples[n].shape == (len(theta_vals), repeats, n)

    return samples


### Paired 2D Gaussian Simulation
def generate_sample_collection_paired_gaussian_prior(n_vals, theta_vals_A, theta_vals_B, repeats=1, seed=None, rho_min=-1, rho_max=1):
    '''
    Generate a collection of samples from Bernoulli distributions with different parameters (theta_vals_A and theta_vals_B)
    and sample sizes (n) according to a 2D Gaussian prior.

    1. Figure out the possible correlation values, rho, between theta_A and theta_B and set covariance matrices = (1, rho; rho, 1)
    2. Repeat the following (3-5) for each n value:
    3. Sample a 2D Gaussian prior for each (n, theta_A, theta_B) setting (repeat no. of times) using the covariance matrices
    4. Find thresholds for the 2D Gaussian prior that correspond to the desired theta_A and theta_B values
    5. Sample Bernoulli data for each (n, theta_A, theta_B) setting (repeat no. of times) using the thresholds: success if sample is above threshold

    Parameters
    ----------
    n_vals: list
        Sample sizes
    theta_vals_A, theta_vals_B: list
        Probabilities of success for model A and model B
    repeats: int
        Number of samples for each (n, theta) pair
    seed: int
        Random seed
    rho_min, rho_max: float
        Minimum and maximum possible correlation values
        If None, then we sample rho from 2*Beta(4,2)-1

    Returns
    -------
    samples_A, samples_B: dict
        Dictionaries with keys n and values numpy arrays of shape (len(theta_vals), repeats, n)
    '''
    if seed is not None:
        np.random.seed(seed)

    # make sure both theta_vals_A and theta_vals_B are one-dimensional
    assert len(theta_vals_A.shape) == len(theta_vals_B.shape) == 1

    # either both are the same length or one of them is of length 1
    assert len(theta_vals_A) == len(theta_vals_B) or len(theta_vals_A) == 1 or len(theta_vals_B) == 1

    # if one of them is of length 1, repeat it to match the length of the other
    if len(theta_vals_A) == 1:
        theta_vals_A = np.repeat(theta_vals_A, len(theta_vals_B))
    if len(theta_vals_B) == 1:
        theta_vals_B = np.repeat(theta_vals_B, len(theta_vals_A))

    num_thetas = len(theta_vals_A)

    # calculate means mu_A and mu_B for each (n, theta_A, theta_B, num_repeat) setting
    # mu_A = Phi^{-1}(theta_A) and mu_B = Phi^{-1}(theta_B) where Phi is the CDF of the standard normal distribution
    mu_A = stats.norm.ppf(theta_vals_A, loc=0, scale=1)
    mu_B = stats.norm.ppf(theta_vals_B, loc=0, scale=1)
    assert mu_A.shape == mu_B.shape == (num_thetas,)

    # stack into (num_thetas, 2) array
    mu = np.stack([mu_A, mu_B], axis=1)
    assert mu.shape == (num_thetas, 2)

    # sample rho for each (n, theta_A, theta_B, num_repeat) setting
    if rho_min is None and rho_max is None:
        rho = 2*stats.beta.rvs(4,2, size=(len(n_vals), num_thetas, repeats)) - 1
    else:
        rho = np.random.uniform(low=rho_min, high=rho_max, size=(len(n_vals), num_thetas, repeats))
    assert rho.shape == (len(n_vals), num_thetas, repeats)

    # calculate the covariance matrix for each (n, theta_A, theta_B, num_repeat) setting
    cov = np.array([[np.ones((len(n_vals), num_thetas, repeats)), rho], [rho, np.ones((len(n_vals), num_thetas, repeats))]])
    assert cov.shape == (2, 2, len(n_vals), num_thetas, repeats)

    # move the 2x2 dimensions to the end
    cov = cov.transpose(2,3,4,0,1)
    assert cov.shape == (len(n_vals), num_thetas, repeats, 2, 2)

    # get cholesky decomposition of covariance matrix
    # (we'll use reparameterization trick to sample from N(0,cov=LL^T) with all our extra dimensions,
    # rather than using np.random.multivariate_normal which doesn't support extra dimensions)
    L = np.linalg.cholesky(cov)
    assert L.shape == (len(n_vals), num_thetas,repeats, 2, 2)

    L_transpose = L.transpose(0,1,2,4,3)
    
    samples_A = {}
    samples_B = {}

    for n_idx, n in enumerate(n_vals):
        # sample from N(0,1) and multiply by L to get samples from N(0,cov)
        standard_gaussian_samples = np.random.normal(size=(num_thetas,repeats, 2, n))
        assert standard_gaussian_samples.shape == (num_thetas, repeats, 2, n)

        gaussian_samples = np.einsum('trij,trin->trjn', L_transpose[n_idx], standard_gaussian_samples) 
        assert gaussian_samples.shape == (num_thetas, repeats, 2, n)

        gaussian_samples += mu[:, None, :, None]

        data_A = (gaussian_samples[...,0,:] > 0).astype(int)
        data_B = (gaussian_samples[...,1,:] > 0).astype(int)

        assert data_A.shape == (len(theta_vals_A), repeats, n)
        assert data_B.shape == (len(theta_vals_B), repeats, n)

        samples_A[n] = data_A
        samples_B[n] = data_B

    return samples_A, samples_B, rho


### Paired Dirichlet-Multinomial Simulation
def dirichlet_multinomial_sample(alpha, n, **kwargs):
    p = stats.dirichlet.rvs(alpha=alpha, **kwargs)

    totals = multinomial_sample(p, n, **kwargs)

    return totals

def multinomial_sample(p, n, **kwargs):
    # sample multinomial manually because we have extra dimensions that scipy.stats doesn't like
    samples = np.zeros((*p.shape[:-1], n), dtype=int)
    noise = np.random.uniform(size=(*p.shape[:-1], n))
    p_cumsum = p.cumsum(axis=-1)
    for i in range(n):
        samples[...,i] = (noise[...,i,None] < p_cumsum).argmax(axis=-1)
    
    totals = np.array([np.sum(samples == i, axis=-1) for i in range(p.shape[-1])])
    assert totals.shape == (p.shape[-1],) + p.shape[:-1]

    totals = totals.transpose(*range(1,len(p.shape)), 0)
    assert totals.shape == p.shape

    return totals

def generate_sample_collection_paired_dirichlet_prior(n_vals, theta_vals_A, theta_vals_B, repeats=1, seed=None):
    '''
    Generate a collection of samples from Bernoulli distributions with different parameters (theta_vals_A and theta_vals_B)
    and sample sizes (n) according to a dirichlet(1,1,1,1) prior distribution.

    Parameters
    ----------
    n_vals: list
        Sample sizes
    theta_vals_A, theta_vals_B: list
        Probabilities of success for model A and model B
    repeats: int
        Number of samples for each (n, theta) pair
    seed: int
        Random seed

    Returns
    -------
    samples_A, samples_B: dict
        Dictionaries with keys n and values numpy arrays of shape (len(theta_vals), repeats, n)
    '''
    if seed is not None:
        np.random.seed(seed)

    # make sure both theta_vals_A and theta_vals_B are one-dimensional
    assert len(theta_vals_A.shape) == len(theta_vals_B.shape) == 1

    # either both are the same length or one of them is of length 1
    assert len(theta_vals_A) == len(theta_vals_B) or len(theta_vals_A) == 1 or len(theta_vals_B) == 1

    # if one of them is of length 1, repeat it to match the length of the other
    if len(theta_vals_A) == 1:
        theta_vals_A = np.repeat(theta_vals_A, len(theta_vals_B))
    if len(theta_vals_B) == 1:
        theta_vals_B = np.repeat(theta_vals_B, len(theta_vals_A))

    num_thetas = len(theta_vals_A)

    # sample from a dirichlet(1,1,1,1) distribution to get a contingency table for each (theta_A, theta_B, repeat) setting
    dir_samples = stats.dirichlet.rvs([1,1,1,1], size=(num_thetas)) 
    assert dir_samples.shape == (num_thetas, 4)

    theta_vals_A = dir_samples[:,0] + dir_samples[:,1]
    theta_vals_B = dir_samples[:,0] + dir_samples[:,2]

    assert theta_vals_A.shape == theta_vals_B.shape == (num_thetas,)

    # repeat dir_samples across repeats
    dir_samples = np.repeat(dir_samples[:, None, :], repeats, axis=1)
    assert dir_samples.shape == (num_thetas, repeats, 4)
    
    samples_A, samples_B = {}, {}
    for n in n_vals:
        # generate a contingency table
        # contingency_table = dirichlet_multinomial_sample([1, 1, 1, 1], n, size=(num_thetas,repeats))
        contingency_table = multinomial_sample(dir_samples, n)
        assert contingency_table.shape == (num_thetas, repeats, 4)
        # theta_vals_A = (contingency_table[:,0] + contingency_table[:,1])/n
        # theta_vals_B = (contingency_table[:,0] + contingency_table[:,2])/n

        # manually generate samples to match the contingency table
        samples_A[n] = np.zeros((len(theta_vals_A), repeats, n))
        samples_B[n] = np.zeros((len(theta_vals_B), repeats, n))

        # add in contingency table values
        for i in range(len(theta_vals_A)):
            for j in range(repeats):
                # success for both A and B, then success for A but not B, then success for B but not A
                S_AB = contingency_table[i,j,0]
                S_AnotB = contingency_table[i,j,1]
                S_BnotA = contingency_table[i,j,2]

                samples_A[n][i, j, :S_AB] = 1
                samples_B[n][i, j, :S_AB] = 1

                samples_A[n][i, j, S_AB : (S_AB+S_AnotB)] = 1
                samples_B[n][i, j, (S_AB+S_AnotB) : (S_AB + S_AnotB + S_BnotA)] = 1

    return samples_A, samples_B, theta_vals_A, theta_vals_B

### Paired Model (per-question) Simulation
def lambda_min(theta_A, theta_B):
    if isinstance(theta_A, np.ndarray):
        assert isinstance(theta_B, np.ndarray)
        assert theta_A.shape == theta_B.shape
    
    output = np.max(np.array([-theta_B/(1-theta_A), -(1-theta_B)/theta_A]), axis=0)

    if isinstance(theta_A, np.ndarray):
        assert output.shape == theta_A.shape

    return output

def lambda_max(theta_A, theta_B):
    if isinstance(theta_A, np.ndarray):
        assert isinstance(theta_B, np.ndarray)
        assert theta_A.shape == theta_B.shape

    output = np.min(np.array([theta_B/theta_A, (1-theta_B)/(1-theta_A)]), axis=0)
    # output = np.min(np.array([-theta_B/theta_A, -(1-theta_B)/(1-theta_A)]), axis=0)

    if isinstance(theta_A, np.ndarray):
        assert output.shape == theta_A.shape

    return output

def generate_sample_collection_paired_model_prior(n_vals, theta_vals_A, theta_vals_B, repeats=1, seed=None):
    '''
    Generate a collection of samples from Bernoulli distributions with different parameters (theta_vals_A and theta_vals_B)
    and sample sizes (n) according to the paired model (where we model B's performance as a function of A's performance per-question).

    Parameters
    ----------
    n_vals: list
        Sample sizes
    theta_vals_A, theta_vals_B: list
        Probabilities of success for model A and model B
    repeats: int
        Number of samples for each (n, theta) pair
    seed: int
        Random seed

    Returns
    -------
    samples_A, samples_B: dict
        Dictionaries with keys n and values numpy arrays of shape (len(theta_vals), repeats, n)
    '''
    if seed is not None:
        np.random.seed(seed)

    # make sure both theta_vals_A and theta_vals_B are one-dimensional
    assert len(theta_vals_A.shape) == len(theta_vals_B.shape) == 1

    # either both are the same length or one of them is of length 1
    assert len(theta_vals_A) == len(theta_vals_B) or len(theta_vals_A) == 1 or len(theta_vals_B) == 1

    # if one of them is of length 1, repeat it to match the length of the other
    if len(theta_vals_A) == 1:
        theta_vals_A = np.repeat(theta_vals_A, len(theta_vals_B))
    if len(theta_vals_B) == 1:
        theta_vals_B = np.repeat(theta_vals_B, len(theta_vals_A))

    # generate samples for A
    samples_A = generate_sample_collection(n_vals, theta_vals_A, repeats=repeats, seed=seed)
    
    # now generate samples for B using the paired model
    if seed is not None:
        np.random.seed(seed+1)

    lambdas_min = lambda_min(theta_vals_A, theta_vals_B)
    lambdas_max = lambda_max(theta_vals_A, theta_vals_B)
    lambdas = np.random.uniform(low=lambdas_min, high=lambdas_max, size=(repeats,len(theta_vals_A))).transpose()
    assert lambdas.shape == (len(theta_vals_A), repeats)

    assert (np.abs(theta_vals_B[:,None] - lambdas*theta_vals_A[:,None]) <= 1).all()
    assert (np.abs(theta_vals_B[:,None] + lambdas*(1-theta_vals_A[:,None])) <= 1).all()

    samples_B = {}

    for n in n_vals:
        theta_vals_B_given_A = theta_vals_B[:,None,None] + lambdas[:,:,None] * (samples_A[n] - theta_vals_A[:,None,None])
        assert theta_vals_B_given_A.shape == (len(theta_vals_B), repeats, n)

        samples_B[n] = np.random.binomial(1, theta_vals_B_given_A)

        assert samples_B[n].shape == (len(theta_vals_B), repeats, n)

    return samples_A, samples_B

################################################################
## Subtask Bernoulli Experiments
def sample_subtasks(n_per_task, theta, shape=None, seed=None):
    '''
    FOR EACH REPEAT: sample data according to the subtask model
              d ~ Gamma(1,1)
        theta_t ~ Beta(d*theta, d*(1-theta))
          x_t_i ~ Bernoulli(theta_t)

    Parameters
    ----------
    n_per_task: numpy.ndarray
        Length T.
    theta: float
        Underlying across-task model performance
    shape: tuple
        extra sampling dimensions

    Returns 
    -------
    A list of length T containing numpy arrays each of shape (n_per_task[t],) if shape is None, otherwise
    each numpy array has shape (shape[0], shape[1], ..., shape[-1], n_per_task[t]).
    '''
    if seed is not None:
        np.random.seed(seed)

    assert len(n_per_task.shape) == 1
    T = n_per_task.shape[0]
    
    d = np.random.gamma(1,1, size=(*shape,1) if shape is not None else 1)
    theta_ts = np.random.beta(d*theta, d*(1-theta), size=(*shape,T) if shape is not None else T)

    data = []
    for t in range(T):
        data.append(sample_bernoulli(n_per_task[t], theta_ts[:,t,None] if shape is not None else theta_ts[t], shape))

    return data

def flatten_subtask_data(data):
    '''
    Flatten data from the subtask model into a single array.

    Parameters
    ----------
    data: list
        List of numpy arrays of shape (n_per_task[t],) or (shape[0], shape[1], ..., shape[-1], n_per_task[t])

    Returns
    -------
    A numpy array of shape (n_total,) or (shape[0], shape[1], ..., shape[-1], n_total)
    '''
    return np.concatenate(data, axis=1) if len(data[0].shape) > 1 else np.concatenate(data)

def generate_sample_subtask_collection(n_vals, theta_vals, n_per_tasks, repeats=1, seed=None):
    '''
    Generate a collection of samples from the subtask model with different parameters (theta) and sample sizes (n).

    NOTE: This requires only a single possible set of n_per_tasks for each n in n_vals.

    Parameters
    ----------
    n_vals: list
        Sample sizes (n_total)
    theta_vals: list
        Probabilities
    n_per_tasks: list
        List of len(n_vals) np.ndarrays of shape (T,) where T is the number of tasks 
        and n_per_tasks[n_idx][t] is the number of samples for task t
    repeats: int
        Number of samples for each (n, theta) pair
    seed: int
        Random seed

    Returns
    -------
    subtask_samples: dict
        Dictionary with 
            keys: n 
            values: lists of T np.ndarrays of shape (len(theta_vals), repeats, n)
    subtask_samples_flat: dict
        Dictionary with 
            keys: n
            values: np.ndarrays of shape (repeats, n)
    '''
    assert len(n_per_tasks) == len(n_vals)
    assert all(n_per_tasks[i].sum() == n_vals[i] for i in range(len(n_vals)))
    assert all(len(n_per_tasks[i].shape) == 1 for i in range(len(n_vals)))

    if seed is not None:
        np.random.seed(seed)

    subtask_samples = {n_vals[i]: [sample_subtasks(n_per_tasks[i], theta, shape=(repeats,)) for theta in theta_vals] for i in range(len(n_vals))}
    # subtask_samples = {n_vals[i]: np.array([sample_subtasks(n_per_tasks[i], theta, shape=(repeats,)) for theta in theta_vals]) for i in range(len(n_vals))}
    for n_idx, n in enumerate(n_vals):
        T = n_per_tasks[n_idx].shape[0]
        for X_ts in subtask_samples[n]:
            for t in range(T):
                assert X_ts[t].shape == (repeats, n_per_tasks[n_idx][t])

    subtask_samples_flat = {n: [flatten_subtask_data(X_ts) for X_ts in subtask_samples[n]] for n in n_vals}
    for n_idx, n in enumerate(n_vals):
        for X_ts in subtask_samples_flat[n]:
            assert X_ts.shape == (repeats, n)

    return subtask_samples, subtask_samples_flat


################################################################
## Obtaining and Analyzing Collections of Intervals

def get_intervals(n_vals, theta_vals, samples, alphas, method_fn, **kwargs):
    '''
    Compute confidence intervals for the parameter of a Bernoulli distribution.

    Parameters
    ----------
    samples: dict[int, np.ndarray]
        Dictionary with keys n and values numpy arrays of shape (len(theta_vals), repeats, n)
    alphas: list
        Significance levels
    method_fn: callable
        Function that computes confidence intervals
    kwargs: dict
        Additional arguments for method_fn

    Returns
    -------
    intervals: np.ndarray
        numpy array of shape (repeats, len(n_vals), len(theta_vals), len(alphas), 2)
    '''
    repeats = samples[n_vals[0]][0].shape[0]   
    assert all(all([samples[n][theta_idx].shape[0] == repeats for theta_idx in range(len(theta_vals))]) for n in n_vals)


    intervals = np.array([[method_fn(samples[n][theta_idx], alpha=alphas, **kwargs) for theta_idx in range(len(theta_vals))] for n in n_vals])
    if intervals.shape != (len(n_vals), len(theta_vals), len(alphas), repeats, 2):
        # this can happen if alphas is a single value, in which case we need to expand it in the correct dimension
        intervals = np.expand_dims(intervals, axis=2)

    assert intervals.shape == (len(n_vals), len(theta_vals), len(alphas), repeats, 2)

    # move the repeats dimension to the beginning and swap the theta and alpha dimensions 
    #    (we want n_values to be second and theta to be second-last)
    intervals = intervals.transpose(3, 0, 2, 1, 4)

    assert intervals.shape == (repeats, len(n_vals), len(alphas), len(theta_vals), 2)

    return intervals

def get_intervals_using_subtasks(n_vals, theta_vals, samples, alphas, method_fn, **kwargs):
    '''
    Compute confidence intervals for the parameter of a Bernoulli distribution at the base of the subtask model.

    Parameters
    ----------
    samples: dict[int, np.ndarray]
        Dictionary with 
            keys: n
            values: lists of T numpy arrays of shape (len(theta_vals), repeats, n_per_task[t])
    alphas: list
        Significance levels
    method_fn: callable
        Function that computes confidence intervals
    kwargs: dict
        Additional arguments for method_fn

    Returns
    -------
    intervals: np.ndarray
        numpy array of shape (repeats, len(n_vals), len(theta_vals), len(alphas), 2)
    '''
    repeats = samples[n_vals[0]][0][0].shape[0]   

    # assert all(all(subtask_samples.shape[1] == repeats for subtask_samples in samples[n]) for n in n_vals)
    # repeats = samples[n_vals[0]][0].shape[0]   
    # assert all(all([samples[n][theta_idx].shape[0] == repeats for theta_idx in range(len(theta_vals))]) for n in n_vals)


    # for n in n_vals:
    #     assert all(subtask_samples.shape[:2] == (len(theta_vals), repeats) for subtask_samples in samples[n])
    #     assert all(len(subtask_samples.shape) == 3 for subtask_samples in samples[n])

    intervals = np.array([[method_fn(samples[n][theta_idx], alpha=alphas, **kwargs) for theta_idx in range(len(theta_vals))] for n in n_vals])
    
    if intervals.shape != (len(n_vals), len(theta_vals), len(alphas), repeats, 2):
        # this can happen if alphas is a single value, in which case we need to expand it in the correct dimension
        intervals = np.expand_dims(intervals, axis=2)

    assert intervals.shape == (len(n_vals), len(theta_vals), len(alphas), repeats, 2)

    # move the repeats dimension to the beginning and swap the theta and alpha dimensions 
    #    (we want n_values to be second and theta to be second-last)
    intervals = intervals.transpose(3, 0, 2, 1, 4)

    assert intervals.shape == (repeats, len(n_vals), len(alphas), len(theta_vals), 2)

    return intervals

def get_intervals_using_pairing(n_vals, num_theta_vals, samples_A, samples_B, alphas, method_fn, **kwargs):
    '''
    Compute confidence intervals for the difference between parameters of two Bernoulli distributions
    using paired samples.

    Parameters
    ----------
    n_vals: list
        Sample sizes
    num_theta_vals: int
        Number of theta values
    samples_A, samples_B: dict[int, np.ndarray]
        Dictionary with keys n and values numpy arrays of shape (num_theta_vals, repeats, n)
    alphas: list
        Significance levels
    method_fn: callable
        Function that computes confidence intervals
    kwargs: dict
        Additional arguments for method_fn

    Returns
    -------
    intervals: np.ndarray
        numpy array of shape (repeats, len(n_vals), num_theta_vals, len(alphas), 2)
    '''

    repeats = samples_A[n_vals[0]][0].shape[0]   
    assert all(all([samples_A[n][theta_idx].shape[0] == repeats for theta_idx in range(num_theta_vals)]) for n in n_vals)
    assert all(all([samples_B[n][theta_idx].shape[0] == repeats for theta_idx in range(num_theta_vals)]) for n in n_vals)

    intervals = np.array([[method_fn(samples_A[n][theta_idx], samples_B[n][theta_idx], alpha=alphas, **kwargs) for theta_idx in range(num_theta_vals)] for n in n_vals])

    if intervals.shape != (len(n_vals), num_theta_vals, len(alphas), repeats, 2):
        # this can happen if alphas is a single value, in which case we need to expand it in the correct dimension
        intervals = np.expand_dims(intervals, axis=2)

    assert intervals.shape == (len(n_vals), num_theta_vals, len(alphas), repeats, 2)

    # move the repeats dimension to the beginning and swap the theta and alpha dimensions 
    #    (we want n_values to be second and theta to be second-last)
    intervals = intervals.transpose(3, 0, 2, 1, 4)

    assert intervals.shape == (repeats, len(n_vals), len(alphas), num_theta_vals, 2)

    return intervals

def get_coverages_and_widths(intervals, n_vals, theta_vals, alphas, repeats):
    '''
    Compute coverages and widths of confidence intervals (averaging over repeats and theta values).

    Parameters
    ----------
    intervals: dict[str, np.ndarray]
        Dictionary with keys method and values numpy arrays of shape (repeats, len(n_vals), len(alphas), len(theta_vals), 2)
    n_vals: list
        Sample sizes
    theta_vals: list
        Probabilities
    alphas: list
        Significance levels
    repeats: int
        Number of samples for each (n, theta) pair

    Returns
    -------
    coverages: dict[str, np.ndarray]
        Dictionary with 
            keys: method name (str) 
            values: numpy arrays of shape (len(n_vals), len(alphas))
    '''
    coverages = {}
    widths = {}

    for x in intervals.keys():
        assert intervals[x].shape == (repeats, len(n_vals), len(alphas), len(theta_vals), 2)

        coverages[x] = np.mean((intervals[x][:, :, :, :, 0] <= theta_vals) & (theta_vals <= intervals[x][:, :, :, :, 1]), axis=(0, -1)) # average over repeats and p_vals
        widths[x] = np.mean(intervals[x][:, :, :, :, 1] - intervals[x][:, :, :, :, 0], axis=(0, -1)) # average over repeats and p_vals

        assert coverages[x].shape == (len(n_vals), len(alphas))
        assert widths[x].shape == (len(n_vals), len(alphas))

    return coverages, widths


def get_mean_abs_distance_from_coverage(coverages, n_vals, alphas):
    '''
    Compute the mean absolute distance from the coverage level for each method.

    Parameters
    ----------
    coverages: dict[str, np.ndarray]
        Dictionary with 
            keys: method name (str) 
            values: numpy arrays of shape (len(n_vals), len(alphas))
    n_vals: list
        Sample sizes
    alphas: list
        Significance levels

    Returns
    -------
    distances: dict[str, np.ndarray]
        Dictionary with 
            keys: method name (str) 
            values: numpy arrays of shape (len(n_vals), len(alphas))
    '''
    distances = {}
    for x in coverages.keys():
        distances[x] = np.abs(coverages[x] - (1-np.array(alphas)[None, :]))
        assert distances[x].shape == (len(n_vals), len(alphas))
        distances[x] = np.mean(distances[x], axis=1)
    return distances


################################################################
## Testing

if __name__ == '__main__':
    print("Sampling simple Bernoulli data")
    data = sample_bernoulli(10, 0.5)
    print(data)
    print(data.shape)

    data = sample_bernoulli(10, 0.5, (3,))
    print(data)
    print(data.shape)
    
    print()

    print("Generating a collection of simple Bernoulli samples")
    samples = generate_sample_collection([10, 20], [0.5, 0.6], repeats=3)
    print(samples)
    print(samples[10].shape)
    print(samples[20].shape)

    print()

    print("Sampling subtask Bernoulli data")
    data = sample_subtasks(np.array([5, 4]), 0.5)
    print(data)
    print(data[0].shape)
    print(data[1].shape)
    print(f"Flattened: {flatten_subtask_data(data)} with shape {flatten_subtask_data(data).shape}")


    data = sample_subtasks(np.array([5, 4]), 0.5, (3,))
    print(data)
    print(f"Flattened: {flatten_subtask_data(data)} with shape {flatten_subtask_data(data).shape}")

    print()

    print("Generating a collection of subtask Bernoulli samples")
    n_per_tasks = [np.array([5, 4]), np.array([4, 6])]
    samples, samples_flat = generate_sample_subtask_collection([9, 10], [0.5, 0.6], n_per_tasks, repeats=3)
    print(samples)
    print(f"Shapes: {[[x_t.shape for x_t in x] for x in samples[9]]}, {[[x_t.shape for x_t in x] for x in samples[10]]}")


    print("Flattened:")
    print(samples_flat)
    print(f"Shapes: {[x.shape for x in samples_flat[9]]}, {[x.shape for x in samples_flat[10]]}")
    print()

