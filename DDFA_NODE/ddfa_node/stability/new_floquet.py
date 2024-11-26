import jax
import jax.numpy as jnp
import warnings
import numpy as np
import matplotlib.pyplot as plt
import traceback
from tqdm import tqdm
from .floquet import get_phased_signals
from functools import partial

@jax.jit
def weighted_multivariate_regression_with_bias(X, Y, W=None):
    """
    Fits a weighted multivariate regression model with an intercept and computes covariance errors.
    
    Parameters:
        X (jax.numpy.ndarray): Design matrix of shape (n_samples, n_features)
        Y (jax.numpy.ndarray): Output matrix of shape (n_samples, n_targets)
        W (jax.numpy.ndarray): Weight vector of shape (n_samples,)
    """
    # Add intercept
    X_aug = jnp.hstack([jnp.ones((X.shape[0], 1)), X])

    if W is None:
        W = jnp.diag(jnp.ones(X.shape[0]))
    
    # Convert W to diagonal matrix
    if W.ndim == 1:
        W = jnp.diag(W)
    
    # Matrix multiplications
    XTW = X_aug.T @ W
    XT_W_X = XTW @ X_aug
    XT_W_Y = XTW @ Y
    
    # Solve system
    coefficients = jnp.linalg.solve(XT_W_X, XT_W_Y)
    
    # Compute residuals
    residuals = Y - X_aug @ coefficients

    # Compute the covariance matrix of residuals
    covariance = jnp.linalg.pinv(XT_W_X)
    
    # Standard error of the coefficients (including bias)
    std_errors = jnp.sqrt(jnp.diag(covariance))
    
    return {
        "coefficients": coefficients,  # shape: (n_features + 1, n_targets)
        "covariance": covariance,      # shape: (n_features + 1, n_features + 1)
        "std_errors": std_errors,      # shape: (n_features + 1,)
        "residuals": residuals        # shape: (n_samples, n_targets)
    }

def create_random_subset_indices(key, n_samples, k, n_repeats):
    """
    Create random subset indices for splitting data into k groups.
    """
    subset_size = n_samples // k
    
    # Generate keys for each repeat
    keys = jax.random.split(key, n_repeats)
    
    @jax.vmap
    def get_permutations(key):
        # Generate random permutation of indices
        indices = jax.random.permutation(key, n_samples)
        # Reshape into k groups
        return indices[:subset_size * k].reshape(k, subset_size)
    
    return get_permutations(keys)

@jax.jit
def compute_subset_regression(X, Y, W=None, indices=None):
    """
    Compute regression on a subset of data specified by indices.
    
    Args:
        X: Full feature matrix (n_samples, n_features)
        Y: Full target matrix (n_samples, n_targets)
        W: Full weights vector (n_samples,)
        indices: Indices for subset selection (subset_size,)
    
    Returns:
        Regression results for the subset
    """
    X_subset = X[indices]
    Y_subset = Y[indices]
    if W is not None:
        W_subset = W[indices]
    else:
        W_subset = None
    
    return weighted_multivariate_regression_with_bias(X_subset, Y_subset, W_subset)

@jax.jit
def compute_multiple_subset_regressions(X, Y, W=None, all_indices=None):
    """
    Compute regressions for multiple subsets using vmap.
    
    Args:
        X: Full feature matrix
        Y: Full target matrix
        W: Full weights vector
        all_indices: Indices for all subsets (n_subsets, subset_size)
    
    Returns:
        Results for all subsets
    """
    return jax.vmap(lambda idx: compute_subset_regression(X, Y, W, idx))(all_indices)


def run_subset_analysis(X, Y, W=None, k=2, n_repeats=100, seed=0):
    """
    Run analysis for a single k value.
    """
    if W is None:
        W = jnp.diag(jnp.ones(X.shape[0]))
    n_samples = X.shape[0]
    key = jax.random.PRNGKey(seed)
    
    # Generate all subset indices
    all_subset_indices = create_random_subset_indices(key, n_samples, k, n_repeats)
    
    # Reshape indices to (n_repeats * k, subset_size)
    flat_indices = all_subset_indices.reshape(-1, n_samples // k)
    
    # Compute regressions for all subsets
    all_results = compute_multiple_subset_regressions(X, Y, W, flat_indices)
    
    # Reshape results back to (n_repeats, k)
    return {
        'coefficients': all_results['coefficients'].reshape(n_repeats, k, -1, Y.shape[1]),
        'std_errors': all_results['std_errors'].reshape(n_repeats, k, -1),
        'subset_indices': all_subset_indices
    }

# Now let's run the analysis for multiple k values
def run_multiple_k_analysis(X, Y, W=None, ks=[2, 3, 4, 5], n_repeats=100):
    """
    Run analysis for multiple k values.
    """
    results = []
    for i, k in enumerate(ks):
        result = run_subset_analysis(X, Y, W, k, n_repeats, i)
        results.append(result)
    return results


def get_eigenvalues(coefficients):
    # Get eigenvalues - shape will be (n_subsets, n_features)
    eigenvalues = jnp.linalg.eigvals(coefficients[:, 1:, :])
    # Sort by magnitude for each subset independently
    sorted_idx = jnp.argsort(jnp.abs(eigenvalues), axis=1, descending=True)
    # Use advanced indexing to sort each row
    batch_idx = jnp.arange(eigenvalues.shape[0])[:, None]
    sorted_eigenvalues = eigenvalues[batch_idx, sorted_idx]
    return sorted_eigenvalues

get_eigenvalues = jax.jit(get_eigenvalues, device=jax.devices('cpu')[0])

@partial(jax.jit, static_argnums=(1,))
def analyze_eigenvalue_scaling(data, max_groups, phase):
    """
    Analyze the scaling of eigenvalues of the coefficient matrices calculated for subsets.
    
    Args:
        data (jax.numpy.ndarray): Input data of shape (n_cycles, 101, n_features)
        max_groups (int): Maximum number of groups to split the data into
    
    Returns:
        dict: Contains x_values, y_values, weights, and regression results for eigenvalue scaling
    """
    n_cycles, _, n_features = data.shape
    results = []
    
    # Split data into non-overlapping subsets and compute regressions
    for n_groups in range(2, max_groups + 1):
        subset_size = n_cycles // n_groups
        key = jax.random.PRNGKey(0)  # Use a fixed key for reproducibility
        indices = create_random_subset_indices(key, n_cycles, n_groups, 1)[0]

        X = data[:-1, phase, :]
        Y = data[1:, phase, :]
        
        group_results = compute_multiple_subset_regressions(X, Y, W=None, all_indices=indices)
        jax.debug.print(str(group_results['coefficients'].shape))
        eigenvalues = get_eigenvalues(group_results['coefficients'])  # Exclude bias term, move to CPU for eigenvalue computation
        
        mean_eigenvalues = jnp.mean(eigenvalues, axis=0)
        std_eigenvalues = jnp.std(eigenvalues, axis=0)
        
        results.append({
            'n_groups': n_groups,
            'mean_eigenvalues': mean_eigenvalues,
            'std_eigenvalues': std_eigenvalues
        })
    
    # Prepare data for meta regression
    x_values = jnp.array([1.0 / (n_cycles // res['n_groups']) for res in results])
    y_values = jnp.array([res['mean_eigenvalues'] for res in results])
    weights = 1.0 / jnp.sqrt(jnp.sum(jnp.array([res['std_eigenvalues']**2 for res in results]), axis=1))
    print(weights.shape)
    print(y_values.shape)
    print(x_values.shape)
    
    # Perform weighted least squares regression for each eigenvalue
    regression_results = jax.vmap(lambda x, y, w: weighted_multivariate_regression_with_bias(
        x[:, None],  # Independent variable (1/N)
        y[:, None],  # Dependent variable (eigenvalues)
        w  # Weights
    ), in_axes=(None, 1, None))(x_values, y_values, weights)
    
    return {
        'x_values': x_values,
        'y_values': y_values,
        'weights': weights,
        'regression_results': regression_results
    }



