import jax
import jax.numpy as jnp

@jax.jit
def weighted_multivariate_regression_with_bias(X, Y, W):
    """
    Fits a weighted multivariate regression model with an intercept and computes covariance errors.
    
    Parameters:
        X (jax.numpy.ndarray): Design matrix of shape (n_samples, n_features)
        Y (jax.numpy.ndarray): Output matrix of shape (n_samples, n_targets)
        W (jax.numpy.ndarray): Weight vector of shape (n_samples,)
    """
    # Add a column of ones to X for the intercept term
    X_aug = jnp.hstack([jnp.ones((X.shape[0], 1)), X])
    
    # If W is a vector, convert it to a diagonal matrix
    if W.ndim == 1:
        W = jnp.diag(W)
    
    # Compute the weighted regression coefficients
    XT_W_X = X_aug.T @ W @ X_aug
    XT_W_Y = X_aug.T @ W @ Y
    
    # Solve for coefficients
    coefficients = jnp.linalg.solve(XT_W_X, XT_W_Y)
    
    # Compute residuals
    residuals = Y - X_aug @ coefficients

    # Compute the covariance matrix of residuals
    n_samples = X_aug.shape[0]
    n_features = X_aug.shape[1]  # includes bias term
    
    # Modified covariance calculation
    covariance = jnp.linalg.inv(XT_W_X)
    
    # Standard error of the coefficients (including bias)
    std_errors = jnp.sqrt(jnp.diag(covariance))
    
    return {
        "coefficients": coefficients,  # shape: (n_features + 1, n_targets)
        "covariance": covariance,      # shape: (n_features + 1, n_features + 1)
        "std_errors": std_errors,      # shape: (n_features + 1,)
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
def compute_subset_regression(X, Y, W, indices):
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
    W_subset = W[indices]
    
    return weighted_multivariate_regression_with_bias(X_subset, Y_subset, W_subset)

@jax.jit
def compute_multiple_subset_regressions(X, Y, W, all_indices):
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


def run_subset_analysis(X, Y, W, k, n_repeats, seed):
    """
    Run analysis for a single k value.
    """
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
def run_multiple_k_analysis(X, Y, W, ks, n_repeats=100):
    """
    Run analysis for multiple k values.
    """
    results = []
    for i, k in enumerate(ks):
        result = run_subset_analysis(X, Y, W, k, n_repeats, i)
        results.append(result)
    return results



def analyze_eigenvalue_scaling(results, ks):
    """
    Analyze relationship between 1/n and eigenvalues using WLS regression.
    
    Args:
        results: List of results from run_multiple_k_analysis
        ks: List of k values used in the analysis
        
    Returns:
        Dictionary containing regression results for each eigenvalue
    """
    # Calculate sample sizes and x values (1/n)
    n_samples = results[0]['coefficients'].shape[-2] * ks[0]  # Total samples
    ns = jnp.array([n_samples // k for k in ks])
    x = 1.0 / ns
    
    # Get dimensions
    n_repeats = results[0]['coefficients'].shape[0]
    n_k = len(ks)
    dim = results[0]['coefficients'].shape[-1]
    
    # For each k, compute eigenvalues for each repeat/subset
    def get_eigenvalues(coeffs):
        # Remove bias term and reshape to square matrix
        matrix = coeffs[1:].reshape(dim, dim)
        return jnp.linalg.eigvals(matrix)
    
    # Vectorize over repeats and subsets
    batch_eigenvalues = jax.vmap(jax.vmap(get_eigenvalues))
    
    # Process each k value
    all_eigenvalues = []
    all_variances = []
    
    for k_idx, result in enumerate(results):
        # Get coefficients for this k (shape: n_repeats, k, n_features+1, dim)
        coeffs = result['coefficients']
        
        # Compute eigenvalues (shape: n_repeats, k, dim)
        eigs = batch_eigenvalues(coeffs)
        
        # Compute mean across repeats and subsets
        mean_eigs = jnp.mean(eigs, axis=(0,1))
        
        # Compute variance across all repeats and subsets
        var_eigs = jnp.var(eigs, axis=(0,1)) / (n_repeats * ks[k_idx])
        
        all_eigenvalues.append(mean_eigs)
        all_variances.append(var_eigs)
    
    # Stack results
    y = jnp.stack(all_eigenvalues)  # shape: (n_k, dim)
    weights = 1.0 / jnp.stack(all_variances)  # shape: (n_k, dim)
    
    # Perform WLS regression for each eigenvalue
    regression_results = []
    for i in range(dim):
        result = weighted_multivariate_regression_with_bias(
            x[:, None],  # X values (1/n)
            y[:, i:i+1], # Y values (eigenvalue estimates)
            weights[:, i]  # Weights from inverse variance
        )
        regression_results.append(result)
        
    return {
        'x_values': x,
        'y_values': y,
        'weights': weights,
        'regression_results': regression_results
    }
