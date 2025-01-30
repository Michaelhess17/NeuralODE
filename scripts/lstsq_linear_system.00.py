import os

# Set environment variables
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import itertools
import jax
import jax.numpy as jnp
from jax import vmap, jit
import pathlib
import equinox as eqx
from ddfa_node import embed_data, takens_embedding, change_trial_length, split_data, phaser, stats as statistics, jax_utils
import ddfa_node
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
# set 64-bit mode
# jax.config.update("jax_enable_x64", True)
# don't preallocate memory for XLA

project_path = pathlib.Path("/mnt/Mouse_Face_Project/Desktop/Data/Python/NeuralODE/")
window_length = 30

data = jnp.load(project_path.joinpath("outputs/VDP_oscillators.npy"))[:, :, ::3]
data = data.reshape(data.shape[0]*data.shape[1], data.shape[2], data.shape[3])


# Define the convolution function for a single time series
def convolve_1d(x):
    return jnp.convolve(x, jnp.ones((window_length,))/window_length, mode='valid')

# Vectorize over features
convolve_features = vmap(convolve_1d, in_axes=1, out_axes=1)
# Vectorize over trials
convolve_trials = vmap(convolve_features, in_axes=0, out_axes=0)

# Apply convolution to all trials and features at once
new_data = convolve_trials(data[:, :, :1])

# Use a savgol filter to get first derivative
polyorder = 4
dt = 1/20
# new_data_deriv = savgol_filter(new_data, window_length//2, polyorder, deriv=1, delta=dt, axis=1)
new_data_deriv = jnp.gradient(new_data, dt, axis=1)
new_data = jnp.concatenate([new_data, new_data_deriv], axis=2)
print(new_data.shape)

# Standardize the data
new_data = (new_data - jnp.mean(new_data, axis=1)[:, None, :]) / jnp.std(new_data, axis=1)[:, None, :]
data = new_data
print(data.shape)

def generate_feature_functions(num_features):
    """
    Generate a list of functions to apply to the features.
    
    Parameters:
    - num_features: Number of features in the dataset.

    Returns:
    - feature_functions: List of functions to apply to the features.
    """
    feature_functions = []

    # Add functions for squaring each feature
    for i in range(num_features):
        feature_functions.append(lambda x, i=i: x[i] ** 2)  # Capture i squared
        feature_functions.append(lambda x, i=i: x[i] ** 3)  # Capture i cubed
        feature_functions.append(lambda x, i=i: jnp.sin(x[i])) # Capture sin(i)
        feature_functions.append(lambda x, i=i: jnp.cos(x[i])) # Capture cos(i)


    if num_features > 1:
        # Add functions for multiplying pairs of features
        for i, j in itertools.combinations(range(num_features), 2):
            feature_functions.append(lambda x, i=i, j=j: x[i] * x[j])  # Capture i * j
            feature_functions.append(lambda x, i=i, j=j: x[i] * x[j] * x[i])  # Capture i² * j
            feature_functions.append(lambda x, i=i, j=j: x[i] * x[j] * x[j])  # Capture i * j²
    
    if num_features > 2:
        # Add functions for multiplying triplets of features
        for i, j, k in itertools.combinations(range(num_features), 3):
            feature_functions.append(lambda x, i=i, j=j, k=k: x[i] * x[j] * x[k])  # Capture i * j * k

    return feature_functions

def apply_feature_functions(data, feature_functions):
    """
    Apply a list of functions to each feature of the dataset.

    Parameters:
    - data: JAX array of shape (trials, timesteps, features)
    - feature_functions: List of functions to apply to each feature

    Returns:
    - new_data: JAX array with the original data and new features appended
    """
    # Ensure the input data is a JAX array
    data = jnp.asarray(data)

    # Create a function to apply all feature functions to a single timestep
    def apply_functions_to_timestep(timestep):
        arr = jnp.array([func(timestep) for func in feature_functions])
        return arr

    # Vectorize the function to apply it across all timesteps
    vectorized_apply = vmap(apply_functions_to_timestep, in_axes=0, out_axes=0)

    vectorized_apply_trials = vmap(vectorized_apply, in_axes=1, out_axes=1)

    # Apply the vectorized function across all trials and timesteps
    new_features = vectorized_apply_trials(data)

    # Concatenate the original data with the new features
    new_data = jnp.concatenate((data, new_features), axis=-1)

    return new_data

# Generate feature functions automatically
features = data.shape[-1]
feature_functions = generate_feature_functions(features)

# Apply the feature functions
data = apply_feature_functions(data, feature_functions)
print(data.shape)

# Normalize again
data = (data - jnp.mean(data, axis=1)[:, None, :]) / jnp.std(data, axis=1)[:, None, :]

def fit_linear_system_lstsq(time_series: jnp.ndarray, delta_t: float) -> tuple[jnp.ndarray, jnp.ndarray]:
    derivatives = (time_series[:, 1:] - time_series[:, :-1]) / delta_t
    X = time_series[:, :-1].reshape(-1, time_series.shape[2])
    Y = derivatives.reshape(-1, time_series.shape[2])
    
    # Use numpy's built-in least squares solver
    solution, residuals, rank, s = jnp.linalg.lstsq(X, Y, rcond=None)
    
    n_dims = time_series.shape[1]
    A = solution.T
    
    return A

A = fit_linear_system_lstsq(data, dt)
print(A.shape)