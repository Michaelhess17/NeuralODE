import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
from jax import vmap, jit
# set 64-bit mode
# jax.config.update("jax_enable_x64", True)

from ddfa_node import embed_data, takens_embedding, change_trial_length, split_data, get_aics, get_λs, phaser, stats as statistics, jax_utils
import ddfa_node

from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

window_length = 30
data = jnp.load("outputs/VDP_oscillators.npy")[:, :, ::3]
data = data.reshape(data.shape[0]*data.shape[1], data.shape[2], data.shape[3])


# Define the convolution function for a single time series
def convolve_1d(x):
    return jnp.convolve(x, jnp.ones((window_length,))/window_length, mode='valid')

# Vectorize over features
convolve_features = vmap(convolve_1d, in_axes=1, out_axes=1)
# Vectorize over trials
convolve_trials = vmap(convolve_features, in_axes=0, out_axes=0)

# Apply convolution to all trials and features at once
new_data = convolve_trials(data)


# Standardize the data
new_data = (new_data - jnp.mean(new_data, axis=1)[:, None, :]) / jnp.std(new_data, axis=1)[:, None, :]
data = new_data
print(data.shape)

# Plot data and save figure
plt.plot(data[-1, :, 0])
plt.plot(data[-1, :, 1])
plt.savefig("figures/VDP_data_filtered.png")

import numpy as np
skip = 300
_, k, τ = embed_data(np.array(data[:, skip:, :1]))

data_tde = takens_embedding(data[:, :, :1], τ, k)

print("Embedded data shape: ", data_tde.shape)


# , static_argnames=["timesteps_per_trial", "skip", "t1", "width_size", "hidden_size", "ode_size", "depth", "batch_size", "seed", "print_every", "length_strategy", "lr_strategy", "seeding_strategy", "steps_strategy", "plot","k"
ts, ys, model = jax_utils.train_NODE(
    # model=model,
    data_tde,
    timesteps_per_trial=500,
    t1=5.0,
    width_size=128,
    hidden_size=256,
    ode_size=6,
    depth=2,
    batch_size=256,
    seed=6969,
    lr_strategy=(8e-4, 3e-4),
    steps_strategy=(40000, 70000),
    length_strategy=(0.3, 1.0),
    skip_strategy=(1, 2),
    seeding_strategy=(0.1, 0.25),
    plot=False,
    print_every=1000,
    k=1,
    linear=False,
    plot_fn=None,
)

# serialize the model
import equinox as eqx
eqx.tree_serialise_leaves("outputs/vdp_model.eqx", model)