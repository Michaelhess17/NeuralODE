import os

# Set environment variables
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

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



# Apply convolution to all trials and features at once
new_data = jax_utils.convolve_trials(data[:, :, :1])

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



# Generate feature functions automatically
features = data.shape[-1]
feature_functions = jax_utils.generate_feature_functions(features)

# Apply the feature functions
# data = apply_feature_functions(data, feature_functions)
# print(data.shape)

# # Normalize again
# data = (data - jnp.mean(data, axis=1)[:, None, :]) / jnp.std(data, axis=1)[:, None, :]

# Plot data and save figure
plt.plot(data[-1, :, 0])
plt.plot(data[-1, :, 1])
plt.savefig(project_path.joinpath("figures/VDP_data_filtered.png"))

import numpy as np
skip = 300
_, k, τ = embed_data(np.array(data[:, :, :]))

data_tde = takens_embedding(data[:, :, :], τ, k)

print("Embedded data shape: ", data_tde.shape)

timesteps_per_trial = 200
t1 = timesteps_per_trial * dt

ts, ys, model = jax_utils.train_NODE(
    data_tde,
    timesteps_per_trial=timesteps_per_trial,
    t1=t1,
    width_size=128,
    hidden_size=256,
    depth=3,
    batch_size=2**10,
    seed=6970,
    lr_strategy=(1e-3,),
    steps_strategy=(20000,),
    length_strategy=(1.0,),
    skip_strategy=(3,),
    seeding_strategy=(1,),
    plot=True,
    print_every=100,
    k=1,
    use_recurrence=False,
    augment_dims=0,
    use_linear=True,
    only_linear=False,
    plot_fn=None,
    model=None,
    filter_spec=lambda _: True, # need to change when augmenting dimensions
    optim_type='adabelief',
    lmbda=0.1,
)

#def filter_spec(tree, features):
#    if isinstance(tree, eqx.nn.Linear):
#        if tree.shape == (features, features):
#            return False
#    return True
#
#best_linear_model = jnp.copy(model.func.A.weight)
#best_linear_bias = jnp.copy(model.func.A.bias)
## print(best_linear_model)
## print(jnp.linalg.eigvals(best_linear_model))
#
## serialize the model
## import equinox as eqx
## eqx.tree_serialise_leaves(project_path.joinpath("outputs/vdp_model.eqx"), model)
#
#
## # deserialize the model
#model = ddfa_node.networks.jax_utils.NeuralODE(
#    data_size=data.shape[-1],
#    width_size=128,
#    hidden_size=256,
#    depth=3,
#    augment_dims=2,
#    use_recurrence=False,
#    use_linear=True,
#    only_linear=False,
#    key=jax.random.PRNGKey(6970),
#)
#model = eqx.tree_at(lambda tree: tree.func.func2.A.weight, model, best_linear_model)
#model = eqx.tree_at(lambda tree: tree.func.func2.A.bias, model, best_linear_bias)
#assert jnp.allclose(model.func.func2.A.weight, best_linear_model)
#assert jnp.allclose(model.func.func2.A.bias, best_linear_bias)
## model = eqx.tree_deserialise_leaves(project_path.joinpath("outputs/vdp_model.eqx"), model)
#
#timesteps_per_trial = 200
#t1 = timesteps_per_trial * dt
#
#ts, ys, model = jax_utils.train_NODE(
#    data,
#    timesteps_per_trial=timesteps_per_trial,
#    t1=t1,
#    width_size=128,
#    hidden_size=256,
#    depth=3,
#    batch_size=1024,
#    seed=6969,
#    lr_strategy=(1e-5,),
#    steps_strategy=(50000,),
#    length_strategy=(1.0,),
#    skip_strategy=(1,),
#    seeding_strategy=(1/2,),
#    plot=True,
#    print_every=100,
#    k=1,
#    use_recurrence=False,
#    augment_dims=2,
#    use_linear=True,
#    only_linear=False,
#    plot_fn=None,
#    model=model,
#    filter_spec=lambda tree: filter_spec(tree, data.shape[-1]+2),
#    optim_type='adabelief',
#    lmbda=0.1,
#)

# Show example of model
ax = plt.subplot(111, projection="3d")
ax.scatter(data_tde[0, :, 0], data_tde[0, :, 1], data_tde[0, :, 2], c="dodgerblue", label="Data")

model_y = model(ts, data_tde[0, :1, :])
ax.scatter(model_y[:, 0], model_y[:, 1], model_y[:, 2], c="crimson", label="Model")
ax.legend()
plt.tight_layout()
# plt.show()
plt.savefig("/mnt/Mouse_Face_Project/Desktop/Data/Python/NeuralODE/figures/trained_model_generation.png")
plt.clf(); plt.cla()
plt.close()
