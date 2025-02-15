import os
# Set environment variables
from ddfa_node import embed_data, takens_embedding, change_trial_length, split_data, phaser, stats as statistics, jax_utils
args = jax_utils.get_args()
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.GPU}"

import itertools
import jax
import jax.numpy as jnp
from jax import vmap, jit
import pathlib
import equinox as eqx
import ddfa_node
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import json
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
dt = 1/10
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
skip = 1
_, k, τ = embed_data(np.array(data[:, :, :]))

data_tde = takens_embedding(data[:, :, :], τ, k)

print("Embedded data shape: ", data_tde.shape)

t1 = args.timesteps_per_trial * dt
ts = jnp.tile(jnp.linspace(0, t1, args.timesteps_per_trial), reps=(data_tde.shape[0], 1))

ts, ys, model = jax_utils.train_NODE(
    data_tde,
    timesteps_per_trial=args.timesteps_per_trial,
    t1=t1,
    width_size=args.width_size,
    hidden_size=args.hidden_size,
    depth=args.depth,
    batch_size=args.batch_size,
    seed=args.seed,
    lr_strategy=(args.lr,),
    steps_strategy=(args.steps,),
    length_strategy=(args.length,),
    skip_strategy=(args.skip,),
    seeding_strategy=(args.seeding,),
    plot=True,
    print_every=100,
    k=args.k,
    use_recurrence=args.use_recurrence,
    augment_dims=args.augment_dims,
    use_linear=args.use_linear,
    only_linear=args.only_linear,
    plot_fn=None,
    model=None,
    filter_spec=lambda _: True,
    optim_type=args.optim_type,
    lmbda=args.lmbda,
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

ts = jnp.linspace(0, t1*1.5, args.timesteps_per_trial*50)
model_y = model(ts, data_tde[0, :args.timesteps_per_trial, :])
ax.scatter(model_y[:, 0], model_y[:, 1], model_y[:, 2], c="crimson", label="Model")
ax.legend()
plt.tight_layout()
# plt.show()
plt.savefig("/mnt/Mouse_Face_Project/Desktop/Data/Python/NeuralODE/figures/trained_model_generation.png")
plt.clf(); plt.cla()
plt.close()


def save_config():
    with open('commandline_args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)


def load_config():
    parser = ArgumentParser()
    args = parser.parse_args()
    with open('commandline_args.txt', 'r') as f:
        args.__dict__ = json.load(f)


if args.save_config:
    save_config()
