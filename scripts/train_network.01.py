# Script to train the neural ODE using the phase information
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
from ddfa_node.utils import get_args
args = get_args()
print(args.__dict__)
os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.GPU}"
from ddfa_node import embed_data, takens_embedding, change_trial_length, split_data, phaser, stats as statistics, jax_utils
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
from ddfa_node.utils import phaser
import numpy as np
# set 64-bit mode
# jax.config.update("jax_enable_x64", True)
# don't preallocate memory for XLA

project_path = pathlib.Path("/mnt/Mouse_Face_Project/Desktop/Data/Python/NeuralODE/")
window_length = 30

data = jnp.load(project_path.joinpath("outputs/VDP_oscillators.npy"))[:, :, ::3]
data = data.reshape(data.shape[0]*data.shape[1], data.shape[2], data.shape[3])

# Apply convolution to all trials and features at once
new_data = jax_utils.convolve_trials(data[:10, :, :1])

# Use a savgol filter to get first derivative
polyorder = 4
dt = 1/20
new_data_deriv = jnp.gradient(new_data, dt, axis=1)
new_data = jnp.concatenate([new_data, new_data_deriv], axis=2)

# Standardize the data
new_data = (new_data - jnp.mean(new_data, axis=1)[:, None, :]) / jnp.std(new_data, axis=1)[:, None, :]
data = new_data
print(data.shape)

skip = 1
_, k, τ = embed_data(np.array(data[:, :, :]))
data_tde = takens_embedding(data[:, :, :], τ, k)
print("Embedded data shape: ", data_tde.shape)

print("\nComputing phase using Phaser")
phis = jnp.zeros((data_tde.shape[0], data_tde.shape[1]))
for a in range(data_tde.shape[0]):
    raw = np.array(data_tde[a, :, :2].T)

    # Assess phase using Phaser
    phr = phaser.Phaser([ raw ])
    phi = phr.phaserEval( raw ) # extract phase
    phis = phis.at[a, :].set(phi[0, :])

# Interpolate phis and data_tde to regular spacing
print("\nInterpolating phis and data_tde")
new_phis = jnp.zeros(phis.shape)
new_data_tde = jnp.zeros(data_tde.shape)
for a in range(data_tde.shape[0]):
    delta_phi = phis[a, -1] - phis[a, 0]
    new_phi = jnp.linspace(phis[a, 0], delta_phi, data_tde.shape[1])
    new_phis = new_phis.at[a].set(new_phi)
    for feat in range(data_tde.shape[2]):
        new_data_tde = new_data_tde.at[a, :, feat].set(jnp.interp(new_phi, phis[a], data_tde[a, :, feat]))

# Plot data and save figure
plt.plot(new_phis[0], new_data_tde[0, :, :])
plt.savefig(project_path.joinpath("figures/VDP_data_filtered.png_interpolated.png"))

phase_scaler = 1

# Train NODE
print("\nTraining NODE")
ts, ys, model = jax_utils.train_NODE(
    new_data_tde,
    timesteps_per_trial=args.timesteps_per_trial,
    t1=None,
    ts=new_phis*phase_scaler,
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

# Show example of model
ax = plt.subplot(111, projection="3d")
ax.scatter(data_tde[0, :, 0], data_tde[0, :, 1], data_tde[0, :, 2], c="dodgerblue", label="Data")

ts = jnp.linspace(0, phis[0, -1]*1.5, args.timesteps_per_trial*5)
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
