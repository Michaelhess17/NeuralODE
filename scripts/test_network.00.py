import os
import ddfa_node
from ddfa_node.networks import jax_utils
from ddfa_node.utils import tde
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pathlib
import matplotlib.pyplot as plt

project_path = pathlib.Path("/mnt/Mouse_Face_Project/Desktop/Data/Python/NeuralODE/")
window_length = 30

data = jnp.load(project_path.joinpath("outputs/VDP_oscillators.npy"))[:, :, ::3]
data = data.reshape(data.shape[0]*data.shape[1], data.shape[2], data.shape[3])

args = jax_utils.get_args()

# Apply convolution to all trials and features at once
new_data = jax_utils.convolve_trials(data[:, :, :1])

# Use a savgol filter to get first derivative
polyorder = 4
dt = 1/20
# new_data_deriv = savgol_filter(new_data, window_length//2, polyorder, deriv=1, delta=dt, axis=1)
new_data_deriv = jnp.gradient(new_data, dt, axis=1)
new_data = jnp.concatenate([new_data, new_data_deriv], axis=2)

# Standardize the data
new_data = (new_data - jnp.mean(new_data, axis=1)[:, None, :]) / jnp.std(new_data, axis=1)[:, None, :]
data = new_data

skip = 1
_, k, τ = tde.embed_data(np.array(data[:, :, :]))

data_tde = tde.takens_embedding(data[:, :, :], τ, k)

print("Embedded data shape: ", data_tde.shape)

model = jax_utils.NeuralODE(
        data_size=data_tde.shape[-1],
        width_size=args.width_size,
        hidden_size=args.hidden_size,
        depth=args.depth,
        augment_dims=args.augment_dims,
        use_recurrence=args.use_recurrence,
        use_linear=args.use_linear,
        only_linear=args.only_linear,
        key=jax.random.PRNGKey(42)
)

model = eqx.tree_deserialise_leaves(project_path.joinpath("outputs/vdp_model.eqx"), model)


# Show example of model
t1 = data_tde.shape[1] * dt
ts = jnp.linspace(0, t1, data_tde.shape[1])
model_y = jax.vmap(model, in_axes=(None, 0))(ts, data_tde[:, :1, :])

fig, ax = plt.subplots(2)
errs = jnp.abs(model_y - data_tde)
ax[0].plot(data_tde[330, :, 0], label="Data", c="dodgerblue")
ax[0].plot(model_y[330, :, 0], label="Model", c="crimson")
ax[0].legend()
ax[0].set_xlabel("Time step")
ax[0].set_ylabel("Feature magnitude")
ax[0].set_title("Example self driving")

ax[1].set_prop_cycle(color=["skyblue", "dodgerblue", "deepskyblue", "steelblue"])
ax[1].plot(errs.mean(axis=0), label=["Dim 1", "Dim 2", "Dim 3", "Dim 4"])
ax[1].legend()
ax[1].set_xlabel("Time step")
ax[1].set_ylabel("Error (abs)")
ax[1].set_title("Average error of self driving")

plt.tight_layout()
plt.savefig(project_path.joinpath("figures/model_errors_through_time.png"), dpi=200)

plt.clf(); plt.cla(); plt.close('all')
ax = plt.subplot(111, projection="3d")
ax.scatter(data_tde[330, :, 0], data_tde[330, :, 1], data_tde[330, :, 2], c="dodgerblue", label="Data")

ts = jnp.linspace(0, t1*50, args.timesteps_per_trial*100)
model_y = model(ts, data_tde[330, :1, :])
ax.scatter(model_y[:, 0], model_y[:, 1], model_y[:, 2], c="crimson", label="Model")
ax.legend()
plt.tight_layout()
plt.savefig(project_path.joinpath("figures/trained_model_generation.png"), dpi=200)
