import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax.numpy as jnp
import jax.random as jr
import diffrax
from diffrax import ODETerm, ControlTerm, MultiTerm, VirtualBrownianTree

import equinox as eqx
from ddfa_node.networks.jax_utils import NeuralODE
from ddfa_node.utils.tde import takens_embedding, embed_data
from ddfa_node.utils.data_preparation import convolve_trials
import jax
from jax import vmap

window_length = 30
data = jnp.load("../outputs/VDP_oscillators.npy")[:, :, ::3]
data = data.reshape(data.shape[0]*data.shape[1], data.shape[2], data.shape[3])

new_data = convolve_trials(data, window_length)
# Standardize the data
new_data = (new_data - jnp.mean(new_data, axis=1)[:, None, :]) / jnp.std(new_data, axis=1)[:, None, :]
data = new_data
print(data.shape)


import numpy as np
skip = 300
_, k, τ = embed_data(np.array(data[:, skip:, :1]))

data_tde = takens_embedding(data[:, :, :1], τ, k)

print("Embedded data shape: ", data_tde.shape)

@eqx.filter_jit
def solve_sde(model, new_ts, yi, diffusion):
    t0, t1 = new_ts[0], new_ts[-1]
    brownian_motion = VirtualBrownianTree(t0, t1, tol=1e-3, shape=(), key=jr.PRNGKey(0))
    terms = MultiTerm(ODETerm(model.func), ControlTerm(diffusion, brownian_motion))
    if model.use_recurrence:
        hidden = jnp.zeros(model.cell.hidden_size)
        for yi_i in yi[::-1]:
            hidden = model.cell(yi_i, hidden)
        y0 = model.hidden_to_ode(hidden)
    else:
        y0 = yi[0, :]
        # Pad the input if augment_dims > 0
        if model.padding_layer is not None:
            y0 = model.padding_layer(y0)

    solution = diffrax.diffeqsolve(
                terms,
                diffrax.Kvaerno5(),
                stepsize_controller=diffrax.PIDController(rtol=1e-4, atol=1e-6, error_order=1.5),
                t0=t0,
                t1=t1,
                dt0=(new_ts[1] - new_ts[0]),
                y0=y0,
                saveat=diffrax.SaveAt(ts=new_ts),
                max_steps=None,
            )
    ys = solution.ys
    if model.use_recurrence or model.padding_layer is not None:
        out = jax.vmap(model.ode_to_data)(ys)
    else:
        out = ys
    return out

@eqx.filter_jit
def diffusion(t, y, args):
    noise_std = 0.5
    return jr.normal(jr.PRNGKey(jnp.int32(t)), shape=y.shape) * noise_std

model = NeuralODE(data_size=5, 
                    width_size=128, 
                    hidden_size=256, 
                    ode_size=6, 
                    depth=2,
                    augment_dims=0, 
                    key=jax.random.PRNGKey(42))

model = eqx.tree_deserialise_leaves("../outputs/vdp_model.eqx", model)
new_ts = jnp.linspace(0, 5, 500)
seeding_steps = 125
out = jax.vmap(solve_sde, in_axes=(None, None, 0, None))(model, new_ts, data_tde[:, 500:500+seeding_steps, :], diffusion)

# jnp.save("../outputs/gen_vdp_data.npy", out)

