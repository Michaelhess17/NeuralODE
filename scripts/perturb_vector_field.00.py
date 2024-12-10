import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import jax.nn as jnn
from ddfa_node.networks.jax_utils import NeuralODE
from ddfa_node.utils.data_preparation import convolve_trials
import jax
import diffrax
import numpy as np
from ddfa_node.utils.tde import embed_data, takens_embedding
from functools import partial

class PerturbFunc(eqx.Module):
    func: NeuralODE
    mlp: eqx.nn.MLP
    scale: float
    perturb_direction: jnp.ndarray
    perturb_strength: float
    perturb_time: float

    def __init__(self, model, perturb_strength, perturb_time, *, key, scale=1.0, **kwargs):
        super().__init__(**kwargs)
        self.func = model.func
        self.mlp = self.func.mlp
        self.scale = scale
        # self.perturb_direction = jr.uniform(key, (self.mlp.in_size,), minval=-1, maxval=1)
        # self.perturb_direction = self.perturb_direction / jnp.linalg.norm(self.perturb_direction)
        self.perturb_direction = jnp.zeros(self.mlp.in_size)
        self.perturb_direction = self.perturb_direction.at[0].set(1.0)
        self.perturb_strength = perturb_strength
        self.perturb_time = perturb_time

    def perturb(self, t, y):
        # ramp up over the whole perturbation time (and zero out after)
        ramp = jnp.where(t <= self.perturb_time, t/self.perturb_time, 0)
        return self.perturb_strength * ramp * self.perturb_direction

    def __call__(self, t, y, args):
        return self.mlp(y) + self.perturb(t, y)

class PerturbNeuralODE(eqx.Module):
    model: NeuralODE
    perturb_func: PerturbFunc
    terms: diffrax.ODETerm
    def __init__(self, model, perturb_strength, perturb_time, *, key, scale=1.0, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.perturb_func = PerturbFunc(model, perturb_strength, perturb_time, key=key, scale=scale)
        self.terms = diffrax.ODETerm(self.perturb_func)

    @eqx.filter_jit
    def __call__(self, t, y):
        t0, t1 = t[0], t[-1]
        if self.model.use_recurrence:
            hidden = jnp.zeros(self.model.cell.hidden_size)
            for yi_i in y[::-1]:
                hidden = self.model.cell(yi_i, hidden)
            y0 = self.model.hidden_to_ode(hidden)
        else:
            y0 = y[0, :]
            # Pad the input if augment_dims > 0
            if self.model.padding_layer is not None:
                y0 = self.model.padding_layer(y0)

        solution = diffrax.diffeqsolve(
                    self.terms,
                    diffrax.Tsit5(),
                    t0=t0,
                    t1=t1,
                    dt0=(t[1] - t0),
                    y0=y0,
                saveat=diffrax.SaveAt(ts=t),
                max_steps=None,
            )
        ys = solution.ys
        if self.model.use_recurrence or self.model.padding_layer is not None:
            out = jax.vmap(self.model.ode_to_data)(ys)
        else:
            out = ys
        return out
    

window_length = 30
data = jnp.load("../outputs/VDP_oscillators.npy")[:, :, ::3]
data = data.reshape(data.shape[0]*data.shape[1], data.shape[2], data.shape[3])
data = convolve_trials(data, window_length)
model = NeuralODE(data_size=5, 
                    width_size=128, 
                    hidden_size=256, 
                    ode_size=6, 
                    depth=2,
                    augment_dims=0, 
                    key=jax.random.PRNGKey(42))



model = eqx.tree_deserialise_leaves("../outputs/vdp_model.eqx", model)
perturb_strength = -2.0
perturb_time = 0.2
perturb_func = PerturbNeuralODE(model, perturb_strength, perturb_time, key=jax.random.PRNGKey(5678))
new_ts = jnp.linspace(0, 10, 1000)
seeding_steps = 125
offset = 100

skip = 300
_, k, τ = embed_data(np.array(data[:, skip:, :1]))

data_tde = takens_embedding(data[:, :, :1], τ, k)

out = jax.vmap(lambda x: perturb_func(new_ts, x))(data_tde[:, offset:offset+seeding_steps])

import matplotlib.pyplot as plt

plt.scatter(new_ts, out[0, :, 4], s=10, label="Generated")
plt.scatter(new_ts[:data_tde.shape[1]-500], data_tde[0, 500:new_ts.shape[0]+500, 4], s=10, label="Real")
plt.fill_between(x=[0, 0.2], y1=plt.ylim()[0], y2=plt.ylim()[1],  color="gray", alpha=0.2, label="Perturbation")
plt.legend()
plt.title(f"Tsit5 w/ dt=0.01 + |Perturbation|={int(perturb_strength)})")
plt.savefig("../tmp/tmp.png")

print(out.shape)