import jax
import optax
import diffrax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn

import equinox as eqx
from jax import tree_util
import seaborn as sb
import numpy as np

import copy
import time
import matplotlib.pyplot as plt
from ddfa_node.networks.jax_utils import Func, LinearFunc, dataloader
from ddfa_node.utils.tde import takens_embedding, embed_data
from ddfa_node.utils.data_preparation import convolve_trials, change_trial_length


def create_causal_weight_matrix(D, scaling_factor):
    """
    Create a square causal weight matrix with a block-like structure.
    
    Parameters:
    - D: int, the size of the input vector (number of features).
    - scaling_factor: int, the number of output nodes per input feature.
    
    Returns:
    - weight_matrix: numpy.ndarray, the square causal weight matrix of shape (scaling_factor * D, D).
    """
    # Calculate the number of output nodes (it will be scaling_factor * D)
    num_output_nodes = int(scaling_factor * D)
    
    # Initialize the weight matrix with scaled normal distribution
    weight_matrix = jr.normal(key=jr.PRNGKey(0), shape=(num_output_nodes, D))
    
    for i in range(D):
        # For feature i (column i in the input), determine the corresponding block
        end_row = (i + 1) * scaling_factor
        
        # Map this feature (column i) to the appropriate rows in the weight matrix
        # weight_matrix = weight_matrix.at[:end_row, i].set(1)
        weight_matrix = weight_matrix.at[end_row:, i].set(0)
    
    return weight_matrix


def invert_causal_weight_matrix(D, scaling_factor):
    """
    Create a causal weight matrix that pools blocks to return to the original state space size.
    Inputs:
    - D: int, the size of the original input vector (number of features).
    - scaling_factor: int, the number of output nodes per input feature.
    """
    inverted_matrix = jr.normal(key=jr.PRNGKey(0), shape=(D, D*scaling_factor))
    
    for i in range(D):
        start_col = i * scaling_factor
        end_col = D*scaling_factor
        # inverted_matrix = inverted_matrix.at[i, start_col:end_col].set(1)
        inverted_matrix = inverted_matrix.at[i, :start_col].set(0)
        inverted_matrix = inverted_matrix.at[i, end_col:].set(0)

        
    return inverted_matrix


class CausalFunc(eqx.Module):
    layers: list

    def __init__(self, D, scaling_factor, depth, *, key=jr.PRNGKey(0), **kwargs):
        super().__init__(**kwargs)
        self.layers = [create_causal_weight_matrix(D, scaling_factor)]
        for _ in range(depth - 1):
            self.layers.append(create_causal_weight_matrix(D * scaling_factor, 1))
        self.layers.append(invert_causal_weight_matrix(D, scaling_factor))

    def __call__(self, t, x, args):
        for layer in self.layers[:-1]:
            x = jnp.tanh(layer @ x)
        return self.layers[-1] @ x
    
class NeuralODE(eqx.Module):
    cell: eqx.nn.GRUCell
    func: eqx.Module
    hidden_to_ode: eqx.nn.MLP
    ode_to_data: eqx.nn.MLP
    use_recurrence: bool
    linear: bool
    causal: bool
    padding_layer: eqx.nn.MLP  # New layer for padding input

    def __init__(self, data_size, width_size, hidden_size, ode_size, depth, augment_dims, *, key, use_recurrence=True, linear=False, causal=True, **kwargs):
        super().__init__(**kwargs)
        if jnp.sum(jnp.array([linear, causal])) > 1:
            raise ValueError("Only one of linear, use_recurrence, and causal can be True")
        self.use_recurrence = use_recurrence
        self.linear = linear
        self.causal = causal
        rec_key, func_key, dec_key = jr.split(key, 3)
        if use_recurrence:
            self.cell = eqx.nn.GRUCell(data_size, hidden_size, key=rec_key)
        else:
            ode_size = data_size + augment_dims
            self.cell = None
        
        # Create a padding layer if augment_dims > 0
        if augment_dims != 0:
            assert data_size + augment_dims > 0 # allows for negative augment_dims down to 1 ODE dimension
            self.padding_layer = eqx.nn.MLP(
                in_size=data_size,
                out_size=data_size + augment_dims,
                width_size=width_size,
                depth=1,  # Single layer for padding
                activation=jnn.tanh,
                key=dec_key,
            )
        else:
            self.padding_layer = None  # No padding layer if augment_dims is 0

        if linear:
            self.func = LinearFunc(ode_size, key=func_key)
        elif causal:
            self.func = CausalFunc(ode_size, width_size // ode_size, depth, key=func_key)
        else:
            self.func = Func(ode_size, width_size, depth, key=func_key)

        if use_recurrence or augment_dims > 0:
            self.hidden_to_ode = eqx.nn.MLP(
                in_size=hidden_size,
                out_size=ode_size,
                width_size=width_size,
                depth=2,
                activation=jnn.tanh,
                key=dec_key,
            )
            self.ode_to_data = eqx.nn.MLP(
                in_size=ode_size,
                out_size=data_size,
                width_size=width_size,
                depth=2,
                activation=lambda x: x,
                key=dec_key,
            )
        else:
            self.hidden_to_ode = None
            self.ode_to_data = None

    def __call__(self, ts, yi):
        if self.use_recurrence:
            hidden = jnp.zeros(self.cell.hidden_size)
            for yi_i in yi[::-1]:
                hidden = self.cell(yi_i, hidden)
            y0 = self.hidden_to_ode(hidden)
        else:
            y0 = yi[0, :]
            # Pad the input if augment_dims > 0
            if self.padding_layer is not None:
                y0 = self.padding_layer(y0)

        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(self.func),
            diffrax.Tsit5(),
            t0=ts[0],
            t1=ts[-1],
            dt0=ts[1] - ts[0],
            y0=y0,
            stepsize_controller=diffrax.PIDController(rtol=1e-4, atol=1e-6),
            saveat=diffrax.SaveAt(ts=ts),
        )
        ys = solution.ys
        if self.use_recurrence or self.padding_layer is not None:
            out = jax.vmap(self.ode_to_data)(ys)
        else:
            out = ys
        return out

def train_NODE(
    data, timesteps_per_trial=500, t1=5.0, width_size=16, hidden_size=256,
    ode_size=128, depth=2, batch_size=256, seed=55, augment_dims=0,
    lr_strategy=(3e-3, 3e-3), steps_strategy=(500, 500),
    length_strategy=(0.1, 1), plot=True, print_every=100,
    seeding_strategy=(0.1, 0.1), skip_strategy=(50, 100),
    use_recurrence=True, linear=False, causal=True, model=None, plot_fn=None, filter_spec=None, *, k
):
    key = jr.PRNGKey(seed)
    data_key, model_key, loader_key = jr.split(key, 3)
    ts = jnp.linspace(0.0, t1, timesteps_per_trial)
    features = data[0].shape[-1]
    
    def filter_model(layer):
        if isinstance(layer, jnp.ndarray) and layer.ndim == 2:
            return layer != 0
        else:
            return True
        
    if model is None:
        model = NeuralODE(features, width_size, hidden_size, ode_size, depth, augment_dims, key=model_key, use_recurrence=use_recurrence, linear=linear)
        filter_spec = jax.tree.map(filter_model, model)
    
    @eqx.filter_jit
    def make_step(model, ti, yi, opt_state, seeding_steps, filter_spec):
        @eqx.filter_value_and_grad
        def loss(model, ti, yi, seeding_steps):
            return jnp.mean(jnp.abs((jax.vmap(model, in_axes=(None, 0))(ti, yi[:, :seeding_steps, :]) - yi)))

        loss, grads = loss(model, ti, yi, seeding_steps)
        def mask_grad(grad, mask):
            if grad is None or mask is None:
                return None
            else:
                return grad * mask
        masked_grads = jax.tree_util.tree_map(mask_grad, grads, filter_spec, is_leaf=lambda x: x is None)
        updates, opt_state = optim.update(masked_grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state
    
    try:
        for lr, steps, length, seeding, skip in zip(lr_strategy, steps_strategy, length_strategy, seeding_strategy, skip_strategy):
            if 'best_model' in locals():
                model = copy.deepcopy(locals()['best_model'])
            else:
                lr = optax.schedules.warmup_exponential_decay_schedule(init_value=1e-6, peak_value=lr, warmup_steps=500, transition_steps=500, decay_rate=0.995, transition_begin=2000, staircase=True, end_value=3e-5)
                optim = optax.adabelief(lr)
                opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
            if skip is None:
                skip_use = (timesteps_per_trial * length) // 2
            else:
                skip_use = skip
            ys = []
            for idx in range(len(data)):
                ys.append(change_trial_length(jnp.expand_dims(data[idx], axis=0), timesteps_per_subsample=int(timesteps_per_trial * length), skip=skip_use))
            new_ys = []
            for y in ys:
                if len(y.shape) > 1:
                    new_ys.append(y)
            ys = jnp.concatenate(new_ys).astype(float)
            # shuffle the data
            ys = jax.random.permutation(loader_key, ys)
            print(ys.shape)
            trials, timesteps, features = ys.shape
            _ts = ts[: int(timesteps)]
            _ys = ys[:, : int(timesteps)]
            if seeding is not None:
                seeding_steps = int(timesteps_per_trial * seeding)
            else:
                seeding_steps = None
            
            # Training loop
            start = time.time()
            best_loss = jnp.inf
            for step, (yi,) in zip(range(steps), dataloader((_ys,), batch_size, key=loader_key)):
                loss, model, opt_state = make_step(model, _ts, yi, opt_state, seeding_steps, filter_spec)
                
                if loss < best_loss:
                    best_model = copy.deepcopy(model)
                    best_loss = copy.deepcopy(loss)
                
                if (step % print_every) == 0 or step == steps - 1:
                    end = time.time()
                    print(f"Step: {step}, Loss: {loss}, Current Best Loss: {best_loss}, Computation time: {end - start}")
                
                    if plot:
                
                        if plot_fn is None:
                            ax = plt.subplot(111, projection="3d")
                            ax.scatter(_ys[0, :, 0], _ys[0, :, k], _ys[0, :, 2*k], c="dodgerblue", label="Data")
                            model_y = model(_ts, _ys[0, :seeding_steps])
                            ax.scatter(model_y[:, 0], model_y[:, k], model_y[:, 2*k], c="crimson", label="Model")
                            ax.legend()
                            plt.tight_layout()
                            plt.show()
                            plt.savefig("/home/michael/Synology/Python/NeuralODE/tmp/tmp.png")
                
                        else:
                            plot_fn(model, _ts, _ys, seeding_steps)
                    start = time.time()
    
    except KeyboardInterrupt:
        print(f"Exiting early (iteration {step}). Returning best model with loss: {best_loss}")
        return ts, ys, best_model
    print(f"Training finished. Returning best model with loss: {best_loss}")
    return ts, ys, best_model

# Example usage
D = 5  # Size of the input vector
scaling_factor = 12  # Number of output nodes per input feature


# batch_size = 512
# x = jr.normal(key=jr.PRNGKey(42), shape=(batch_size, D))
# A = jr.normal(key=jr.PRNGKey(900), shape=(D, D))
# y = jax.vmap(lambda x: A @ x)(x)
# for i in range(10000):
#     func, opt_state = make_step(func, x, y, opt_state)


window_length = 30
data = jnp.load("outputs/VDP_oscillators.npy")[:, :, ::3]
data = data.reshape(data.shape[0]*data.shape[1], data.shape[2], data.shape[3])

new_data = convolve_trials(data, window_length)
# Standardize the data
new_data = (new_data - jnp.mean(new_data, axis=1)[:, None, :]) / jnp.std(new_data, axis=1)[:, None, :]
data = new_data
print(data.shape)


skip = 300
_, k, τ = embed_data(np.array(data[:, skip:, :1]))

data_tde = takens_embedding(data[:, :, :1], τ, k)

features = data_tde.shape[-1]


key = jr.PRNGKey(0)

# train network
ts, ys, model = train_NODE(data_tde, timesteps_per_trial=300, t1=5.0, width_size=128, hidden_size=256,
    ode_size=8, depth=3, batch_size=256, seed=55, augment_dims=0,
    lr_strategy=(3e-3, 3e-3), steps_strategy=(50000, 50000),
    length_strategy=(0.5, 1), plot=True, print_every=1000,
    seeding_strategy=(0.5, 0.5), skip_strategy=(5, 10),
    use_recurrence=True, linear=False, causal=True, model=None, plot_fn=None, filter_spec=None, k=2)

