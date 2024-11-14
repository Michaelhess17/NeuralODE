import time
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import jax.nn as jnn
import diffrax
import equinox as eqx
import matplotlib.pyplot as plt
import sys
sys.path.append("/home/michael/Code/optax/")
import optax
sys.path.append("/home/michael/Synology/Desktop/Data/Python/Gait-Signatures/NeuralODE/DDFA_NODE/")
from functools import partial
from jax.experimental.host_callback import call
import copy

def change_trial_length(data, timesteps_per_subsample=100, skip=1, get_subject_ids=False):
    """Resamples data into fixed length trials using dynamic slicing.
    Only data argument is traced, others are static."""
    num_subjects, num_time_steps, num_features = data.shape
    num_subsamples = (num_time_steps - timesteps_per_subsample) // skip + 1
    subsamples = jnp.zeros((num_subjects*num_subsamples, timesteps_per_subsample, num_features))
    subject_ids = jnp.zeros((num_subjects*num_subsamples,))

    def process_subject(i, carry):
        subsamples, subject_ids = carry
        
        def process_subsample(j, carry):
            subsamples, subject_ids = carry
            start_index = j * skip
            subsample = jax.lax.dynamic_slice(
                data[i], 
                (start_index, 0), 
                (timesteps_per_subsample, num_features)
            )
            idx = i*num_subsamples + j
            subsamples = subsamples.at[idx].set(subsample)
            subject_ids = subject_ids.at[idx].set(i)
            return (subsamples, subject_ids)
            
        return jax.lax.fori_loop(0, num_subsamples, process_subsample, (subsamples, subject_ids))

    subsamples, subject_ids = jax.lax.fori_loop(
        0, num_subjects, process_subject, (subsamples, subject_ids)
    )

    if get_subject_ids:
        return subsamples, subject_ids
    return subsamples
change_trial_length = jax.jit(change_trial_length, static_argnums=[1, 2, 3])

def dataloader(arrays, batch_size, *, key):
    dataset_size = arrays[0].shape[0]
    indices = jnp.arange(dataset_size)
    while True:
        perm = jr.permutation(key, indices)
        (key,) = jr.split(key, 1)
        start = 0
        end = batch_size
        while start < dataset_size:
            batch_perm = perm[start:end]
            yield tuple(array[batch_perm] for array in arrays)
            start = end
            end = start + batch_size

class Func(eqx.Module):
    mlp: eqx.nn.MLP
    scale: float

    def __init__(self, data_size, width_size, depth, *, key, scale=1.0, **kwargs):
        super().__init__(**kwargs)
        self.scale = scale
        self.mlp = eqx.nn.MLP(
            in_size=data_size,
            out_size=data_size,
            width_size=width_size,
            depth=depth,
            activation=jnn.tanh,
            final_activation=lambda x: x,
            key=key,
        )

    def __call__(self, t, y, args):
        return self.mlp(y)

class LinearFunc(eqx.Module):
    A: jnp.ndarray

    def __init__(self, data_size, *, key, **kwargs):
        super().__init__(**kwargs)
        self.A = jr.uniform(key, (data_size, data_size), minval=-jnp.sqrt(1/3), maxval=jnp.sqrt(1/3))

    def __call__(self, t, y, args):
        return self.A @ y

class NeuralODE(eqx.Module):
    cell: eqx.nn.GRUCell
    func: eqx.Module
    hidden_to_ode: eqx.nn.MLP
    ode_to_data: eqx.nn.MLP
    use_recurrence: bool
    linear: bool
    padding_layer: eqx.nn.MLP  # New layer for padding input

    def __init__(self, data_size, width_size, hidden_size, ode_size, depth, augment_dims, *, key, use_recurrence=True, linear=False, **kwargs):
        super().__init__(**kwargs)
        self.use_recurrence = use_recurrence
        self.linear = linear
        rec_key, func_key, dec_key = jr.split(key, 3)
        if use_recurrence:
            self.cell = eqx.nn.GRUCell(data_size, hidden_size, key=rec_key)
        else:
            ode_size = data_size + augment_dims
            self.cell = None
        
        # Create a padding layer if augment_dims > 0
        if augment_dims > 0:
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

def train_NODE(data, timesteps_per_trial=500, t1=5.0, width_size=16, hidden_size=256, ode_size=128, depth=2, batch_size=256, seed=55, augment_dims=0, lr_strategy=(3e-3, 3e-3), steps_strategy=(500, 500), length_strategy=(0.1, 1), plot=True, print_every=100, seeding_strategy=(0.1, 0.1), skip_strategy=(50, 100), use_recurrence=True, linear=False, model=None, plot_fn=None, *, k):
    key = jr.PRNGKey(seed)
    data_key, model_key, loader_key = jr.split(key, 3)
    ts = jnp.linspace(0.0, t1, timesteps_per_trial)
    features = data[0].shape[-1]
    if model is None:
        model = NeuralODE(features, width_size, hidden_size, ode_size, depth, augment_dims, key=model_key, use_recurrence=use_recurrence, linear=linear)
    @eqx.filter_value_and_grad
    def grad_loss(model, ti, yi, seeding_steps):
        y_pred = jax.vmap(model, in_axes=(None, 0))(ti, yi[:, :seeding_steps, :])
        return jnp.mean(jnp.abs((yi - y_pred)))
    @eqx.filter_jit
    def make_step(ti, yi, model, opt_state, seeding_steps):
        loss, grads = grad_loss(model, ti, yi, seeding_steps)
        updates, opt_state = optim.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state
    try:
        for lr, steps, length, seeding, skip in zip(lr_strategy, steps_strategy, length_strategy, seeding_strategy, skip_strategy):
            if 'best_model' in locals():
                model = copy.deepcopy(best_model)
            else:
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
            seeding_steps = int(timesteps_per_trial * seeding)
            start = time.time()
            best_loss = jnp.inf
            for step, (yi,) in zip(range(steps), dataloader((_ys,), batch_size, key=loader_key)):
                loss, model, opt_state = make_step(_ts, yi, model, opt_state, seeding_steps)
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
                        else:
                            plot_fn(model, _ts, _ys, seeding_steps)
                    start = time.time()
    except KeyboardInterrupt:
        print(f"Exiting early (iteration {step}). Returning best model with loss: {best_loss}")
        return ts, ys, best_model
    print(f"Training finished. Returning best model with loss: {best_loss}")
    return ts, ys, best_model

def get_parameters(model):
    params = jtu.tree_leaves(eqx.filter(model, eqx.is_array))
    return jnp.sum(jnp.array([x.size for x in params]))

def print_parameter_count(model):
    # Get all trainable parameters
    params = eqx.filter(model, eqx.is_array)
    
    total_params = 0
    # Iterate through the parameter tree
    for idx, param in enumerate(jtu.tree_flatten(params)[0]):
        param_count = jnp.size(param)
        # Create a string representation of the path
        path_str = 'Layer ' + '.'.join(str(idx))
        print(f"{path_str}: {param_count:,} parameters")
        total_params += param_count
    
    print(f"\nTotal trainable parameters: {total_params:,}")
