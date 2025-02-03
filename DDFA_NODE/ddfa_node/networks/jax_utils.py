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
            # activation=jnn.tanh,
            final_activation=eqx.nn.PReLU(),
            key=key,
        )

    def __call__(self, t, y, args):
        return self.mlp(y)

class LinearFunc(eqx.Module):
    A: eqx.nn.Linear

    def __init__(self, data_size, *, key, **kwargs):
        super().__init__(**kwargs)
        # self.A = jr.uniform(key, (data_size, data_size), minval=-jnp.sqrt(1/3), maxval=jnp.sqrt(1/3))
        self.A = eqx.nn.Linear(data_size, data_size, key=key)
        # max_eigenvalue = jnp.max(jnp.abs(jnp.linalg.eigvals(self.A)))
        # if max_eigenvalue > 1:
        #     self.A = self.A / max_eigenvalue
        #     print("Warning: LinearFunc max eigenvalue is greater than 1. Normalizing.")
        #     print("New max eigenvalue: ", jnp.max(jnp.abs(jnp.linalg.eigvals(self.A))))

    def __call__(self, t, y, args):
        return self.A(y)

class CombinedFunc(eqx.Module):
    func1: eqx.Module
    func2: eqx.Module

    def __init__(self, data_size, width_size, depth, *, key, scale=1.0, **kwargs):
        super().__init__(**kwargs)
        self.func1 = Func(data_size, width_size, depth, key=key)
        self.func2 = LinearFunc(data_size, key=key)

    def __call__(self, t, y, args):
        return self.func1(t, y, args) + self.func2(t, y, args)

class NeuralODE(eqx.Module):
    cell: eqx.nn.GRUCell
    func: eqx.Module
    hidden_to_ode: eqx.nn.MLP
    ode_to_data: eqx.nn.MLP
    use_recurrence: bool
    use_linear: bool
    padding_layer: eqx.nn.MLP  # New layer for padding input

    def __init__(self, data_size, width_size, hidden_size, depth, augment_dims, *, key, use_recurrence=True, use_linear=False, only_linear=False, **kwargs):
        super().__init__(**kwargs)
        self.use_recurrence = use_recurrence
        self.use_linear = use_linear
        ode_size = data_size + augment_dims
        rec_key, func_key, dec_key = jr.split(key, 3)
        if use_recurrence:
            self.cell = eqx.nn.GRUCell(data_size, hidden_size, key=rec_key)
        else:
            self.cell = eqx.nn.GRUCell(1, 1, key=rec_key)
        
        # Create a padding layer if augment_dims > 0
        if augment_dims != 0:
            assert data_size + augment_dims > 0 # allows for negative augment_dims down to 1 ODE dimension
            self.padding_layer = eqx.nn.MLP(
                in_size=data_size,
                out_size=data_size + augment_dims,
                width_size=width_size,
                depth=1,  # Single layer for padding
                # activation=jnn.tanh,
                key=dec_key,
            )
        # No padding layer if augment_dims is 0
        else:
            self.padding_layer = None

        if use_linear and only_linear:
            self.func = LinearFunc(ode_size, key=func_key)
        elif use_linear and not only_linear:
            self.func = CombinedFunc(ode_size, width_size, depth, key=func_key)
        else:
            self.func = Func(ode_size, width_size, depth, key=func_key)

        if use_recurrence or augment_dims > 0:
            self.hidden_to_ode = eqx.nn.MLP(
                in_size=hidden_size,
                out_size=ode_size,
                width_size=width_size,
                depth=2,
                # activation=jnn.tanh,
                # final_activation=eqx.nn.PReLU(),
                key=dec_key,
            )
            self.ode_to_data = eqx.nn.MLP(
                in_size=ode_size,
                out_size=data_size,
                width_size=width_size,
                depth=2,
                # activation=jnn.tanh,
                # final_activation=eqx.nn.PReLU(),
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
            if isinstance(self.padding_layer, eqx.nn.MLP):
                y0 = self.padding_layer(y0)

        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(self.func),
            diffrax.Tsit5(),
            #diffrax.Kvaerno5(),
            t0=ts[0],
            t1=ts[-1],
            dt0=ts[1] - ts[0],
            y0=y0,
            stepsize_controller=diffrax.PIDController(rtol=1e-4, atol=1e-6),
            saveat=diffrax.SaveAt(ts=ts),
            max_steps=int(2**16)
        )
        ys = solution.ys
        if self.use_recurrence or self.padding_layer is not None:
            out = jax.vmap(self.ode_to_data)(ys)
        else:
            out = ys
        return out
    
def train_NODE(
    data, timesteps_per_trial=500, t1=5.0, width_size=16, hidden_size=256,
    depth=2, batch_size=256, seed=55, augment_dims=0,
    lr_strategy=(3e-3, 3e-3), steps_strategy=(500, 500),
    length_strategy=(0.1, 1), plot=True, print_every=100,
    seeding_strategy=(0.1, 0.1), skip_strategy=(50, 100), filter_spec=lambda _: True,
    use_recurrence=True, use_linear=False, only_linear=False, model=None, plot_fn=None, 
    optim_type='lion', lmbda=1e-3, *, k=1
):
    key = jr.PRNGKey(seed)
    data_key, model_key, loader_key = jr.split(key, 3)
    ts = jnp.linspace(0.0, t1, timesteps_per_trial)
    features = data[0].shape[-1]
    
    if model is None:
        model = NeuralODE(features, width_size, hidden_size, depth, augment_dims, key=model_key, use_recurrence=use_recurrence, use_linear=use_linear, only_linear=only_linear)
    
    #@eqx.filter_jit
    #def l1_loss(diff_model, lmbda):
    #    params, _ = eqx.filter(diff_model, eqx.is_array)
    #    return lmbda*jnp.sum(jnp.abs(jtu.tree_flatten(params)))

    @eqx.filter_value_and_grad
    def grad_loss(diff_model, static_model, ti, yi, seeding_steps, lmbda):
        model = eqx.combine(diff_model, static_model)
        y_pred = jax.vmap(model, in_axes=(None, 0))(ti, yi[:, :seeding_steps, :])
        params, _ = jtu.tree_flatten(eqx.filter(model, eqx.is_array))
        # mse_loss = jnp.mean(jnp.abs((yi - y_pred))) 
        # l2_loss = jnp.sum(jnp.square(jnp.concatenate([param.ravel() for param in params])))
        # return mse_loss + lmbda*l2_loss
        return jnp.mean(jnp.abs((yi - y_pred))) + lmbda*jnp.sum(jnp.square(jnp.concatenate([param.ravel() for param in params])))

    
    @eqx.filter_jit
    def make_step(ti, yi, model, opt_state, seeding_steps, lmbda):
        diff_model, static_model = eqx.partition(model, filter_spec)
        value, grads = grad_loss(diff_model, static_model, ti, yi, seeding_steps, lmbda)
        grads = eqx.filter(grads, eqx.is_array)
        opt_state = eqx.filter(opt_state, eqx.is_array)
        # model_ = eqx.filter(model, eqx.is_array)
        updates, opt_state = optim.update(grads, opt_state, model, value=value, grad=grads, value_fn=grad_loss, ti=ti, yi=yi, seeding_steps=seeding_steps)
        model = eqx.apply_updates(model, updates)
        return value, model, opt_state
    
    try:
        for lr, steps, length, seeding, skip in zip(lr_strategy, steps_strategy, length_strategy, seeding_strategy, skip_strategy):
            #if 'best_model' in locals():
            #    model = copy.deepcopy(locals()['best_model'])
            #else:
            if True: # big
                lr = optax.schedules.warmup_exponential_decay_schedule(init_value=1e-6, peak_value=lr, warmup_steps=500, transition_steps=500, decay_rate=0.99, transition_begin=2000, staircase=True, end_value=5e-6)
                if optim_type == 'lion':
                    optim = optax.lion(learning_rate=lr)
                elif optim_type == 'lbfgs':
                    optim = optax.lbfgs(learning_rate=lr, linesearch=None)
                else:
                    optim = optax.adabelief(learning_rate=lr, nesterov=False)

                filtered_model = eqx.filter(model, eqx.is_array)
                opt_state = optim.init(filtered_model)
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
                loss, model, opt_state = make_step(_ts, yi, model, opt_state, seeding_steps, lmbda)
                
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
                            # plt.show()
                            plt.savefig("/mnt/Mouse_Face_Project/Desktop/Data/Python/NeuralODE/tmp/tmp.png")
                            plt.clf(); plt.cla()
                            plt.close()
                
                        else:
                            plot_fn(model, _ts, _ys, seeding_steps)
                    start = time.time()
    
    except (KeyboardInterrupt, eqx.EquinoxRuntimeError) as e:
        print(f"Exiting early (iteration {step}) due to {e}. Returning best model with loss: {best_loss}")
        return ts, ys, best_model
        # return ts, ys, model
    print(f"Training finished. Returning best model with loss: {best_loss}")
    return ts, ys, best_model
    # return ts, ys, model

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

from jax import vmap
import itertools
window_length = 30
# Define the convolution function for a single time series
def convolve_1d(x):
    return jnp.convolve(x, jnp.ones((window_length,))/window_length, mode='valid')

# Vectorize over features
convolve_features = vmap(convolve_1d, in_axes=1, out_axes=1)
# Vectorize over trials
convolve_trials = vmap(convolve_features, in_axes=0, out_axes=0)

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

# Create a function to apply all feature functions to a single timestep
def apply_functions_to_timestep(timestep, feature_functions):
    arr = jnp.array([func(timestep) for func in feature_functions])
    return arr

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

    # Vectorize the function to apply it across all timesteps
    vectorized_apply = vmap(lambda timestep: apply_functions_to_timestep(timestep, feature_functions), in_axes=0, out_axes=0)

    vectorized_apply_trials = vmap(vectorized_apply, in_axes=1, out_axes=1)

    # Apply the vectorized function across all trials and timesteps
    new_features = vectorized_apply_trials(data)

    # Concatenate the original data with the new features
    new_data = jnp.concatenate((data, new_features), axis=-1)

    return new_data