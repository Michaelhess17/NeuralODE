import time
import jax
# jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn
import jax.tree_util as jtu
import diffrax
import equinox as eqx
import matplotlib.pyplot as plt
import sys
sys.path.append("/home/michael/Code/optax/")
import optax
sys.path.append("/home/michael/Synology/Desktop/Data/Python/Gait-Signatures/NeuralODE/DDFA_NODE/")
# from src.utils_data_preparation import change_trial_length, split_data
from functools import partial
from jax.experimental.host_callback import call
import copy

def change_trial_length(data, timesteps_per_subsample=100, skip=1, get_subject_ids=False):
    num_subjects, num_time_steps, num_features = data.shape
    
    # Calculate the number of subsamples
    num_subsamples = (num_time_steps - timesteps_per_subsample) // skip + 1
    subsamples = jnp.zeros((num_subjects*num_subsamples, timesteps_per_subsample, num_features))
    subject_ids = jnp.zeros((num_subjects*num_subsamples,))

    # Iterate over each subject
    subject = 0
    for idx, subject_data in enumerate(data):
        # Iterate over each subsample
        for jdx in range(num_subsamples):
            start_index = jdx * skip
            end_index = start_index + timesteps_per_subsample
            subsample = subject_data[start_index:end_index, :]
            subsamples = subsamples.at[idx*num_subsamples+jdx].set(subsample)
            subject_ids = subject_ids.at[idx*num_subsamples+jdx].set(subject)
        subject += 1

    if get_subject_ids:
        return subsamples, subject_ids
    return subsamples

def dataloader(arrays, batch_size, *, key):
    dataset_size = arrays[0].shape[0]
    # print([array.shape for array in arrays])
    # assert all(array.shape[0] == dataset_size for array in arrays)
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
    # bn: eqx.nn.BatchNorm
    scale: float

    def __init__(self, data_size, width_size, depth, *, key, scale=1.0, **kwargs):
        super().__init__(**kwargs)
        self.scale = scale
        # self.bn = eqx.nn.BatchNorm(data_size, axis_name="batch")
        self.mlp = eqx.nn.MLP(
            in_size=data_size,
            out_size=data_size,
            width_size=width_size,
            depth=depth,
            activation=jnn.tanh,
            # final_activation=lambda x: jnn.selu(x) * self.scale,
            final_activation=lambda x: x,
            key=key,
        )

    def __call__(self, t, y, args):
        return self.mlp(y)
        
class GRU(eqx.Module):
    cell: eqx.nn.GRUCell

    def __init__(self, input_size, hidden_size, *, key):
        self.cell = eqx.nn.GRUCell(input_size, hidden_size, key=key)

    def __call__(self, xs):
        scan_fn = lambda state, x: (self.cell(x, state), None)
        
        final_state, _ = jax.lax.scan(scan_fn, init_state, xs)
        return final_state
    
    
class AugmentedNeuralODE(eqx.Module):
    cell: eqx.nn.GRUCell
    func: Func
    hidden_to_ode: eqx.nn.MLP
    ode_to_data: eqx.nn.MLP

    def __init__(self, data_size, width_size, hidden_size, ode_size, depth, *, key, scale=1.0, **kwargs):
        super().__init__(**kwargs)
        rec_key, func_key, dec_key = jr.split(key, 3)
        self.cell = eqx.nn.GRUCell(data_size, hidden_size, key=rec_key)
        self.func = Func(ode_size, width_size, depth, key=func_key, scale=scale)
        
        self.hidden_to_ode = eqx.nn.MLP(
            in_size=hidden_size,
            out_size=ode_size,
            width_size=width_size,
            depth=2,
            activation=jnn.tanh,
            key=key,
        )
        self.ode_to_data = eqx.nn.MLP(
            in_size=ode_size,
            out_size=data_size,
            width_size=width_size,
            depth=2,
            activation=lambda x: x,
            key=key,
        )
        
    def __call__(self, ts, yi):
        hidden = jnp.zeros(self.cell.hidden_size)
        for yi_i in yi[::-1]:
            hidden = self.cell(yi_i, hidden)
        y0 = self.hidden_to_ode(hidden)
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(self.func),
            diffrax.Tsit5(),
            # diffrax.Euler(),
            t0=ts[0],
            t1=ts[-1],
            dt0=ts[1] - ts[0],
            y0=y0,
            # y0=hidden,
            stepsize_controller=diffrax.PIDController(rtol=1e-4, atol=1e-6),
            saveat=diffrax.SaveAt(ts=ts),
#             max_steps=,
#             adjoint=diffrax.BacksolveAdjoint()
        )
        ys = solution.ys
        out = jax.vmap(self.ode_to_data)(ys)
        return out
    
    def forward(self, ts, yi):
        hidden = jnp.zeros(self.cell.hidden_size)
        for yi_i in yi[::-1]:
            hidden = self.cell(yi_i, hidden)
        y0 = self.hidden_to_ode(hidden)
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(self.func),
            diffrax.Tsit5(),
            t0=ts[0],
            t1=ts[-1],
            dt0=ts[1] - ts[0],
            y0=y0,
            # y0=hidden,
            stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-8),
            saveat=diffrax.SaveAt(ts=ts),
            max_steps=None,
#             adjoint=diffrax.BacksolveAdjoint()
        )
        ys = solution.ys
        out = jax.vmap(self.ode_to_data)(ys)
        return out


class StabilizedFunc(eqx.Module):
    mlp: eqx.nn.MLP
    A: jnp.ndarray

    def __init__(self, data_size, width_size, depth, *, key, **kwargs):
        super().__init__(**kwargs)
        model_key, A_key = jr.split(key, 2)
        self.mlp = eqx.nn.MLP(
            in_size=data_size,
            out_size=data_size,
            width_size=width_size,
            depth=depth,
            activation=jnn.tanh,
            final_activation=lambda x: x,
            key=model_key,
        )
        self.A = jr.uniform(A_key, (data_size, data_size), minval=-jnp.sqrt(1/3), maxval=jnp.sqrt(1/3))

    def __call__(self, t, y, args):
        return self.mlp(y) + self.A @ y


class LinearFunc(eqx.Module):
    A: jnp.ndarray

    def __init__(self, data_size, *, key, **kwargs):
        super().__init__(**kwargs)
        self.A = jr.uniform(key, (data_size, data_size), minval=-jnp.sqrt(1/3), maxval=jnp.sqrt(1/3))

    def __call__(self, t, y, args):
        return self.A @ y


class StabilizedNeuralODE(eqx.Module):
    cell: eqx.nn.GRUCell
    func: StabilizedFunc
    # hidden_to_ode: eqx.nn.MLP
    # ode_to_data: eqx.nn.MLP
    linear: bool
    

    def __init__(self, data_size, width_size, hidden_size, ode_size, depth, *, key, linear, **kwargs):
        super().__init__(**kwargs)
        self.linear = linear
        rec_key, func_key, linear_func_key, dec_key = jr.split(key, 4)
        self.cell = eqx.nn.GRUCell(data_size, hidden_size, key=rec_key)
        if linear:
            self.func = LinearFunc(ode_size, key=func_key)
        else:
            self.func = StabilizedFunc(data_size, width_size, depth, key=func_key)
        
        # self.hidden_to_ode = eqx.nn.MLP(
        #     in_size=hidden_size,
        #     out_size=ode_size,
        #     width_size=width_size,
        #     depth=1,
        #     activation=jnn.tanh,
        #     key=key,
        # )
        # self.ode_to_data = eqx.nn.MLP(
        #     in_size=ode_size,
        #     out_size=data_size,
        #     width_size=width_size,
        #     depth=1,
        #     activation=jnn.tanh,
        #     final_activation=lambda x: x,
        #     key=key,
        # )
        
    def __call__(self, ts, yi, max_steps=8192):
        # if not self.linear:
        #     hidden = jnp.zeros(self.cell.hidden_size)
        #     for yi_i in yi[::-1]:
        #         hidden = self.cell(yi_i, hidden)
        #     y0 = self.hidden_to_ode(hidden)
        # else:
        y0 = yi[0, :]
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(self.func),
            diffrax.Tsit5(),
            t0=ts[0],
            t1=ts[-1],
            dt0=ts[1] - ts[0],
            y0=y0,
            # stepsize_controller=diffrax.PIDController(rtol=1e-4, atol=1e-6),
            saveat=diffrax.SaveAt(ts=ts),
            max_steps=max_steps
        )
        out = solution.ys
        # if not self.linear:
        #     out = jax.vmap(self.ode_to_data)(out)
        return out
    
    def forward(self, ts, yi):
        hidden = jnp.zeros(self.cell.hidden_size)
        for yi_i in yi[::-1]:
            hidden = self.cell(yi_i, hidden)
        y0 = self.hidden_to_ode(hidden)
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(self.func),
            diffrax.Tsit5(),
            t0=ts[0],
            t1=ts[-1],
            dt0=ts[1] - ts[0],
            y0=y0,
            stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-8),
            saveat=diffrax.SaveAt(ts=ts),
            max_steps=None,
        )
        ys = solution.ys
        out = jax.vmap(self.ode_to_data)(ys)
        return out

def train_NODE(
    data,
    timesteps_per_trial=500,
    t1=5.0,
    width_size=16,
    hidden_size=256,
    ode_size=128,
    depth=2,
    batch_size=256,
    seed=55,
    lr_strategy=(3e-3, 3e-3),
    steps_strategy=(500, 500),
    length_strategy=(0.1, 1),
    plot=True,
    print_every=100,
    seeding_strategy=(0.1, 0.1),
    skip_strategy=(50, 100),
    return_best_model=True,
    use_stabilized_node=True,
    linear=False,
    model=None,
    plot_fn=None,
    mlp_scale=0.2,
    *,
    k
):
    key = jr.PRNGKey(seed)
    data_key, model_key, loader_key = jr.split(key, 3)

    ts = jnp.linspace(0.0, t1, timesteps_per_trial)
    features = data[0].shape[-1]

    if model is None:
        if use_stabilized_node:
            model = StabilizedNeuralODE(features, width_size, hidden_size, ode_size, depth, key=model_key, linear=linear)
        else:
            model = AugmentedNeuralODE(features, width_size, hidden_size, ode_size, depth, key=model_key, scale=mlp_scale)
            
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
            print(ys.shape)
            trials, timesteps, features = ys.shape
            _ts = ts[: int(timesteps)]
            _ys = ys[:, : int(timesteps)]
            seeding_steps = int(timesteps_per_trial * seeding)
    
            start = time.time()
            best_loss = jnp.inf
            for step, (yi,) in zip(
                range(steps), dataloader((_ys,), batch_size, key=loader_key)
            ):
                
    #             yi = jnp.concatenate([jnp.tile(_ts[jnp.newaxis, :, jnp.newaxis], (yi.shape[0], 1, 1)), yi],axis=-1)
                loss, model, opt_state = make_step(_ts, yi, model, opt_state, seeding_steps)
                if loss < best_loss:
                    best_model = copy.deepcopy(model)
                    best_loss = copy.deepcopy(loss)
                if (step % print_every) == 0 or step == steps - 1:
                    end = time.time()
                    

                    print(f"Step: {step}, Loss: {loss}, Current Best Loss: {best_loss}, Computation time: {end - start}")
    
                    if plot:
                        # plt.plot(_ts, _ys[0, :, 0], c="dodgerblue", label="Data")
                        # plt.plot(_ts, _ys[0, :, k+1], c="dodgerblue")
                        if plot_fn is None:
                            ax = plt.subplot(111, projection="3d")
                            ax.scatter(_ys[0, :, 0], _ys[0, :, 1], _ys[0, :, 2], c="dodgerblue", label="Data")
                            
                            model_y = model(_ts, _ys[0, :seeding_steps])
                            
                            # plt.plot(_ts, model_y[:, 0], c="crimson", label="Model")
                            # plt.plot(_ts, model_y[:, k+1], c="crimson")
                            ax.scatter(model_y[:, 0], model_y[:, 1], model_y[:, 2], c="crimson", label="Model")
                            ax.legend()
                            plt.tight_layout()
                            plt.show()
                        else:
                            plot_fn(model, _ts, _ys, seeding_steps)
    
                    start = time.time()
    except KeyboardInterrupt:
        print(f"Exiting early. Returning best model with loss: {best_loss}")
        return ts, ys, best_model
    print(f"Training finished. Returning best model with loss: {best_loss}")
    return ts, ys, best_model