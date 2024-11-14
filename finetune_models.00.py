import os
import argparse
import sys
import glob
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

import jax
import jax.numpy as jnp
from jax import vmap, jit
import jax.random as jr
# set 64-bit mode
# jax.config.update("jax_enable_x64", True)
import equinox as eqx

from scipy import signal, interpolate, stats
from ddfa_node import embed_data, takens_embedding, change_trial_length, split_data, get_aics, get_λs, phaser, stats as statistics, jax_utils
import ddfa_node

from tqdm.auto import tqdm

from scipy.signal import savgol_filter
from ddfa_node.networks.jax_utils import NeuralODE
import diffrax
from diffrax import diffeqsolve, ControlTerm, Euler, MultiTerm, ODETerm, SaveAt, VirtualBrownianTree

import warnings

# ### Make finetuning function
def finetune_model(model, data_tde, subject):
    print(f"Finetuning model for subject {subject}")
    # deserialize the model
    model = eqx.tree_deserialise_leaves("outputs/vdp_model.eqx", model)

    ts, ys, model = jax_utils.train_NODE(
        data_tde,
        model=model,
        timesteps_per_trial=500,
        t1=5.0,
        width_size=128,
        hidden_size=256,
        ode_size=8,
        depth=3,
        batch_size=128,
        seed=6969,
        lr_strategy=(1e-4,),
        steps_strategy=(1500, 30000, 25000),
        length_strategy=(1,),
        skip_strategy=(1,),
        seeding_strategy=(1/5,),
        plot=False,
        print_every=50,
        k=1,
        linear=False,
        plot_fn=None,
        # k=max_power+2
    )
    eqx.tree_serialise_leaves(f"outputs/vdp_model_subject_{subject}.eqx", model)

def main():
    window_length = 50
    polyorder = 5
    data = jnp.load("/home/michael/Synology/Julia/data/VDP_SDEs.npy")[25:-20, ::2, :1]
    feats = data.shape[-1]
    n_trials = data.shape[0]
    new_data = np.zeros((n_trials, data.shape[1], data.shape[2]))
    for trial in range(n_trials):
        new_data[trial, :, :feats] = savgol_filter(data[trial, :, :], window_length=window_length, polyorder=polyorder, axis=0)
        # new_data[trial, :, feats:] = savgol_filter(data[trial, :, :], window_length=window_length, polyorder=polyorder, axis=0, deriv=1)

    # Standardize the data
    new_data = (new_data - jnp.mean(new_data, axis=1)[:, None, :]) / jnp.std(new_data, axis=1)[:, None, :]
    data = new_data
    τ = 32
    k = 4
    data_tde = takens_embedding(data[:, :, :1], τ, k)

    # Initialize the model
    model = NeuralODE(data_size=4, 
                    width_size=128, 
                    hidden_size=256, 
                    ode_size=8, 
                    depth=3, 
                    augment_dims=0, 
                    key=jax.random.PRNGKey(0))

    # deserialize the model
    model = eqx.tree_deserialise_leaves("outputs/vdp_model.eqx", model)
    subject_ids = jnp.repeat(jnp.arange(20), 5)[25:-20]
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=0)
    args = parser.parse_args()
    data_tde = data_tde[subject_ids == args.subject]
    print(data_tde.shape)
    finetune_model(model, data_tde, args.subject)

if __name__ == "__main__":
    main()