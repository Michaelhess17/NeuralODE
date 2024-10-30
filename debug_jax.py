# %%
import os
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import jax
import jax.numpy as jnp
from ddfa_node import sample_floquet_multipliers, backtrace_multipliers
import matplotlib.pyplot as plt

# %%
generated_data = jnp.load("outputs/gen_human_data.npy")
print(generated_data.shape)


# %%
# sfm_vmap = jax.vmap(lambda x: 
sample_floquet_multipliers(generated_data, 
                            nSegments=101, 
                            nCovReps=1,
                            phaser_feats=None, 
                            splits=jnp.array(range(2, 5)), 
                            nReplicates=300, 
                            usePCA=False, 
                            height=0.85, 
                            distance=100)
                    # in_axes=(0,))


# outputs = sfm_vmap(generated_data)
# allEigenvals, allEigenvecs, allRs, allPhis, Ns = outputs

# %%
backtrace_multipliers(splits, eigVals, Ns, subject=0, nPoints=5, phase=50, eig=0, plot=True, plot_title=None, ax=None)


