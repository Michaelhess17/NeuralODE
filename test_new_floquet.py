from ddfa_node.stability.new_floquet import analyze_eigenvalue_scaling, get_phased_signals
import jax
import jax.numpy as jnp
from tqdm.auto import tqdm
# Global flag to set a specific platform, must be used at startup.
jax.config.update('jax_platform_name', 'cpu')
"""
Test the analyze_eigenvalue_scaling function with synthetic data.
"""
# Generate synthetic data
n_cycles = 1000
n_features = 5
data = jnp.load("outputs/gen_vdp_data.npy")[:3, :1500, :]

trials, nSegments, features = data.shape
trim_cycles = 1
nSegments = 101
height = 0.85
distance = 60
data_arr = []
skipped = []
for a in tqdm(range(trials)):
    try:
        allsigs, phi2 = get_phased_signals(data[a].T, features, trim_cycles, nSegments, height, distance)
        data_arr.append(allsigs)
    except:
        skipped.append(a)
print(f"Skipped {len(skipped)} trials")

max_groups = 4
phases = jnp.arange(nSegments)
# results = jax.vmap(analyze_eigenvalue_scaling, in_axes=(None, None, 0))(data, max_groups, phases)
all_results = [jax.vmap(analyze_eigenvalue_scaling, in_axes=(None, None, 0))(data_arr[i].swapaxes(1, 2), max_groups, phases) for i in range(len(data_arr))]

# Check results
# assert 'x_values' in results
# assert 'y_values' in results
# assert 'weights' in results
# assert 'regression_results' in results
# assert results['x_values'].shape == (nSegments, max_groups - 1)
# assert results['y_values'].shape == (nSegments, max_groups - 1, n_features)
# assert results['weights'].shape == (nSegments, max_groups - 1)

print("Test passed successfully!")
