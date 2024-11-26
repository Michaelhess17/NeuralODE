import jax
import jax.numpy as jnp
import diffrax
from diffrax import VirtualBrownianTree, MultiTerm, ODETerm, ControlTerm
import matplotlib.pyplot as plt
from typing import Tuple
import equinox as eqx

# Set random seed for reproducibility
key = jax.random.PRNGKey(42)

class VanDerPol(eqx.Module):
    mu: float
    noise_scale: float

    def __init__(self, mu: float, noise_scale: float = 0.125):
        self.mu = mu
        self.noise_scale = noise_scale

    def drift(self, t, y, args):
        # y[0] is position, y[1] is velocity
        return jnp.array([
            y[1],
            self.mu * (1 - y[0]**2) * y[1] - y[0]
        ])

    def diffusion(self, t, y, args):
        # Additive noise
        return jnp.array([[self.noise_scale], [self.noise_scale]])

def simulate_single_oscillator(
    key: jax.random.PRNGKey,
    mu: float,
    initial_condition: jnp.ndarray,
    t_span: Tuple[float, float],
    dt: float
) -> jnp.ndarray:
    
    oscillator = VanDerPol(mu=mu)
    ts = jnp.arange(t_span[0], t_span[1], dt)
    
    # Define the terms for the SDE
    drift_term = ODETerm(oscillator.drift)
    diffusion_term = ControlTerm(
        oscillator.diffusion, 
        VirtualBrownianTree(
            key=key, 
            t0=t_span[0], 
            t1=t_span[1], 
            tol=1e-3, 
            shape=(1,)
        )
    )
    
    # Combine terms and solve
    terms = MultiTerm(drift_term, diffusion_term)
    solver = diffrax.Euler()
    solution = diffrax.diffeqsolve(
        terms,
        solver,
        t0=t_span[0],
        t1=t_span[1],
        dt0=dt,
        y0=initial_condition,
        saveat=diffrax.SaveAt(ts=ts),
        max_steps=None
    )
    
    return solution.ys

@eqx.filter_jit
def simulate_oscillators(
    key: jax.random.PRNGKey,
    n_oscillators: int,
    mu_range: Tuple[float, float],
    t_span: Tuple[float, float],
    n_replicates: int,
    dt: float
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    
    # Generate random mu values and initial conditions
    key1, key2, key3 = jax.random.split(key, 3)
    mus = jnp.linspace(mu_range[0], mu_range[1], n_oscillators, dtype=jnp.float32)
    initial_conditions = 2*jax.random.normal(key2, shape=(n_replicates, n_oscillators, 2))
    
    # Generate simulation keys for each oscillator
    sim_keys = jax.random.split(key3, n_oscillators)
    
    # Time points
    ts = jnp.arange(t_span[0], t_span[1], dt)
    
    # Vectorize the simulation function across oscillators
    vectorized_sim = jax.vmap(simulate_single_oscillator, in_axes=(0, 0, 0, None, None))

    # Vectorize the simulation function across initial conditions
    vectorized_sim_replicates = jax.vmap(vectorized_sim, in_axes=(None, None, 0, None, None))
    # Run all simulations in parallel
    solutions = vectorized_sim_replicates(sim_keys, mus, initial_conditions, t_span, dt)
    
    return ts, solutions, mus

def main():

    # Simulation parameters
    n_oscillators = 30  # Now we can easily handle more oscillators!
    mu_range = (1.0, 3.0)
    t_span = (0., 120.)
    n_replicates = 10
    dt = 0.02

    key = jax.random.PRNGKey(42)
    ts, solutions, mus = simulate_oscillators(key, n_oscillators, mu_range, t_span, n_replicates, dt)


    # Plot results
    plt.figure(figsize=(12, 8))

    # Phase space trajectories
    plt.subplot(121)
    for i in range(n_replicates):
        for j in range(n_oscillators):
            plt.plot(solutions[i, j, :, 0], solutions[i, j, :, 1], 
                label=f'μ={mus[j]:.2f}')
    plt.xlabel('Position')
    plt.ylabel('Velocity')
    plt.title('Phase Space Trajectories')
    plt.legend()
    plt.grid(True)

    # Time series
    plt.subplot(122)
    for i in range(n_replicates):
        for j in range(n_oscillators):
            plt.plot(ts, solutions[i, j, :, 0], 
                    label=f'μ={mus[j]:.2f}')
    plt.xlabel('Time')
    plt.ylabel('Position')
    plt.title('Position vs Time')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("figures/oscillators.png")


    # Clear the figure
    plt.clf()
    plt.figure(figsize=(12, 8))

    # Save the solutions and Floquet multipliers

    mus = -mus.astype(jnp.complex64)
    eigs = lambda x: (((x + jnp.sqrt(x ** 2 - 4)) / 2), ((x - jnp.sqrt(x ** 2 - 4)) / 2))
    FMs_1 = jnp.exp(eigs(mus)[0])
    FMs_2 = jnp.exp(eigs(mus)[1])
    FMs = jnp.stack([FMs_1, FMs_2], axis=1)

    plt.scatter(mus, jnp.abs(FMs[:, 0]), label="FM 1")
    plt.scatter(mus, jnp.abs(FMs[:, 1]), label="FM 2")
    plt.legend()
    plt.show()
    plt.savefig("figures/VDP_FMs.png")

    jnp.save("outputs/VDP_oscillators.npy", solutions)
    jnp.save("outputs/VDP_FMs.npy", FMs)
    return

if __name__ == "__main__":
    main()