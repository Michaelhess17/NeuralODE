import numpy as np
from sklearn.decomposition import PCA
from . import phaser
from scipy.signal import find_peaks
from scipy.stats import sem
from sklearn.linear_model import LinearRegression
from resample.bootstrap import bootstrap
import traceback
# from tqdm import tqdm
# from utils.load_Carey_data import moving_average
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import linregress
import random
import warnings
import statsmodels.api as sm
from tqdm import tqdm

import jax
import jax.numpy as jnp
from jax import random, jit, vmap
from functools import partial
from jax.scipy.linalg import eigh

def get_phased_signals(raw, phaser_feats=None, trim_cycles=1, nSegments=101, height=0.85, distance=80):
    feats = raw.shape[0]
    if phaser_feats is None:
        phaser_feats = feats
    raw = raw - raw.mean(axis=1)[:, np.newaxis]
    dats = [raw[:phaser_feats]]

    # Assess phase using Phaser    
    phr = phaser.Phaser(dats)
    phi = [ phr.phaserEval( d ) for d in dats ] # extract phase
    phi2  = (phi[0].T % (2*np.pi))/np.pi-1;
    peaks = find_peaks(phi2.ravel(), height=height, distance=distance)

    # Find cycle start and stop locations
    peak_arr = np.asarray(peaks, dtype="object")
    peaks_start = peak_arr[0][:] # index is the trial #
    peaks_end = peaks_start

    # ensure full gait cycles
    peaks_end = peaks_end[trim_cycles:]
    peaks_start = peaks_start[:-trim_cycles]

    # Use all features for Poincare map fitting
    dats = [raw]

    # For each cycle extract the PCs, and interpolate the HC data so we can make a new phase axis
    nPeaks = len(peaks_start)
    allsigs = np.zeros((nPeaks, feats, nSegments))
    for d in range(nPeaks): 
        sig = np.zeros((feats, nSegments))
        y = dats[0][:,peaks_start[d]:peaks_end[d]]
        x = np.linspace(-1, 1, num=y.shape[1]) # set axis for gait segment
        for idx in range(feats):
            xnew = np.linspace(-1,1, num=nSegments,endpoint=True) #set phase axis
            f2 = np.interp(xnew, x, y[idx, :]) # don't use scipy here
            sig[idx, :] = f2 # will hold all interpolated PCs
            
        allsigs[d] = sig
        
    return allsigs, phi2

# @jit
def get_eigenstuff(s, reps=500, usePCA=False, eigenBasis=None):
    s = s - jnp.mean(s, axis=0)  # Mean center the return map
    s1 = s[:-1]  # Initial values
    s2 = s[1:]  # Return values

    n_features = s1.shape[1]
    coeff = jnp.zeros((n_features, n_features))
    errors = jnp.zeros((n_features, n_features, n_features))
    r2s = jnp.zeros(n_features)
    if s1.shape[0] > 0 and s2.shape[0] > 0:
        X = jnp.hstack([s1, jnp.ones((s1.shape[0], 1))])  # Add intercept
        for idx in range(n_features):
            beta, residuals, rank, s = jnp.linalg.lstsq(X, s2[:, idx], rcond=None)
            coeff.at[idx, :].set(beta[:-1])  # Exclude intercept
            # Calculate R-squared
            ss_tot = jnp.sum((s2[:, idx] - jnp.mean(s2[:, idx]))**2)
            ss_res = jnp.sum(residuals)
            r2 = jnp.where(ss_tot == 0, 0, 1 - ss_res / ss_tot)
            r2s.at[idx].set(r2)
            # Calculate covariance matrix of the parameters
            # Using the formula Cov(b) = sigma^2 * (X'X)^-1
            # where sigma^2 is the variance of the residuals
            sigma_squared = ss_res / (s1.shape[0] - n_features)
            XTX_inv = jnp.linalg.inv(jnp.dot(X.T, X))
            cov_params = sigma_squared * XTX_inv
            errors.at[idx].set(cov_params[:-1, :-1])  # Exclude intercept terms
    else:
        return (jnp.nan, jnp.nan), jnp.nan

    allEigenvals, allEigenvecs = jnp.zeros((1, n_features), dtype=complex), jnp.zeros((1, n_features, n_features), dtype=complex)
    eigenvals, eigenvecs = jnp.linalg.eig(coeff)

    # Replace the Python sorting with JAX-compatible operations
    sorted_indices = jnp.argsort(jnp.abs(eigenvals))[::-1]  # Sort in descending order
    allEigenvals = eigenvals[sorted_indices]
    allEigenvals = allEigenvals.reshape(1, -1)  # Reshape to match original shape
    
    # Sort eigenvectors according to the same ordering
    eigenvecs = eigenvecs[:, sorted_indices]
    allEigenvecs = eigenvecs  # Reshape to match original shape
    # for idx in range(reps):
    #     try:
    #         sample_coeff = np.random.multivariate_normal(coeff.flatten(), errors).reshape(n_features, n_features)
    #         eigenvals, eigenvecs = np.linalg.eig(sample_coeff)
    #         allEigenvals[idx, :] = [x for _, x in sorted(zip(np.absolute(eigenvals), eigenvals), reverse=True)]
    #         inds = np.argsort(-np.absolute(eigenvals))
    #         allEigenvecs[idx, :, :] = eigenvecs[:, inds]
    #     except np.linalg.LinAlgError:
    #         print(coeff.flatten(), "\n", errors)

    # if usePCA:
    #     coeff = eigenBasis @ coeff @ eigenBasis.T

    # r2 = lr.score(s1, s2)
    return (allEigenvals, allEigenvecs), r2s#, total_error

def fitreg(data):
    feats = data.shape[1] // 2
    X, Y = data[:, :feats], data[:, feats:]
    lm = LinearRegression(fit_intercept=False)
    lm.fit(X, Y)
    return lm.coef_

def get_eigenstuff_bootstrap(s1, s2, reps=5000, usePCA=False, eigenBasis=None):
    feats = s1.shape[1]
    data = np.hstack([s1, s2])
    boot_coef = bootstrap(sample=data, fn=fitreg, size=reps)        

    allEigenvals, allEigenvecs = np.zeros((reps, feats), dtype=complex), np.zeros((reps, feats, feats), dtype=complex)
    for idx in range(reps):
        eigenvals, eigenvecs = np.linalg.eig(boot_coef[idx])
        allEigenvals[idx, :] = [x for _, x in sorted(zip(np.absolute(eigenvals), eigenvals), reverse=True)]
        inds = np.argsort(-np.absolute(eigenvals))
        allEigenvecs[idx, :, :] = eigenvecs[:, inds]

    return (allEigenvals, allEigenvecs), []

def process_phase(s_data, phase_idx, usePCA, nCovReps):
    s = s_data[:, :, phase_idx]
    if usePCA:
        s = get_PCA(s)
    (eigenvals, eigenvecs), r2 = get_eigenstuff(s, reps=nCovReps, usePCA=usePCA)
    return eigenvals, eigenvecs, r2

# Create a JIT-compatible function that works with a single array
# @partial(jax.jit, static_argnames=('nSegments', 'usePCA', 'nCovReps'))
def process_single_array(array_slice, nSegments, usePCA, nCovReps):
    def scan_fn(carry, phase_idx):
        return carry, process_phase(array_slice, phase_idx, usePCA, nCovReps)
    
    _, results = jax.lax.scan(
        scan_fn,
        init=None,
        xs=jnp.arange(nSegments)
    )
    return results

# Main function
def sample_floquet_multipliers(HC_CellArray, nSegments, nCovReps, phaser_feats, splits, 
                             nReplicates, usePCA, height, distance, seed=0):
    trim_cycles = 1
    feats = HC_CellArray[0].shape[1] # number of PCs to extract for phase averaging
    if phaser_feats is None:
        phaser_feats = feats
    nSplits = len(splits)
    lim = len(HC_CellArray)
    pca = PCA(n_components = feats - 1)
    
    allPhis = []
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="invalid value encountered in multiply")
        allEigenvals = (jnp.inf+1j*jnp.inf)*jnp.ones((lim, nSplits, max(splits), nReplicates, nSegments, nCovReps, feats), dtype=complex)
        allEigenvecs = (jnp.inf+1j*jnp.inf)*jnp.ones((lim, nSplits, max(splits), nReplicates, nSegments, nCovReps, feats, feats), dtype=complex)
        allRs = jnp.empty((lim, nSplits, max(splits), nReplicates, nSegments, feats)); allRs.at[:].set(jnp.nan)
        # allErrors = np.empty((lim, nSplits, max(splits), nReplicates, nSegments, nCovReps)); allErrors[:] = np.nan
    Ns = jnp.zeros(lim)
    for a in tqdm(range(lim)): # loop through all trials
        # Create vmapped version of process_split_rep for fixed split sizes
        for split_idx, split in enumerate(splits):
            try:
                # Prepare data for phaser
                raw = HC_CellArray[a].T # use PC projection of the 1st 6 PCs of the H and Cs to find the cycles
                allsigs, phi2 = get_phased_signals(raw, phaser_feats, trim_cycles, nSegments, height, distance)
                allPhis.append(phi2); Ns.at[a].set(allsigs.shape[0])
                
                # Create data splits once per outer loop
                data_splits = jnp.array_split(allsigs, split)
                
                # Vmap over split_idx2 and rep_idx with fixed shapes
                for split_idx2 in range(split):
                    current_split = data_splits[split_idx2]
                                    
                    # Run computation for this split
                    split_idx2_range = jnp.arange(split)
                    rep_idx_range = jnp.arange(nReplicates)
                    results = process_single_array(current_split, nSegments, usePCA, nCovReps)

                # Update the arrays
                for split_idx, split_result in enumerate(results):
                    for split_idx2, rep_result in enumerate(split_result):
                        for rep_idx, phase_result in enumerate(rep_result):
                            print(phase_result)
                            eigenvals, eigenvecs, r2 = phase_result
                            if not jnp.isnan(eigenvals).all():
                                allEigenvals = allEigenvals.at[a, split_idx, split_idx2, rep_idx, :, :, :].set(eigenvals)
                                allEigenvecs = allEigenvecs.at[a, split_idx, split_idx2, rep_idx, :, :, :, :].set(eigenvecs)
                                allRs = allRs.at[a, split_idx, split_idx2, rep_idx, :, :].set(r2)
            except Exception as e:
                if str(e) == "('newPhaser:emptySection', 'Poincare section is empty -- bailing out')":
                    print(a, " failed")
                    continue
                elif 'computeOffset' in str(e):
                    print(a)
                    continue
                else:
                    print(a, split_idx, split_idx2, rep_idx, phase_idx)
                    print(e)
                    print(traceback.format_exc())
                    continue
    return allEigenvals, allEigenvecs, allRs, allPhis, Ns

def backtrace_multipliers(splits, eigVals, Ns, subject=0, nPoints=5, phase=50, eig=0, plot=True, plot_title=None, ax=None):
    """ The inputs pcts, eigVals, and Ns should be those used or outputted from the sample_floquet_multipliers function """ 

    splits=splits[:nPoints]
    b = jnp.absolute(eigVals[subject, :nPoints, :, :, phase, :, eig]).squeeze()
    if jnp.isnan(b).all():
        return jnp.nan, jnp.nan
    mean = [] 
    std = []
    
    X = 1/Ns[subject]/splits
    for idx in range(len(splits)):
        
        c = b[idx, :splits[idx]]
        std.append(jnp.std(c.flatten()))
        mean.append(c.mean())
        if jnp.isnan(c).sum() != 0:
            return jnp.nan, jnp.nan
 
    coeffs, cov = jnp.polyfit(X, mean, deg=1, w=1/jnp.array(std), cov=True)
    fit = jnp.poly1d(coeffs)
    
    X_test = jnp.linspace(0, max(1/Ns[subject]/splits), 20)
    y_pred = fit(X_test)
    int_err = jnp.sqrt(jnp.diag(jnp.abs(cov)))[-1]
    
    if plot:
        if ax is None:
            plt.figure()
            plt.scatter(1/Ns[subject]/splits, mean)
            plt.errorbar(1/Ns[subject]/splits, mean, std, capsize=5, fmt='o')
            plt.plot(X_test, y_pred, c='grey')
            plt.scatter(0, coeffs[-1], c='r')
            plt.errorbar(0, coeffs[-1], int_err, capsize=5, fmt='o', c='r')
            plt.xlabel("$\\frac{1}{N}$")
            plt.ylabel(r"$\lambda$")
            plt.title(plot_title)
        else:
            ax.scatter(1/Ns[subject]/splits, mean, c='k')
            ax.errorbar(1/Ns[subject]/splits, mean, std, capsize=5, fmt='o', c='k')
            ax.plot(X_test, y_pred, c='grey')
            ax.scatter(0, coeffs[-1], c='r')
            ax.errorbar(0, coeffs[-1], int_err, capsize=5, fmt='o', c='r')
            ax.set_xlabel("$\\frac{1}{N}$")
            ax.set_ylabel(r"$\lambda$")
            ax.set_title(plot_title)
        plt.tight_layout()
    return coeffs[-1], int_err

# def get_floquet_multipliers_PCA():
    

def get_PCA(s):
    s -= np.mean(s, axis = 0)

    cov = np.cov(s, rowvar = False)

    evals , evecs = np.linalg.eig(cov)

    idx = np.argsort(evals)[::-1]
    evecs = evecs[:,idx]
    evals = evals[idx]

    s = np.dot(s, evecs)
    return s



def get_floquet_multipliers(HC_CellArray, nSegments, phaser_feats=None, usePCA=False):
    trim_cycles = 1
    feats = HC_CellArray[0].shape[1] # number of PCs to extract for phase averaging
    lim = len(HC_CellArray)
    pca = PCA(n_components=feats-1)
    
    # initialize arrays
    # allEigenvals = np.zeros((lim, nSegments, feats if not usePCA else feats - 1), dtype=complex)
    # allEigenvecs = np.zeros((lim, nSegments, feats if not usePCA else feats - 1, feats if not usePCA else feats - 1), dtype=complex)
    allEigenvals = np.zeros((lim, nSegments, feats), dtype=complex)
    allEigenvecs = np.zeros((lim, nSegments, feats, feats), dtype=complex)
    allRs = np.zeros((lim, nSegments))
    Ns = np.zeros(lim)
    allPhis = []

    for a in tqdm(range(lim)): # loop through all trials
        try:
            raw = HC_CellArray[a].T # use PC projection of the 1st 6 PCs of the H and Cs to find the cycles            
            allsigs, phi2 = get_phased_signals(raw, phaser_feats, trim_cycles, nSegments)
            allPhis.append(phi2); Ns[a] = allsigs.shape[0]
            # Use floquet analysis to obtain eigenstuff
            
            for phase_idx in range(nSegments):

                s = allsigs[:, :, phase_idx]
                s = (s - np.mean(s, axis = 0))/(np.std(s, axis = 0))
                evecs = None
                
                if usePCA:
                    # s = pca.fit_transform(s)
                    s -= np.mean(s, axis = 0)

                    cov = np.cov(s, rowvar = False)

                    evals , evecs = np.linalg.eig(cov)

                    idx = np.argsort(evals)[::-1]
                    evecs = evecs[:,idx]
                    evals = evals[idx]

                    s = np.dot(s, evecs.real)
                    
                (eigenvals, eigenvecs), r2 = get_eigenstuff(s, usePCA=usePCA, eigenBasis=evecs)
                
                allEigenvals[a, phase_idx, :] = sorted(eigenvals, reverse=True)
                allEigenvecs[a, phase_idx, :, :] = [x for _, x in sorted(zip(eigenvals, eigenvecs), reverse=True)]
                allRs[a, phase_idx] = r2

            # print(f"\n Index: {a} Mean: {np.mean(phase_eigs, axis=0)}, St.Dev.: {np.std(phase_eigs, axis=0)}, R^2: {np.mean(r2s)}")
        except Exception as e:
            print(str(e), 'computeOffset' in str(e))
            if str(e) == "('newPhaser:emptySection', 'Poincare section is empty -- bailing out')":
                print(a)
                continue
            elif 'computeOffset' in str(e):
                print(a)
                continue
            else:
                # print(a)
                # print(e)
                print(traceback.format_exc())

    allPhis = np.array(allPhis)
    
    return allEigenvals, allEigenvecs, allRs, allPhis, Ns

def sample_floquet_multipliers_by_pct(HC_CellArray, nSegments=101, phaser_feats=None, pcts=np.arange(1.0, 0.05, -0.05), nReplicates=10, usePCA=False):
    trim_cycles = 1
    feats = HC_CellArray[0].shape[1] # number of PCs to extract for phase averaging
    if phaser_feats is None:
        phaser_feats = feats
    nPcts = len(pcts)
    lim = len(HC_CellArray)
    pca = PCA(n_components = feats - 1)
    
    allPhis = []
    
    allEigenvals = np.empty((lim, nPcts, nReplicates, nSegments, feats), dtype=complex)
    allEigenvecs = np.empty((lim, nPcts, nReplicates, nSegments, feats, feats), dtype=complex)
    allRs = np.empty((lim, nPcts, nReplicates, nSegments))
    allErrors = np.empty((lim, nPcts, nReplicates, nSegments))
    Ns = np.empty(lim)
    allEigenvals[:], allEigenvecs[:], allRs[:], allErrors[:], Ns[:] = np.nan, np.nan, np.nan, np.nan, np.nan

    for a in tqdm(range(lim)): # loop through all trials
        try:
            # Prepare data for phaser
            raw = HC_CellArray[a].T # use PC projection of the 1st 6 PCs of the H and Cs to find the cycles
            allsigs, phi2 = get_phased_signals(raw, phaser_feats, trim_cycles, nSegments)
            allPhis.append(phi2); Ns[a] = allsigs.shape[0]
            
            # Use floquet analysis to obtain eigenstuff for each pct, replicate, and phase segment
            for pct_idx, pct in enumerate(pcts):
                for rep_idx in range(nReplicates):
                    for phase_idx in range(nSegments):
                        total = allsigs.shape[0]
                        samps = int(total * pct)
                        inds = sorted(np.random.choice(range(total), size=samps, replace=False)) # Randomly sample return pairs

                        s = allsigs[inds, :, phase_idx] # Grab phase of interest
                        evecs = None
                        if usePCA:
                            # s = pca.fit_transform(s)
                            s -= np.mean(s, axis = 0)

                            cov = np.cov(s, rowvar = False)

                            evals , evecs = np.linalg.eig(cov)

                            idx = np.argsort(evals)[::-1]
                            evecs = evecs[:,idx]
                            evals = evals[idx]

                            s = np.dot(s, evecs)
                    
                        (eigenvals, eigenvecs), r2, error = get_eigenstuff(s, usePCA=usePCA, eigenBasis=evecs)
                        if eigenvals is not np.nan:
                            allEigenvals[a, pct_idx, rep_idx, phase_idx, :] = sorted(eigenvals, reverse=True)
                            allEigenvecs[a, pct_idx, rep_idx, phase_idx, :, :] = [x for _, x in sorted(zip(eigenvals, eigenvecs), reverse=True)]
                            allRs[a, pct_idx, rep_idx, phase_idx] = r2
                            allErrors[a, pct_idx, rep_idx, phase_idx] = error
            
        except Exception as e:
            if str(e) == "('newPhaser:emptySection', 'Poincare section is empty -- bailing out')":
                print(a, " failed")
                continue
            elif 'computeOffset' in str(e):
                print(a)
                continue
            else:
                # print(a)
                # print(e)
                # print(traceback.format_exc())
                break

    return allEigenvals, allEigenvecs, allRs, allPhis, Ns, allErrors

def backtrace_multipliers_pct(pcts, eigVals, Ns, subject=0, nPoints=5, phase=50, eig=0, plot=True, plot_title=None, ax=None):
    """ The inputs pcts, eigVals, and Ns should be those used or outputted from the sample_floquet_multipliers function """ 
    pcts=pcts[:nPoints]
    b = np.absolute(eigVals[subject, :nPoints, :, phase, eig])
    mean, std = b.mean(axis=1), sem(b, axis=1)
    
    lr = LinearRegression()
    X = np.array([[a]*b.shape[1] for a in 1/(pcts*Ns[subject])]).flatten()# .reshape(-1, 1)
    y = b.flatten()# .reshape(-1, 1)

    fit = linregress(X, y)

    X_test = np.linspace(0, max(1/(pcts*Ns[subject])), 20)
    y_pred = fit.slope*X_test.reshape(-1, 1) + fit.intercept
    
    if plot:
        if ax is None:
            plt.figure()
            plt.scatter(1/(pcts*Ns[subject]), mean)
            plt.errorbar(1/(pcts*Ns[subject]), mean, std, capsize=5, fmt='o')
            plt.plot(X_test, y_pred, c='grey')
            plt.scatter(0, lr.intercept_[0], c='r')
            plt.errorbar(1/(pcts*Ns[subject]), mean, std, capsize=5, fmt='o', c='k')
            plt.xlabel("$\\frac{1}{N}$")
            plt.ylabel(r"$\lambda$")
            plt.title(plot_title)
        else:
            ax.scatter(1/(pcts*Ns[subject]), mean)
            ax.errorbar(1/(pcts*Ns[subject]), mean, std, capsize=5, fmt='o')
            ax.plot(X_test, y_pred, c='grey')
            ax.scatter(0, fit.intercept, c='r')
            ax.errorbar(1/(pcts*Ns[subject]), mean, std, capsize=5, fmt='o', c='k')
            ax.set_xlabel("$\\frac{1}{N}$")
            ax.set_ylabel(r"$\lambda$")
            ax.set_title(plot_title)
        plt.tight_layout()
    return fit.intercept, fit.intercept_stderr