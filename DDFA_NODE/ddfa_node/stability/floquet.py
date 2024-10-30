import numpy as np
from sklearn.decomposition import PCA
from .. import phaser
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

def get_eigenstuff(s, reps=500, usePCA=False, eigenBasis=None):
    s = s - np.mean(s, axis=0) # Mean center the return map
    s1 = s[:-1] # Intitial values
    s2 = s[1:] # Return values

    n_features = s1.shape[1]
    coeff = np.zeros((n_features, n_features))
    errors = np.zeros((n_features**2, n_features**2))
    r2s = np.zeros(n_features)
    if len(s1) and len(s2):
        for idx in range(n_features):
            mod = sm.OLS(s2[:, idx], s1)

            res = mod.fit()

            coeff[idx, :] = res.params
            errors[idx*(n_features):(idx+1)*n_features, idx*(n_features):(idx+1)*n_features] = res.cov_params()
            r2s[idx] = res.rsquared
    else:
        return (np.nan, np.nan), np.nan, np.nan

    allEigenvals, allEigenvecs = np.zeros((1, n_features), dtype=complex), np.zeros((1, n_features, n_features), dtype=complex)
    eigenvals, eigenvecs = np.linalg.eig(coeff)
    allEigenvals[0, :] = [x for _, x in sorted(zip(np.absolute(eigenvals), eigenvals), reverse=True)]
    inds = np.argsort(-np.absolute(eigenvals))
    allEigenvecs[0, :, :] = eigenvecs[:, inds]
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


def sample_floquet_multipliers(HC_CellArray, nSegments=101, nCovReps=500, phaser_feats=None, splits=range(2, 10), nReplicates=10, usePCA=False, height=0.85, distance=80, vecs=False):
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
        allEigenvals = (np.inf+1j*np.inf)*np.ones((lim, nSplits, max(splits), nReplicates, nSegments, nCovReps, feats), dtype=complex)
        if vecs:
            allEigenvecs = (np.inf+1j*np.inf)*np.ones((lim, nSplits, max(splits), nReplicates, nSegments, nCovReps, feats, feats), dtype=complex)
        else:
            allEigenvecs = []
        allRs = np.empty((lim, nSplits, max(splits), nReplicates, nSegments, feats)); allRs[:] = np.nan
        # allErrors = np.empty((lim, nSplits, max(splits), nReplicates, nSegments, nCovReps)); allErrors[:] = np.nan
    Ns = np.zeros(lim)

    for a in tqdm(range(lim)): # loop through all trials
        try:
            # Prepare data for phaser
            raw = HC_CellArray[a].T # use PC projection of the 1st 6 PCs of the H and Cs to find the cycles
            allsigs, phi2 = get_phased_signals(raw, phaser_feats, trim_cycles, nSegments, height, distance)
            allPhis.append(phi2); Ns[a] = allsigs.shape[0]
            
            # Use floquet analysis to obtain eigenstuff for each pct, replicate, and phase segment
            for split_idx, split in enumerate(splits):
                for split_idx2 in range(split):
                    for rep_idx in range(nReplicates):
                        # print(split_idx)
                        data = np.array_split(random.sample(list(allsigs), allsigs.shape[0]), split)
                        for phase_idx in range(nSegments):
                            s = data[split_idx2][:, :, phase_idx] # Grab phase of interest
                            evecs = None
                            if usePCA:
                                # s = pca.fit_transform(s)
                                s = get_PCA(s)

                            (eigenvals, eigenvecs), r2 = get_eigenstuff(s, reps=nCovReps, usePCA=usePCA, eigenBasis=evecs)

                            # allEigenvals[a, split_idx, split_idx2, rep_idx, phase_idx, :] = sorted(eigenvals, reverse=True)
                            if eigenvals is not np.nan:
                                allEigenvals[a, split_idx, split_idx2, rep_idx, phase_idx, :, :] = eigenvals
                                
                                if vecs:
                                    allEigenvecs[a, split_idx, split_idx2, rep_idx, phase_idx, :, :, :] = eigenvecs
                                # [x for _, x in sorted(zip(np.absolute(eigenvals), eigenvecs), reverse=True)]
                                allRs[a, split_idx, split_idx2, rep_idx, phase_idx, :] = r2
                                # allErrors[a, split_idx, split_idx2, rep_idx, phase_idx] = error
            
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
    return allEigenvals, allEigenvecs, allRs, allPhis, Ns#, allErrors


def backtrace_multipliers(splits, eigVals, Ns, subject=0, nPoints=5, phase=50, eig=0, plot=True, plot_title=None, ax=None):
    """ The inputs pcts, eigVals, and Ns should be those used or outputted from the sample_floquet_multipliers function """ 

    splits=splits[:nPoints]
    b = np.absolute(eigVals[subject, :nPoints, :, :, phase, :, eig]).squeeze()
    if b.max() == np.nan and b.min() == np.nan:
        return np.nan, np.nan
    mean = [] #np.mean(b_nonan, axis=1)
    std = []
    
    X = []; y = []
    
    X = 1/Ns[subject]/splits
    for idx in range(len(splits)):
        
        c = b[idx, :splits[idx]]
        std.append(np.std(c.flatten()))
        mean.append(c.mean())
        if not np.isnan(c).sum() == 0:
            # print(c.shape, np.isnan(c).sum(), c)
            return np.nan, np.nan
        # X.append([a]*len(c.flatten()))
        # y.append(c.flatten())
 
    # fit = linregress(X, y)
    coeffs, cov = np.polyfit(X, mean, deg=1, w=1/np.array(std), cov=True)
    fit = np.poly1d(coeffs)
    

    X_test = np.linspace(0, max(1/Ns[subject]/splits), 20)
    # print(max(1/(Ns[subject]/splits)), X_test)
    # y_pred = fit.slope*X_test.reshape(-1, 1) + fit.intercept
    y_pred = fit(X_test)
    int_err = np.sqrt(np.diag(np.abs(cov)))[-1]
    
    if plot:
        if ax is None:
            plt.figure()
            plt.scatter(1/Ns[subject]/splits, mean)
            plt.errorbar(1/Ns[subject]/splits, mean, std, capsize=5, fmt='o')
            plt.plot(X_test, y_pred, c='grey')
            # plt.scatter(0, fit.intercept, c='r')
            # plt.errorbar(0, fit.intercept, fit.intercept_stderr, capsize=5, fmt='o', c='r')
            plt.scatter(0, coeffs[-1], c='r')
            plt.errorbar(0, coeffs[-1], int_err, capsize=5, fmt='o', c='r')
            plt.xlabel("$\\frac{1}{N}$")
            plt.ylabel(r"$\lambda$")
            plt.title(plot_title)
        else:
            ax.scatter(1/Ns[subject]/splits, mean, c='k')
            ax.errorbar(1/Ns[subject]/splits, mean, std, capsize=5, fmt='o', c='k')
            ax.plot(X_test, y_pred, c='grey')
            # ax.scatter(0, fit.intercept, c='r')
            # ax.errorbar(0, fit.intercept, fit.intercept_stderr, capsize=5, fmt='o', c='r')
            ax.scatter(0, coeffs[-1], c='r')
            ax.errorbar(0, coeffs[-1], int_err, capsize=5, fmt='o', c='r')
            ax.set_xlabel("$\\frac{1}{N}$")
            ax.set_ylabel(r"$\lambda$")
            ax.set_title(plot_title)
        plt.tight_layout()
    # return fit.intercept, fit.intercept_stderr
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