import numpy as np
import os
import matplotlib.pyplot as plt
# from rpy2.robjects.packages import importr
# from rpy2.robjects import numpy2ri
# import rpy2.robjects as ro

# nonlinearTseries = importr("nonlinearTseries")

def get_autocorr_1_e_time(x, threshold=1/np.e, maxlags=100):
    lags, autocorr, _, _ = plt.acorr(x, maxlags=maxlags)
    plt.close()
    # only look at positive lags
    lags, autocorr = lags[lags>0], autocorr[lags>0]

    # Find the first time autocorrelation goes below the threshold
    first_below_threshold = np.nan
    for lag, ac in zip(lags, autocorr):
        if ac < threshold:
            first_below_threshold = lag
            break
    if first_below_threshold is np.nan:
        print("No 1/e time detected. Increase maxlags parameter!")
        
    return first_below_threshold

def get_best_1_e_time(data, threshold=1/np.e, maxlags=100):
    trials, timesteps, features = data.shape
    e_times = np.array([[get_autocorr_1_e_time(data[x, :, y], threshold=1/np.e, maxlags=100) for x in range(trials)] for y in range(features)])
    return np.nanmedian(e_times)

"""
False Nearest Neighbors (FNN) for dimension (n).
==================================================


"""

def ts_recon(ts, dim, tau):
    import numpy as np
    xlen = len(ts)-(dim-1)*tau
    a = np.linspace(0, xlen-1, xlen)
    a = np.reshape(a, (xlen, 1))
    delayVec = np.linspace(0, (dim-1), dim)*tau
    delayVec = np.reshape(delayVec, (1, dim))
    delayMat = np.tile(delayVec, (xlen, 1))
    vec = np.tile(a, (1, dim))
    indRecon = np.reshape(vec, (xlen, dim)) + delayMat
    indRecon = indRecon.astype(np.int64)
    tsrecon = ts[indRecon]
    tsrecon = tsrecon[:, :, 0]
    return tsrecon


def find_neighbors(dist_array, ind_array, wind):
    import numpy as np
    neighbors = []
    dist = []
    for i, pt in enumerate(ind_array):
        if len(pt[np.abs(pt[0]-pt) > wind]) > 0:
            j = np.min(pt[np.abs(pt[0]-pt) > wind])
        else:
            j = pt[1]

        neighbors.append([pt[0], j])
        dist.append([dist_array[0][0], dist_array[i][np.where(pt==j)[0][0]]])
    return np.array(dist), np.array(neighbors)


def FNS_fraction(i, strand_ind, tsrecon, s_ind_m, s_dm, ind, Rtol, Stol, st_dev):
    # Compute the false nearest strands fraction using the composite fraction false nearest strands algorithm by David Chelidze 2017.

    import numpy as np
    starting_index = 0
    epsilon_k = 0
    delta_k = 0
    num1 = []
    num2 = []
    for k in range(0, len(strand_ind)):
        strand_norm = np.linalg.norm(s_dm[k])
        if strand_norm > 0 and starting_index < len(tsrecon):
            epsilon_k = np.linalg.norm(tsrecon[s_ind_m[k], :] - tsrecon[ind, :][starting_index:starting_index+len(s_ind_m[k])])/strand_norm
            delta_k = np.sum(abs(tsrecon[s_ind_m[k], :][:,i-1] - tsrecon[ind, :][starting_index:starting_index+len(s_ind_m[k])][:,i-1]))/strand_norm
        # Criteria 1
        num1.append(np.heaviside(delta_k - epsilon_k*Rtol, 0.5))
        # Criteria 2
        num2.append(np.heaviside(delta_k - Stol*st_dev, 0.5))

        starting_index += len(s_ind_m[k])

    den = len(strand_ind)
    num = sum(np.logical_or(num1, num2))
    fns_frac = (num/den)*100
    return fns_frac 

def compute_strands(xlen2, D, IDX):
    # Function to allocate the points onto nearest neighbor strands using the strand algorithm (Chelidze 2017).

    import numpy as np
    strand_ind = []
    s_ind = []
    s_ind_m = []
    s_dm = []
    #############################################################################
    # Loop through all points
    for rnum, row in enumerate(IDX[0:xlen2]):
        sort_status = 0
        # Loop through all strands
        for k in range(len(strand_ind)):
            # Test if point fits in any current strands, otherwise continue searching
            if row[1] in k + np.array(strand_ind[k])[:,1]:
                strand_ind[k].append(row)
                sort_status = 1
                break
        # If item is not allocated to strand, make new strand
        if not sort_status:
            strand_ind.append([])
            strand_ind[-1].append(row)
    #############################################################################

    # Assign distances to strands
    for k in range(len(strand_ind)):
        s_ind.append(np.array(strand_ind[k])[:,-1]<=xlen2-1)
        s_ind_m.append(np.array(strand_ind[k])[:,-1][s_ind[k]])
        s_dm.append(D[s_ind_m[k]][:, -1])
    
    return strand_ind, s_ind, s_ind_m, s_dm

def cao_method(a, e, e_star, dim, tsrecon, ind_m, ind, ts, tau):
    # Function to compute the Cao method criteria for embedding dimension. (Cao 1996)
    import numpy as np
    a.append(np.divide(np.linalg.norm(tsrecon[ind_m, :]-tsrecon[ind, :], ord=np.inf, axis=1), 
                              np.linalg.norm(tsrecon[ind_m, :-1]-tsrecon[ind, :-1], ord=np.inf, axis=1)))
    e.append(np.multiply(np.divide(1,len(ts)-dim*tau), np.sum(a[dim-2])))
    e_star.append(np.multiply(np.divide(1,len(ts)-dim*tau), np.sum(np.abs(tsrecon[ind_m, -1]-tsrecon[ind, -1]))))
    num1 = np.array(e)[1:][dim-2]
    den1 = np.array(e)[:-1][dim-2]
    num2 = np.array(e_star)[1:][dim-2]
    den2 = np.array(e_star)[:-1][dim-2] 
    if den1 != 0:
        e1 = num1/den1
        e2 = num2/den2
        return e1, e2
    else:
        return 0, 0


def FNN_n(ts, tau, maxDim=10, plotting=False, Rtol=15, Atol=2, Stol=0.9, threshold=10, method=None):
    """This function implements the False Nearest Neighbors (FNN) algorithm described by Kennel et al.
    to select the minimum embedding dimension.

    Args:
       ts (array):  Time series (1d).
       tau (int):  Embedding delay.


    Kwargs:
       maxDim (int):  maximum dimension in dimension search. Default is 10.

       plotting (bool): Plotting for user interpretation. Default is False.

       Rtol (float): Ratio tolerance. Default is 15. (10 recommended for false strands)

       Atol (float): A tolerance. Default is 2.

       Stol (float): S tolerance. Default is 0.9.

       threshold (float): Tolerance threshold for percent of nearest neighbors. Default is 10%.

       method (string): 'strand' Use the composite false nearest strands algorithm (David Chelidze 2017), 'cao' Use the Cao method (Cao 1996). Default is None.

    Returns:
       (int): n, The embedding dimension.

    """

    import numpy as np
    from scipy.spatial import KDTree
    if len(ts)-(maxDim-1)*tau < 20:
        maxDim = len(ts)-(maxDim-1)*tau-1
    ts = np.reshape(ts, (len(ts), 1))  # ts is a column vector
    st_dev = np.std(ts)  # standart deviation of the time series

    Xfnn = []
    dim_array = []

    if method=='strand':
        # Set theiler window for false nearest strands. (4 times delay)
        w = 4 * tau
    elif method=='cao':
        a = []
        e = [0]
        e1 = []
        e_star = [0]
        e2 = []
        w = 1
    else:
        w = 1
    
    flag = False
    i = 0
    while flag == False:
        i = i+1
        dim = i
        tsrecon = ts_recon(ts, dim, tau)  # delay reconstruction

        tree = KDTree(tsrecon)
        D, IDX = tree.query(tsrecon, k=w+1)

        if method=='strand':
            D, IDX = find_neighbors(D, IDX, w)

        # Calculate the false nearest neighbor ratio for each dimension
        if i > 1:
            if method=='strand':
                fns_frac = FNS_fraction(dim, strand_ind, tsrecon, s_ind_m, s_dm, ind, Rtol, Stol, st_dev)
                Xfnn.append(fns_frac)
                dim_array.append(dim-1)
                if fns_frac <= threshold or i == maxDim:
                    flag = True
                    minDim = dim-1
            elif method=='cao':
                e1_new, e2_new = cao_method(a, e, e_star, dim, tsrecon, ind_m, ind, ts, tau)
                e1.append(e1_new)
                e2.append(e2_new)
                Xfnn.append(e1[-1])
                dim_array.append(dim-2)
                if np.abs(1-e1_new) < threshold/100 or i == maxDim:
                    import warnings
                    flag = True
                    minDim = dim-3
                    if not any(np.abs(1 - np.array(e2[1:])) > threshold/100):
                        warnings.warn("This data may be random.", category=Warning)
            else:
                D_mp1 = np.sqrt(
                    np.sum((np.square(tsrecon[ind_m, :]-tsrecon[ind, :])), axis=1))
                # Criteria 1 : increase in distance between neighbors is large
                num1 = np.heaviside(
                    np.divide(abs(tsrecon[ind_m, -1]-tsrecon[ind, -1]), Dm)-Rtol, 0.5)
                # Criteria 2 : nearest neighbor not necessarily close to y(n)
                num2 = np.heaviside(Atol-D_mp1/st_dev, 0.5)
                num = sum(np.multiply(num1, num2))
                den = sum(num2)
                Xfnn.append((num / den) * 100)
                dim_array.append(dim-1)
                if (num/den)*100 <= threshold or i == maxDim:
                    flag = True
                    minDim = dim-1  

        # Save the index to D and k(n) in dimension m for comparison with the
        # same distance in m+1 dimension
        xlen2 = len(ts)-dim*tau
        Dm = D[0:xlen2, -1]
        ind_m = IDX[0:xlen2, -1]
        ind = ind_m <= xlen2-1
        ind_m = ind_m[ind]
        Dm = Dm[ind]

        # Get strands from index of nearest neighbors
        if method=='strand':
            strand_ind, s_ind, s_ind_m, s_dm = compute_strands(xlen2, D, IDX)  
        else:
            pass

    Xfnn = np.array(Xfnn)

    
    import matplotlib.pyplot as plt
    from matplotlib import rc
#     rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
#     rc('text', usetex=True)
#     plt.rc('text', usetex=True)
#     plt.rc('font', family='serif')
    if method == 'cao' and plotting == True:    
        TextSize = 20
        plt.figure(1)
        plt.plot(dim_array[1:], e1[1:], marker='x', linestyle='-', color='blue', label=f'E_1')
        plt.plot(dim_array[1:], e2[1:], marker='o', linestyle='-', color='red', label='$_2')
        plt.xlabel(f'Dimension {n}', size=TextSize)
        plt.ylabel('E1, E2', size=TextSize)
        plt.xticks(size=TextSize)
        plt.yticks(size=TextSize)
        plt.ylim([-0.1, 1.2])
        plt.legend()
        plt.show()
    elif plotting == True:
        TextSize = 20
        plt.figure(1)
        plt.plot(dim_array, Xfnn, marker='x', linestyle='-', color='blue')
        plt.xlabel(f'Dimension n', size=TextSize)
        plt.ylabel('Percent FNN', size=TextSize)
        plt.xticks(size=TextSize)
        plt.yticks(size=TextSize)
        plt.ylim([-0.1, 101])
        plt.show()

    return Xfnn, minDim



    perc_FNN, n = FNN_n(ts, tau, plotting=True)
    print('FNN embedding Dimension: ', n)

# def get_embedding_dim(x, delay=3, max_dim=12, threshold=0.95, max_rel_change=0.1, plot=False, noise=0.0):
#     x = numpy2ri.numpy2rpy(x)
#     cao_emb_dim = nonlinearTseries.estimateEmbeddingDim(
#         x,  # time series
#         len(x),  # number of points to use, use entire series
#         delay,  # time delay
#         max_dim,  # max no. of dimension
#         threshold,  # threshold value
#         max_rel_change,  # max relative change
#         plot,  # do the plot
#         "Computing the embedding dimension",  # main
#         "dimension (d)",  # x_label
#         "E1(d) & E2(d)",  # y_label
#         ro.NULL,  # x_lim
#         ro.NULL,  # y_lim
#         noise # add a small amount of noise to the original series to avoid the
#               # appearance of false neighbours due to discretization errors.
#               # This also prevents the method to fail with periodic signals, 0 for no noise
#     )
#     return int(cao_emb_dim[0])

def get_embedding_dim(data, tau, maxDim, plot, threshold):
    trials, timesteps, features = data.shape
    dims = np.array([[FNN_n(data[x, :, y], tau, maxDim=10, plotting=plot, Rtol=15, Atol=2, Stol=0.9, threshold=threshold, method="cao")[1] for x in range(trials)] for y in range(features)])
    return np.nanmedian(dims)

# def takens_embedding(data, tau, k):
#     data_TE = np.zeros((data.shape[0], data.shape[1]-tau*k, data.shape[2]), dtype = object)
    
#     for i in range(data.shape[0]):
#         for j in range(data.shape[2]):
#             for t in range(data.shape[1]-tau*k):
#                 data_TE[i,t,j] = data[i, t:t+tau*k+1, j][::tau][::-1]
                
#     data_TE = np.array(data_TE.tolist())
#     data_TE = data_TE.reshape(data_TE.shape[0],data_TE.shape[1], data.shape[2]*(k+1))
    
#     return data_TE


def embed_data(data, e_time_threshold=1/np.e, maxlags=100, max_dim=12, nn_threshold=10, plot=False):
    best_delay = np.ceil(get_best_1_e_time(data, maxlags=maxlags, threshold=e_time_threshold)).astype(int)
    print(best_delay)
    best_dim = np.ceil(get_embedding_dim(data, tau=best_delay, maxDim=max_dim, threshold=nn_threshold, plot=plot)).astype(int)
    print(f"Data has been embedded using a delay of {best_delay} timesteps and an embedding dimension of {best_dim}")
    return takens_embedding(data, best_delay, best_dim), best_dim, best_delay

def convolutional_embedding(data, base, max_power):
    """
    Embeds the data using lags of powers of two (up to a given power) for each feature
    in a multi-trial dataset and adds these embeddings as additional features along with the original features.
    
    Parameters:
    data (array-like): The input data with dimensions (trials, timesteps, features).
    max_power (int): The maximum power of two for the lags.

    Returns:
    np.ndarray: The embedded data with original and additional lagged features.
    """
    data = np.asarray(data)
    trials, timesteps, features = data.shape
    max_lag = int(np.ceil(base ** max_power))
    if max_lag >= timesteps:
        raise ValueError("The maximum lag is larger than or equal to the number of timesteps.")
    
    # Calculate the required lags
    lags = [int(np.ceil(base ** i)) for i in range(max_power + 1)]
    
    # Initialize the embedded data array with the original and additional features
    num_lags = len(lags)
    total_features = features * (num_lags + 1)
    embedded_data = np.zeros((trials, timesteps - max_lag, total_features))
    
    # Loop through each trial
    for trial in range(trials):
        # Loop through each feature
        for feature in range(features):
            original_series = data[trial, :, feature]
            # Insert the original feature
            embedded_data[trial, :, feature * (num_lags + 1)] = original_series[max_lag:]
            # Fill the embedded data array with the appropriate lags for the current feature
            for i, lag in enumerate(lags):
                embedded_data[trial, :, feature * (num_lags + 1) + i + 1] = original_series[max_lag - lag: timesteps - lag]
    
    return embedded_data
    
import jax
import jax.numpy as jnp


def takens_embedding(data, tau: int, dimension: int):
    """
    Apply Takens embedding to multiple trials of multivariate time series data.
    
    Args:
        data: Array of shape (trials, timesteps, features)
        tau: Integer time delay (static)
        dimension: Integer embedding dimension (static)
    
    Returns:
        Embedded data of shape (trials, timesteps - (dimension-1)*tau, features*dimension)
    """
    trials, timesteps, features = data.shape
    valid_timesteps = timesteps - (dimension-1) * tau
    
    def get_delayed_slice(d):
        start = d * tau
        return jax.lax.dynamic_slice_in_dim(data, start, valid_timesteps, axis=1)
    
    # Get all delayed versions using vmap over the dimension
    delayed_versions = jax.vmap(get_delayed_slice)(jnp.arange(dimension))
    
    # Reshape to get the desired output format
    # delayed_versions shape: (dimension, trials, valid_timesteps, features)
    return jnp.transpose(delayed_versions, (1, 2, 0, 3)).reshape(trials, valid_timesteps, dimension * features)

takens_embedding = jax.jit(takens_embedding, static_argnums=[1, 2])
