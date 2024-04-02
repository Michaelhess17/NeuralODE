import numpy as np
import torch
def load_data_normalize(obs_dim, datafilepath, noise_std=0.2):
    data = np.load(datafilepath)
    traj_tot = np.load(datafilepath).reshape(72, 1500, obs_dim)
    traj_tot = traj_tot[:,150:1350,:]
    data = data[:, 300:1200, :]
    data = data.reshape(72, 900, obs_dim)
    

    orig_trajs = np.zeros((data.shape[0],data.shape[1],data.shape[2]))
    for i in range(data.shape[0]):
        for j in range(data.shape[2]):
            trajs = data[i,:,j]
            trajs_tot = traj_tot[i,:,j]
            orig_trajs[i,:,j] = (trajs - trajs_tot.mean()) / trajs_tot.std()
            
    #samp_trajs += npr.randn(*samp_trajs.shape) * noise_std #add noise

    return orig_trajs


def split_data(data, train_size=0.5):
    indices = torch.randperm(data.shape[0])
    n = data.shape[0]
    n_train = int(n * train_size)
    return data[indices[:n_train], :, :], data[indices[n_train:], :, :]

def change_trial_length(data, timesteps_per_subsample=100, skip=1):
    num_subjects, num_time_steps, num_features = data.shape
    subsamples = []
    
    # Calculate the number of subsamples
    num_subsamples = (num_time_steps - timesteps_per_subsample) // skip + 1
    
    # Iterate over each subject
    for subject_data in data:
        # Iterate over each subsample
        for i in range(num_subsamples):
            start_index = i * skip
            end_index = start_index + timesteps_per_subsample
            subsample = subject_data[start_index:end_index, :]
            subsamples.append(subsample)
    
    return np.array(subsamples)


def augment_data_with_noise(data, n_copies=5, noise_std=0.1):
    new_data = []
    for _ in range(n_copies):
        for trial in data:
            new_data.append(trial + (np.random.randn(*trial.shape) * noise_std))
    return np.array(new_data)