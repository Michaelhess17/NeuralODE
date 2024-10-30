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
    indices = np.random.permutation(data.shape[0])
    n = data.shape[0]
    n_train = int(n * train_size)
    return data[indices[:n_train], :, :], data[indices[n_train:], :, :]


def change_trial_length(data, timesteps_per_subsample=100, skip=1, get_subject_ids=False):
    num_subjects, num_time_steps, num_features = data.shape
    subsamples = []
    subject_ids = []

    # Calculate the number of subsamples
    num_subsamples = (num_time_steps - timesteps_per_subsample) // skip + 1

    # Iterate over each subject
    subject = 0
    for subject_data in data:
        # Iterate over each subsample
        for i in range(num_subsamples):
            start_index = i * skip
            end_index = start_index + timesteps_per_subsample
            subsample = subject_data[start_index:end_index, :]
            subsamples.append(subsample)
            subject_ids.append(subject)
        subject += 1

    if get_subject_ids:
        return np.array(subsamples), np.array(subject_ids)
    return np.array(subsamples)


def augment_data_with_noise(data, n_copies=5, noise_std=0.1):
    new_data = []
    for _ in range(n_copies):
        for trial in data:
            new_data.append(trial + (np.random.randn(*trial.shape) * noise_std))
    return np.array(new_data)

def prepare_train_val_data(time_delayed_data, sequence_length=500, skip=250, train_size=0.7, noise_std=0.1, n_noisy_copies=3,dt=0.025,device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    # Subsample whole trials to smaller ones
    data = change_trial_length(time_delayed_data, timesteps_per_subsample=sequence_length, skip=skip)

    # Train/test splitting
    data_train, data_val = split_data(data, train_size=train_size)

    # Add noise to data
    data_train = augment_data_with_noise(data_train, n_copies=n_noisy_copies, noise_std=noise_std)

    # put data on the GPU if present
    data_train = torch.from_numpy(data_train).float().to(device) 
    data_val = torch.from_numpy(data_val).float().to(device)
    
    # prepare timestep data for ODE solver to use
    tot_num = data_train.shape[1]
    ts_num = tot_num * dt

    samp_ts = np.arange(0, ts_num, step=dt)
    samp_ts = torch.from_numpy(samp_ts).float().to(device)
    
    val_tot_num = data_val.shape[1]
    val_ts_num = val_tot_num * dt
    
    val_ts = np.arange(0, val_ts_num, step=dt)
    val_ts = torch.from_numpy(val_ts).float().to(device)
    return data_train, data_val, samp_ts, val_ts
    
    