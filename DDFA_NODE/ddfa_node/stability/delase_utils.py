from delase import DeLASE, dmd
from delase.metrics import aic, mase, mse, r2_score
from tqdm.auto import tqdm
import numpy as np
import torch
from ..utils.data_preparation import change_trial_length


def get_aics(all_data, matrix_sizes, ranks, dt=0.002,
                     max_freq=None, max_unstable_freq=None,
                     device=torch.device("cuda"), n_delays = None,
                     delay_interval = 1, N_time_bins = 50):
    if max_freq is None:
        max_freq = (1/dt)//2
    if max_unstable_freq is None:
        max_unstable_freq = (1/dt)//2

    aics = np.zeros((len(all_data), len(matrix_sizes), len(ranks)))
    iterator = tqdm(total=len(matrix_sizes)*len(ranks)*len(all_data))
    for ii, data_trials in enumerate(all_data):
        for idx, matrix_size in enumerate(matrix_sizes):
            for jdx, rank in enumerate(ranks):
                params = {
                    "matrix_size": matrix_size,
                    "r": rank
                }
                delase = DeLASE(data_trials,
                                matrix_size=params['matrix_size'],
                                rank=params['r'],
                                dt=dt,
                                max_freq=max_freq,
                                max_unstable_freq=max_unstable_freq,
                                device=device,
                                verbose=False,
                                n_delays=n_delays,
                                delay_interval=delay_interval,
                                N_time_bins=N_time_bins
                                )
                delase.fit(verbose=False)

                result = {} | params
                result['stability_params'] = delase.stability_params.cpu().numpy()
                result['stability_freqs'] = delase.stability_freqs.cpu().numpy()

                # HAVOK
                preds = delase.DMD.predict(data_trials)
                preds = preds.cpu().numpy()
                aic_val = aic(data_trials.cpu()[delase.n_delays:],
                              preds[delase.n_delays:],
                              k=delase.DMD.A_v.shape[0]*delase.DMD.A_v.shape[1])
                aics[ii, idx, jdx] = aic_val
                iterator.update()
    iterator.close()
    return aics


def get_λs(all_data, aics, matrix_sizes, ranks, full_output=True,
           top_percent=None, dt=0.002, max_freq=None, max_unstable_freq=None,
           device=torch.device("cuda"), trial_len=1500, skip=1500,  n_delays=1,
                                delay_interval=1, N_time_bins=50):
    if torch.is_tensor(all_data):
        all_data = all_data.cpu()
    if max_freq is None:
        max_freq = (1/dt)//2
    if max_unstable_freq is None:
        max_unstable_freq = (1/dt)//2

    if full_output and top_percent is not None:
        raise ValueError("You must select either to get the full stability curve by setting full_output=True or to get the mean of the top X percent of the stability parameters by using top_percent=X")

    n_trials = change_trial_length(all_data[0, :, :][None, :, :], timesteps_per_subsample=trial_len, skip=skip).shape[0]

    if full_output:
        λs = np.zeros((all_data.shape[0], n_trials), dtype="O")
    else:
        λs = np.zeros((all_data.shape[0], n_trials))
    for idx, data_trials in enumerate(all_data):
        matrix_size_idx, rank_idx = np.unravel_index(np.argmin(aics[idx]),
                                                     aics[idx].shape)
        trials = change_trial_length(data_trials[None, :, :], timesteps_per_subsample=trial_len, skip=skip)
        for jdx, data in enumerate(trials):
            delase = DeLASE(data,
                            matrix_size=matrix_sizes[matrix_size_idx],
                            rank=ranks[rank_idx],
                            dt=dt,
                            max_freq=max_freq,
                            max_unstable_freq=max_unstable_freq,
                            device=device,
                            verbose=False,
                            n_delays=n_delays,
                            delay_interval=delay_interval,
                            N_time_bins=N_time_bins
                            )
            delase.fit(verbose=False)
            stab_curve = delase.stability_params.cpu().numpy()
            if full_output:
                λs[idx, jdx] = stab_curve
            else:
                λs[idx, jdx] = stab_curve[:int(top_percent/100*len(stab_curve))].mean()
    return λs