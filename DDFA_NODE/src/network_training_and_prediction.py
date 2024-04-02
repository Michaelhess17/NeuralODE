import numpy as np
import numpy.random as npr
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
# from torchdiffeq import odeint_adjoint as odeint
from torchdiffeq import odeint as odeint
from torch.utils.data import DataLoader
from tqdm import tqdm
from .latent_neural_ode_model import LatentODEfunc, RecognitionRNN, Decoder, plot_graph, save_model, RunningAverageMeter, log_normal_pdf, normal_kl

MSELoss = torch.nn.MSELoss()

def train_network(data_train, data_val, device, samp_ts, val_ts, n_itrs, latent_dim, n_hidden, obs_dim, rnn_hidden, dec_hidden, batch_size, lr=0.008, func=None, rec=None, dec=None, checkpoint_itr=5, dropout=0.1, noise_std=0.2, alpha=1e-5):
    if func is None:
        func = LatentODEfunc(latent_dim, n_hidden, dropout).to(device)
    if rec is None:
        rec = RecognitionRNN(latent_dim, obs_dim, rnn_hidden, dropout, batch_size).to(device)
    if dec is None:
        dec = Decoder(latent_dim, obs_dim, dec_hidden, dropout).to(device)
    params = (list(func.parameters()) + list(dec.parameters()) + list(rec.parameters()))
    optimizer = optim.Adam(params, lr=lr)
    scheduler = ReduceLROnPlateau(optimizer)
    loss_meter = RunningAverageMeter()
    
    train_mses = []
    train_kls = []
    train_losses = []
    val_losses = []
    val_mses = []
    val_kls = []

    torch.cuda.empty_cache()

    train_loader = DataLoader(dataset = data_train, batch_size = batch_size, shuffle = True, drop_last = True)
    val_loader = DataLoader(dataset = data_val, batch_size = batch_size, shuffle = True, drop_last = False)
    try:
        for itr in range(n_itrs):
            for data in train_loader:
                optimizer.zero_grad()
                h = rec.initHidden().to(device)
                c = rec.initHidden().to(device)
                hn = h[0, :, :]
                cn = c[0, :, :]
                for t in reversed(range(data.size(1))):
                    obs = data[:, t, :]
                    out, hn, cn = rec.forward(obs, hn, cn)
                qz0_mean, qz0_logvar = out[:, :latent_dim], out[:, latent_dim:]
                epsilon = torch.randn(qz0_mean.size()).to(device)
                z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean   

                # forward in time and solve ode for reconstructions
                pred_z = odeint(func, z0, samp_ts).permute(1, 0, 2)
                pred_x = dec(pred_z)

                # compute loss
                noise_std_ = torch.zeros(pred_x.size()).to(device) + noise_std
                noise_logvar = 2. * torch.log(noise_std_).to(device)
                logpx = log_normal_pdf(
                    data, pred_x, noise_logvar).sum(-1).sum(-1)
                pz0_mean = pz0_logvar = torch.zeros(z0.size()).to(device)
                analytic_kl = normal_kl(qz0_mean, qz0_logvar,
                                        pz0_mean, pz0_logvar).sum(-1)
                kl_loss = torch.mean(-logpx + analytic_kl, dim=0)
                mse_loss = MSELoss(pred_x, data)
                loss = alpha*kl_loss + mse_loss
                
#                 loss = MSELoss(pred_x, data)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
                train_kls.append(alpha*kl_loss.item())
                train_mses.append(mse_loss.item())

            with torch.no_grad():
                for data_val in val_loader:
                    h = torch.zeros(1, data_val.shape[0], rnn_hidden).to(device)
                    c = torch.zeros(1, data_val.shape[0], rnn_hidden).to(device)
                    hn = h[0, :, :]
                    cn = c[0, :, :]

                    for t in reversed(range(data_val.size(1))):
                        obs = data_val[:, t, :]
                        out, hn, cn = rec.forward(obs, hn, cn)
                    qz0_mean, qz0_logvar = out[:, :latent_dim], out[:, latent_dim:]
                    epsilon = torch.randn(qz0_mean.size()).to(device)
                    z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean

                    #forward in time and solve ode for reconstructions
                    pred_z = odeint(func, z0, val_ts).permute(1, 0, 2)
                    pred_x = dec(pred_z)
                    
                    noise_std_ = torch.zeros(pred_x.size()).to(device) + noise_std
                    noise_logvar = 2. * torch.log(noise_std_).to(device)
                    logpx = log_normal_pdf(
                        data_val, pred_x, noise_logvar).sum(-1).sum(-1)
                    pz0_mean = pz0_logvar = torch.zeros(z0.size()).to(device)
                    analytic_kl = normal_kl(qz0_mean, qz0_logvar,
                                            pz0_mean, pz0_logvar).sum(-1)
                    val_kl_loss = torch.mean(-logpx + analytic_kl, dim=0)
                    val_mse_loss = MSELoss(pred_x, data_val)
                    val_loss = alpha*val_kl_loss + val_mse_loss
                    
                    val_losses.append(val_loss.item())
                    val_kls.append(alpha*val_kl_loss.item())
                    val_mses.append(val_mse_loss.item())

            # if ((itr > 1000) and (itr % 15 == 0)):
            #     pass
                # save_model(tau, k, latent_dim, itr)
            if (itr % checkpoint_itr == 0):
                print(f'Iter: {itr}, total loss: {loss.item()}, kl_loss: {alpha*kl_loss.item()}, mse_loss: {mse_loss.item()} val loss: {val_loss.item()}, val_kl: {alpha*val_kl_loss.item()}, val_mse: {val_mse_loss.item()}')
    except KeyboardInterrupt:
        print("Training interrupted. Current model's loss:")
        print(f'Iter: {itr}, running avg mse: {loss.item()}, val_loss: {val_loss.item()}')
        return func, rec, dec, train_losses, val_losses, val_mses
    return func, rec, dec, train_losses, val_losses, val_mses

def generate_data_from_model(ts_pos, samp_trajs_TE):
    with torch.no_grad():
        # sample from trajectorys' approx. posterior

#         ts_pos = np.linspace(0, ts_num*gen_index, num=tot_num*gen_index)
#         ts_pos = torch.from_numpy(ts_pos).float().to(device)
    
        h = torch.zeros(1, samp_trajs_TE_test.shape[0], rnn_nhidden).to(device)
        c = torch.zeros(1, samp_trajs_TE_test.shape[0], rnn_nhidden).to(device)
    
        hn = h[0, :, :]
        cn = c[0, :, :]
    
        for t in reversed(range(samp_trajs_TE_test.size(1))):
            obs = samp_trajs_TE_test[:, t, :]
            out, hn, cn = rec.forward(obs, hn, cn)
        qz0_mean, qz0_logvar = out[:, :latent_dim], out[:, latent_dim:]
        epsilon = torch.randn(qz0_mean.size()).to(device)
        z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean

        # forward in time and solve ode for reconstructions
        pred_z = odeint(func, z0, ts_pos).permute(1, 0, 2) #change time and batch with permute
        pred_x = dec(pred_z)
        
        return pred_x, pred_z

# def load_model(path='model/ODE_TakenEmbedding_RLONG_rnn2_lstm256_tau18k5_LSTM_lr0.008_latent12_LSTMautoencoder_Dataloader_timestep500_epoch1410.pth')
    
#     checkpoint = torch.load(path)
#     func = LatentODEfunc(latent_dim, nhidden).to(device)
#     rec = RecognitionRNN(latent_dim, obs_dim, rnn_nhidden, batch_size).to(device)
#     dec = Decoder(latent_dim, obs_dim, dec_nhidden).to(device)
#     rec.load_state_dict(checkpoint['encoder_state_dict'])
#     func.load_state_dict(checkpoint['odefunc_state_dict'])
#     dec.load_state_dict(checkpoint['decoder_state_dict'])