# import numpy as np
# import numpy.random as npr
# import matplotlib
# # matplotlib.use('agg')
# import matplotlib.pyplot as plt
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.optim.lr_scheduler import ReduceLROnPlateau
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# from .latent_neural_ode_model import LatentODEfunc, RecognitionRNN, Decoder, plot_graph, save_model, RunningAverageMeter, log_normal_pdf, normal_kl


# MSELoss = torch.nn.MSELoss()

# def train_network(data_train, data_val, samp_ts, val_ts, n_itrs, latent_dim, n_hidden, obs_dim, rnn_hidden, dec_hidden, batch_size, lr=0.008, func=None, rec=None, dec=None, checkpoint_itr=5, dropout=0.1, noise_std=0.2, alpha=1e-5, device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), adjoint=False):
#     if func is None:
#         func = LatentODEfunc(latent_dim, n_hidden, dropout).to(device)
#     else:
#         func.train()
#     if rec is None:
#         rec = RecognitionRNN(latent_dim, obs_dim, rnn_hidden, dropout, batch_size).to(device)
#     else:
#         rec.train()
#     if dec is None:
#         dec = Decoder(latent_dim, obs_dim, dec_hidden, dropout).to(device)
#     else:
#         dec.train()
#     if adjoint:
#         from torchdiffeq import odeint_adjoint as odeint
#     else:
#         from torchdiffeq import odeint as odeint
#     params = (list(func.parameters()) + list(dec.parameters()) + list(rec.parameters()))
#     optimizer = optim.Adam(params, lr=lr)
#     scheduler = ReduceLROnPlateau(optimizer)
#     loss_meter = RunningAverageMeter()
    
#     train_mses = []
#     train_kls = []
#     train_losses = []
#     val_losses = []
#     val_mses = []
#     val_kls = []

#     torch.cuda.empty_cache()

#     train_loader = DataLoader(dataset = data_train, batch_size = batch_size, shuffle = True, drop_last = True)
#     val_loader = DataLoader(dataset = data_val, batch_size = batch_size, shuffle = True, drop_last = False)
#     try:
#         for itr in range(n_itrs):
#             for data in train_loader:
#                 optimizer.zero_grad()
#                 h = rec.initHidden().to(device)
#                 c = rec.initHidden().to(device)
#                 hn = h[0, :, :]
#                 cn = c[0, :, :]
#                 for t in reversed(range(data.size(1))):
#                     obs = data[:, t, :]
#                     out, hn, cn = rec.forward(obs, hn, cn)
#                 qz0_mean, qz0_logvar = out[:, :latent_dim], out[:, latent_dim:]
#                 epsilon = torch.randn(qz0_mean.size()).to(device)
#                 z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean

#                 # forward in time and solve ode for reconstructions
#                 pred_z = odeint(func, z0, samp_ts).permute(1, 0, 2)
#                 pred_x = dec(pred_z)

#                 # compute loss
#                 noise_std_ = torch.zeros(pred_x.size()).to(device) + noise_std
#                 noise_logvar = 2. * torch.log(noise_std_).to(device)
#                 logpx = log_normal_pdf(
#                     data, pred_x, noise_logvar).sum(-1).sum(-1)
#                 pz0_mean = pz0_logvar = torch.zeros(z0.size()).to(device)
#                 analytic_kl = normal_kl(qz0_mean, qz0_logvar,
#                                         pz0_mean, pz0_logvar).sum(-1)
#                 kl_loss = torch.mean(-logpx + analytic_kl, dim=0)
#                 mse_loss = MSELoss(pred_x, data)
#                 mean_loss = MSELoss(torch.mean(pred_x, dim=1), torch.mean(data, dim=1))
#                 std_loss = MSELoss(torch.std(pred_x, dim=1), torch.std(data, dim=1))
#                 loss = alpha*kl_loss + 10*mse_loss + mean_loss + std_loss
                
# #                 loss = MSELoss(pred_x, data)
#                 loss.backward()
#                 optimizer.step()
#                 train_losses.append(loss.item())
#                 train_kls.append(alpha*kl_loss.item())
#                 train_mses.append(mse_loss.item())

#             with torch.no_grad():
#                 for data_val in val_loader:
#                     h = torch.zeros(1, data_val.shape[0], rnn_hidden).to(device)
#                     c = torch.zeros(1, data_val.shape[0], rnn_hidden).to(device)
#                     hn = h[0, :, :]
#                     cn = c[0, :, :]

#                     for t in reversed(range(data_val.size(1))):
#                         obs = data_val[:, t, :]
#                         out, hn, cn = rec.forward(obs, hn, cn)
#                     qz0_mean, qz0_logvar = out[:, :latent_dim], out[:, latent_dim:]
#                     epsilon = torch.randn(qz0_mean.size()).to(device)
#                     z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean

#                     #forward in time and solve ode for reconstructions
#                     pred_z = odeint(func, z0, val_ts).permute(1, 0, 2)
#                     pred_x = dec(pred_z)
                    
#                     noise_std_ = torch.zeros(pred_x.size()).to(device) + noise_std
#                     noise_logvar = 2. * torch.log(noise_std_).to(device)
#                     logpx = log_normal_pdf(
#                         data_val, pred_x, noise_logvar).sum(-1).sum(-1)
#                     pz0_mean = pz0_logvar = torch.zeros(z0.size()).to(device)
#                     analytic_kl = normal_kl(qz0_mean, qz0_logvar,
#                                             pz0_mean, pz0_logvar).sum(-1)
#                     val_kl_loss = torch.mean(-logpx + analytic_kl, dim=0)
#                     val_mse_loss = MSELoss(pred_x, data_val)
#                     val_mean_loss = MSELoss(torch.mean(pred_x, dim=1), torch.mean(data_val, dim=1))
#                     val_std_loss = MSELoss(torch.std(pred_x, dim=1), torch.std(data_val, dim=1))
#                     val_loss = alpha*val_kl_loss + val_mse_loss + val_mean_loss + val_std_loss
                    
#                     val_losses.append(val_loss.item())
#                     val_kls.append(alpha*val_kl_loss.item())
#                     val_mses.append(val_mse_loss.item())

#             # if ((itr > 1000) and (itr % 15 == 0)):
#             #     pass
#                 # save_model(tau, k, latent_dim, itr)
#             if (itr % checkpoint_itr == 0):
#                 print(f'Iter: {itr}, total loss: {loss.item()}, kl loss: {alpha*kl_loss.item()}, mse loss: {mse_loss.item()}')
#                 print(f'mean loss: {mean_loss.item()}, std loss: {std_loss.item()}')
#                 print(f'val loss: {val_loss.item()}, val kl loss: {alpha*val_kl_loss.item()}, val mse loss: {val_mse_loss.item()}')
#                 print(f'val mean loss: {val_mean_loss.item()}, val std loss: {val_std_loss.item()}')
#                 print("\n")
#     except KeyboardInterrupt:
#         print("Training interrupted. Current model's loss:")
#         print(f'Iter: {itr}, running avg mse: {loss.item()}, val_loss: {val_loss.item()}')
#         return (func, rec, dec), (train_losses, val_losses, val_mses)
#     return (func, rec, dec), (train_losses, val_losses, val_mses)

# def generate_data_from_model(model, seeding_data, steps, dt, adjoint=False):
#     func, rec, dec = model
#     if adjoint:
#         from torchdiffeq import odeint_adjoint as odeint
#     else:
#         from torchdiffeq import odeint as odeint
#     func.eval()
#     rec.eval()
#     dec.eval()
#     rnn_hidden = rec.nhidden
#     latent_dim = func.fc1.in_features
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     new_ts = np.arange(0, steps*dt, step=dt)
#     new_ts = torch.from_numpy(new_ts).float().to(device)

#     all_data = torch.from_numpy(seeding_data).float().to(device)

#     h = torch.zeros(1, all_data.shape[0], rnn_hidden).to(device)
#     c = torch.zeros(1, all_data.shape[0], rnn_hidden).to(device)

#     hn = h[0, :, :]
#     cn = c[0, :, :]

#     with torch.no_grad():
#         for t in reversed(range(all_data.shape[1])):
#             obs = all_data[:, t, :]
#             out, hn, cn = rec.forward(obs, hn, cn)
#         qz0_mean, qz0_logvar = out[:, :latent_dim], out[:, latent_dim:]
#         epsilon = torch.randn(qz0_mean.size()).to(device)
#         z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean

#         # forward in time and solve ode for reconstructions
#         pred_z = odeint(func, z0, new_ts).permute(1, 0, 2) #change time and batch with permute
#         pred_x = dec(pred_z)
        
#     return pred_x.detach().cpu(), pred_z.detach().cpu()

# def load_model(path='model/ODE_TakenEmbedding_RLONG_rnn2_lstm256_tau18k5_LSTM_lr0.008_latent12_LSTMautoencoder_Dataloader_timestep500_epoch1410.pth')
    
#     checkpoint = torch.load(path)
#     func = LatentODEfunc(latent_dim, nhidden).to(device)
#     rec = RecognitionRNN(latent_dim, obs_dim, rnn_nhidden, batch_size).to(device)
#     dec = Decoder(latent_dim, obs_dim, dec_nhidden).to(device)
#     rec.load_state_dict(checkpoint['encoder_state_dict'])
#     func.load_state_dict(checkpoint['odefunc_state_dict'])
#     dec.load_state_dict(checkpoint['decoder_state_dict'])