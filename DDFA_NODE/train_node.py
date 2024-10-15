def main(train_dir="/home/mhess3/Synology/Desktop/Data/Python/Gait-Signatures/NeuralODE/DDFA_NODE/models/vdp", 
         epochs=500, batch_size=64, latent_dim=16, timesteps_per_sample=500, n_hidden=64, lr=5e-4, 
         dec_hidden=32, rnn_hidden=256, dec_dropout=0.2, weight_decay=1e-5, alpha=1e-5, pause_every=5):
    parser = argparse.ArgumentParser(description='Process some random seed.')
    parser.add_argument('--seed', type=int, required=True, help='Random seed')
    parser.add_argument('--device', type=int, required=True, help='GPU number')

    # Parse arguments
    args = parser.parse_args()
    Path(os.path.join(train_dir, f"{args.seed}")).mkdir(parents=True, exist_ok=True)

    # Set the random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    print(f"Random seed set to {args.seed}")
    
    device = torch.device('cuda')
    # Create our data loaders, model, and optmizer.

    #data = ddfa_node.load_data_normalize(6, '/home/mhess3/Synology/Desktop/Data/Julia/data/VDP_data.npy')
    data = np.load("/home/mhess3/Synology/Desktop/Data/Julia/data/VDP_SDEs.npy")
    # time_delayed_data, k, τ = ddfa_node.embed_data(data, maxlags=500)
    k, τ = 4, 12
    time_delayed_data = ddfa_node.takens_embedding(data[:, :10000, :], tau=τ, k=k)


    data = ddfa_node.change_trial_length(time_delayed_data, timesteps_per_subsample=timesteps_per_sample, skip=timesteps_per_sample//2)

    # Train/test splitting
    train_size = 0.8
    data_train, data_val = ddfa_node.split_data(data, train_size=train_size)
    obs_dim = data_train.shape[-1]

    # Add noise to data
    noise_std = 0.1
    data_train = ddfa_node.augment_data_with_noise(data_train, n_copies=5, noise_std=noise_std)

    data_train, data_val = torch.from_numpy(data_train).float().to(device), torch.from_numpy(data_val).float().to(device)

    ######### REMOVE EVENTUALLY #############
    data_val = torch.from_numpy(time_delayed_data[:, 1:timesteps_per_sample*2, :]).float().to(device)
    #########################################

    train_loader = DataLoader(dataset = data_train, batch_size = batch_size, shuffle = True, drop_last = True)
    val_loader = DataLoader(dataset = data_val, batch_size = batch_size, shuffle = True, drop_last = True)

    dt = 0.01
    ts_num = timesteps_per_sample * dt
    tot_num = data_train.shape[1]

    samp_ts = np.linspace(0, ts_num, num=tot_num)
    samp_ts = torch.from_numpy(samp_ts).float().to(device)

    val_ts = np.linspace(0, dt*data_val.shape[1], num=data_val.shape[1])
    val_ts = torch.from_numpy(val_ts).float().to(device)

    func = ddfa_node.LatentODEfunc(latent_dim, n_hidden).to(device)
    rec = ddfa_node.RecognitionRNN(latent_dim, obs_dim, rnn_hidden, dec_dropout, batch_size).to(device)
    dec = ddfa_node.Decoder(latent_dim, obs_dim, dec_hidden, dropout=dec_dropout).to(device)


    model = NODE(func, rec, dec, latent_dim, odeint, samp_ts, val_ts, device)
    MSELoss = nn.MSELoss()

    optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
    )
    df = pd.DataFrame(columns=["epoch", "total_loss", "kl_loss", "mse_loss", "total_val_loss", "val_mse_loss"])
    for epoch in range(1, epochs + 1):
        for data in train_loader:
            optimizer.zero_grad()
            pred_x, z0, qz0_mean, qz0_logvar = model(data)

            # compute loss
            noise_std_ = torch.zeros(pred_x.size()).to(device) + noise_std
            noise_logvar = 2. * torch.log(noise_std_).to(device)
            logpx = ddfa_node.log_normal_pdf(
                    data, pred_x, noise_logvar).sum(-1).sum(-1)
            pz0_mean = pz0_logvar = torch.zeros(z0.size()).to(device)
            analytic_kl = ddfa_node.normal_kl(qz0_mean, qz0_logvar,
                                                            pz0_mean, pz0_logvar).sum(-1)
            kl_loss = torch.mean(-logpx + analytic_kl, dim=0)
            mse_loss = MSELoss(pred_x, data)
            loss = alpha*kl_loss + mse_loss

            loss.backward()
            optimizer.step()

        if epoch % pause_every == 0:
            model.mode = "val"
            total_losses = torch.zeros(len(val_loader))
            mse_losses = torch.zeros(len(val_loader))

            with torch.no_grad():
                for idx, data in enumerate(val_loader):
                    pred_x, z0, qz0_mean, qz0_logvar = model(data)

                    # compute loss
                    noise_std_ = torch.zeros(pred_x.size()).to(device) + noise_std
                    noise_logvar = 2. * torch.log(noise_std_).to(device)
                    logpx = ddfa_node.log_normal_pdf(
                            data, pred_x, noise_logvar).sum(-1).sum(-1)
                    pz0_mean = pz0_logvar = torch.zeros(z0.size()).to(device)
                    analytic_kl = ddfa_node.normal_kl(qz0_mean, qz0_logvar,
                                                                    pz0_mean, pz0_logvar).sum(-1)
                    val_kl_loss = torch.mean(-logpx + analytic_kl, dim=0)
                    val_mse_loss = MSELoss(pred_x, data)
                    val_loss = alpha*val_kl_loss + val_mse_loss

                    total_losses[idx] = val_loss
                    mse_losses[idx] = val_mse_loss
                    
            model.mode = "train"
            metrics = {"epoch": epoch, "total_loss": -loss.item(), "kl_loss": -kl_loss.item(), "mse_loss": -mse_loss.item(), "total_val_loss": -total_losses.mean().item(), "val_mse_loss": -mse_losses.mean().item()}
            df = pd.concat([df,pd.DataFrame([metrics])], ignore_index=True)
            df.to_csv(os.path.join(train_dir, f"{args.seed}/train_results.csv"))
            print(metrics)

    torch.save(model.state_dict(), os.path.join(train_dir, f"{args.seed}/model_weights.pt"))
    
if __name__ == '__main__':
    import os
    import sys
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, required=True, help='GPU number')
    parser.add_argument('--seed', type=int, required=True, help='Random seed')


    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.device}"

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    import src as ddfa_node
    from torchdiffeq import odeint_adjoint as odeint

    import numpy as np
    import pandas as pd
    import random
    from pathlib import Path

    class NODE(torch.nn.Module):
        def __init__(self, func, rec, dec, latent_dim, odeint, samp_ts, val_ts, device):
            super(NODE, self).__init__()
            self.func = func
            self.rec = rec
            self.dec = dec
            self.latent_dim = latent_dim
            self.odeint = odeint
            self.samp_ts = samp_ts
            self.val_ts = val_ts
            self.device = device
            self.mode = "train"

        def forward(self, x):
            h = self.rec.initHidden().to(self.device)
            c = self.rec.initHidden().to(self.device)
            hn = h[0, :, :]
            cn = c[0, :, :]
            if self.mode == "train":
                    for t in reversed(range(len(self.samp_ts))):
                            obs = x[:, t, :]
                            out, hn, cn = self.rec.forward(obs, hn, cn)
            else:
                    for t in reversed(range(len(self.val_ts))):
                            obs = x[:, t, :]
                            out, hn, cn = self.rec.forward(obs, hn, cn)
            qz0_mean, qz0_logvar = out[:, :self.latent_dim], out[:, self.latent_dim:]
            epsilon = torch.randn(qz0_mean.size()).to(self.device)
            z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean	 

            if self.mode == "train":
                    pred_z = self.odeint(self.func, z0, self.samp_ts).permute(1, 0, 2)
            else:
                    pred_z = self.odeint(self.func, z0, self.val_ts).permute(1, 0, 2)
            # forward in time and solve ode for reconstructions

            pred_x = self.dec(pred_z)

            return pred_x, z0, qz0_mean, qz0_logvar
    main(epochs=50, timesteps_per_sample=1500)