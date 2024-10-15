import os
import sys
sys.path.append('/home/michael/Synology/Desktop/Data/Python/Gait-Signatures/NeuralODE/DDFA_NODE/')
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,3"
import tempfile

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import src as ddfa_node
import ray
from ray import train, tune
from ray.train import Checkpoint, ScalingConfig
from ray.train.torch import TorchTrainer, get_device
from ray.tune.examples.mnist_pytorch import ConvNet, get_data_loaders, test_func
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.schedulers.pb2 import PB2
from torchdiffeq import odeint
from hyperopt import hp

import numpy as np
from scipy.signal import savgol_filter

def train_convnet(config):
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		# Create our data loaders, model, and optmizer.
		step = 1
		alpha_norm = 1e-5
		batch_size = config.get("batch_size", 32)
		latent_dim = config.get("latent_dim", 32)
		timesteps_per_sample = config.get("timesteps_per_sample", 300)
		n_hidden = 64
		dec_hidden = 16
		rnn_hidden = config.get("rnn_hidden", 128)
		dec_dropout = config.get("dec_dropout", 0.1)
		weight_decay = config.get("weight_decay", 1e-3)
		alpha = config.get("alpha", 1e-5)
		
		#data = ddfa_node.load_data_normalize(6, '/home/mhess3/Synology/Desktop/Data/Julia/data/VDP_data.npy')
		window_length = 50
		polyorder = 3
		data = np.load("/home/michael/Synology/Desktop/Data/Julia/data/VDP_SDEs.npy")[:, 3000:, :][:, :, :]
		data = (data - data.mean(axis=1)[:, None, :]) / data.std(axis=1)[:, None, :]
		for trial in range(data.shape[0]):
			data[trial, :, :] = savgol_filter(data[trial], window_length=window_length, polyorder=polyorder, axis=0)
		data = data[:, ::5, :]
		#time_delayed_data, k, τ = ddfa_node.tde.embed_data(data, maxlags=500)
		k, τ = 7, 2
		time_delayed_data = ddfa_node.takens_embedding(data, tau=τ, k=k)

		
		data = ddfa_node.change_trial_length(time_delayed_data, timesteps_per_subsample=timesteps_per_sample, skip=5)

		# Train/test splitting
		train_size = 0.8
		data_train, data_val = ddfa_node.split_data(data, train_size=train_size)
		print(data_val.shape)
		obs_dim = data_train.shape[-1]

		# Add noise to data
		noise_std = 0.05
		#data_train = ddfa_node.augment_data_with_noise(data_train, n_copies=5, noise_std=noise_std)

		data_train, data_val = torch.from_numpy(data_train).float().to(device), torch.from_numpy(data_val).float().to(device)
		
		######### REMOVE EVENTUALLY #############
		data_val = torch.from_numpy(time_delayed_data[:, 1:1000, :]).float().to(device)
		#########################################
		
		train_loader = DataLoader(dataset = data_train, batch_size = batch_size, shuffle = True, drop_last = True)
		val_loader = DataLoader(dataset = data_val, batch_size = batch_size, shuffle = True, drop_last = True)
		# train_loader, val_loader = ray.train.torch.prepare_data_loader(train_loader), ray.train.torch.prepare_data_loader(val_loader)
		dt = 0.01
		ts_num = timesteps_per_sample * dt
		tot_num = data_train.shape[1]

		samp_ts = np.linspace(0, ts_num, num=tot_num)
		samp_ts = torch.from_numpy(samp_ts).float().to(device)
		
		
		#dt = np.diff(samp_ts.cpu())[1]
		val_ts = np.linspace(0, dt*data_val.shape[1], num=data_val.shape[1])
		val_ts = torch.from_numpy(val_ts).float().to(device)
		
		func = ddfa_node.LatentODEfunc(latent_dim, n_hidden).to(device)
		rec = ddfa_node.RecognitionRNN(latent_dim, obs_dim, rnn_hidden, dec_dropout, batch_size).to(device)
		dec = ddfa_node.Decoder(latent_dim, obs_dim, dec_hidden, dropout=dec_dropout).to(device)
		
		
		model = NODE(func, rec, dec, latent_dim, odeint, samp_ts, val_ts, device)
		MSELoss = nn.MSELoss()
		
		optimizer = optim.Adam(
				model.parameters(),
				lr=config.get("lr", 0.000075),
				weight_decay=weight_decay
		)
		loss_fn_1 = torch.nn.MSELoss()
		# If `train.get_checkpoint()` is populated, then we are resuming from a checkpoint.
		checkpoint = train.get_checkpoint()
		if checkpoint:
				with checkpoint.as_directory() as checkpoint_dir:
						checkpoint_dict = torch.load(os.path.join(checkpoint_dir, "checkpoint.pt"))

				# Load model state and iteration step from checkpoint.
				model.load_state_dict(checkpoint_dict["model_state_dict"])
				# Load optimizer state (needed since we're using momentum),
				# then set the `lr` and `momentum` according to the config.
				optimizer.load_state_dict(checkpoint_dict["optimizer_state_dict"])
				for param_group in optimizer.param_groups:
						if "lr" in config:
								param_group["lr"] = config["lr"]
						if "momentum" in config:
								param_group["momentum"] = config["momentum"]
						if "batch_size" in config:
								param_group["batch_size"] = config["batch_size"]
						if "latent_dim" in config:
								param_group["latent_dim"] = config["latent_dim"]
						if "rnn_hidden" in config:
								param_group["rnn_hidden"] = config["rnn_hidden"]
						if "dec_dropout" in config:
								param_group["dec_dropout"] = config["dec_dropout"]
						if "alpha" in config:
								param_group["alpha"] = config["alpha"]
						if "weight_decay" in config:
								param_group["weight_decay"] = config["weight_decay"]
						if "timesteps_per_sample" in config:
								param_group["timesteps_per_sample"] = config["timesteps_per_sample"]
						

				# Note: Make sure to increment the checkpointed step by 1 to get the current step.
				last_step = checkpoint_dict["step"]
				step = last_step + 1

		while True:
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
						
				
				losses = torch.zeros(len(val_loader))
				with torch.no_grad():
						model.mode = "val"
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
								val_loss = alpha_norm*val_kl_loss + val_mse_loss
								
								losses[idx] = val_loss
				model.mode = "train"
				metrics = {"total_loss": -loss.item(), "kl_loss": -kl_loss.item(), "mse_loss": -mse_loss.item(), "total_val_loss": -val_loss.item(), "val_kl_loss": -val_kl_loss.item(), "val_mse_loss": -val_mse_loss.item(), "lr": config["lr"], "alpha": config["alpha"], "weight_decay": config["weight_decay"]}

				# Every `checkpoint_interval` steps, checkpoint our current state.
				if step % 10 == 0:
						with tempfile.TemporaryDirectory() as tmpdir:
								torch.save(
										{
												"step": step,
												"model_state_dict": model.state_dict(),
												"optimizer_state_dict": optimizer.state_dict(),
										},
										os.path.join(tmpdir, "checkpoint.pt"),
										
								)
								train.report(metrics, checkpoint=Checkpoint.from_directory(tmpdir))
				else:
						train.report(metrics)

				step += 1

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

pb2_scheduler = PB2(
		time_attr='training_iteration',
		metric='total_val_loss',
		mode='max',
		perturbation_interval=10,
		hyperparam_bounds={
				"lr": [1e-6, 1e-2],
				"dec_dropout": [0, 0.3],
				"alpha": [0, 1e-3],
				"weight_decay": [0, 1e-1]
		}
)

if ray.is_initialized():
		ray.shutdown()
ray.init(include_dashboard=True, runtime_env={"working_dir": "/home/michael/Synology/Desktop/Data/Python/Gait-Signatures/NeuralODE/DDFA_NODE/", "excludes" : ["data"]})

trainable_with_gpu = tune.with_resources(train_convnet, {"gpu": 1.0})

tuner = tune.Tuner(
		trainable_with_gpu,
		run_config=train.RunConfig(
				name="pb2_euler_vdp",
				# Stop when we've reached a threshold accuracy, or a maximum
				# training_iteration, whichever comes first
				# stop={"test_loss": 0.05, "training_iteration": 2500},
				stop={"total_val_loss": -0.01, "training_iteration": 3000},
				checkpoint_config=train.CheckpointConfig(
						checkpoint_score_attribute="total_val_loss",
						num_to_keep=10,
				),
				storage_path="/home/michael/Synology/Desktop/Data/Python/Gait-Signatures/ray_results",
		),
		tune_config=tune.TuneConfig(
				scheduler=pb2_scheduler,
				num_samples=60,
				max_concurrent_trials=10
		),
		param_space={
				"lr": tune.uniform(1e-5, 1e-2),
				"latent_dim": tune.choice([2**x for x in range(3, 6)]),
				"rnn_hidden": tune.choice([2**x for x in range(6, 10)]),
				"batch_size": tune.choice([32]),
				"timesteps_per_trial": tune.choice([150, 300, 500]),
				"dec_dropout": tune.uniform(0.01, 0.25),
				"alpha": tune.uniform(1e-7, 5e-4),
				"weight_decay": tune.loguniform(1e-16, 1e-1)
		},
)
results_grid = tuner.fit()
