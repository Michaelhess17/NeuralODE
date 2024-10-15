import torch
from .latent_neural_ode_model import LatentODEfunc, RecognitionRNN, Decoder


def save_model(model, path):
    func, rec, dec = model
    func_weights = func.state_dict()
    rec_weights = rec.state_dict()
    dec_weights = dec.state_dict()

    params = {"latent_dim": func.fc1.in_features,
              "n_hidden": func.fc2.in_features,
              "dec_hidden":  dec.fc2.in_features,
              "obs_dim": rec.h1o.in_features,
              "rnn_hidden": rec.lstm.hidden_size}

    torch.save(func_weights, path + "/latent_ode_func_weights")
    torch.save(rec_weights, path + "/latent_ode_rec_weights")
    torch.save(dec_weights, path + "/latent_ode_dec_weights")
    torch.save(params, path + "/latent_ode_params")


def load_model(path):
    func_weights = torch.load(path + "/latent_ode_func_weights")
    rec_weights = torch.load(path + "/latent_ode_rec_weights")
    dec_weights = torch.load(path + "/latent_ode_dec_weights")
    params = torch.load(path + "/latent_ode_params")

    latent_dim = params["latent_dim"]
    n_hidden = params["n_hidden"]
    dec_hidden = params["dec_hidden"]
    obs_dim = params["obs_dim"]
    rnn_hidden = params["rnn_hidden"]

    func = LatentODEfunc(latent_dim=latent_dim, nhidden=n_hidden)
    func.load_state_dict(func_weights)

    rec = RecognitionRNN(latent_dim=latent_dim, nhidden=rnn_hidden,
                         obs_dim=obs_dim, nbatch=32)
    rec.load_state_dict(rec_weights)

    dec = Decoder(latent_dim=latent_dim, obs_dim=obs_dim, nhidden=dec_hidden)
    dec.load_state_dict(dec_weights)
    return (func, rec, dec)