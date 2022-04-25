import numpy as np
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict
from sklearn import metrics
import torch
from scipy import sparse
import torch.utils.data as utils
from torch.autograd import Variable
torch.cuda.set_device(1)
import pickle
import pandas as pd
import os
import sys
from utils import load_data_for_latent_space_plot, get_all_data_loaders, get_config, get_model_list
from trainer import Trainer
from sklearn.preprocessing import StandardScaler

def load_data(isatac=False, data_size=1874, for_training=True, supervise=[], drop=False):
    DATA_DIRECTORY = "processed_data/transcription_factor/"
    log_data = True 
    normalize_data = True

    if isatac:
        f = DATA_DIRECTORY + "diff_atac_shared_cells.npz"
    else:
        f = DATA_DIRECTORY + "diff_expr_shared_cells.npz"

    data = sparse.load_npz(f).T.todense()

    if drop:
        print("drop")
        #threshold = 0.01 if isatac else 0.1
        threshold = 0 if isatac else 0.1
        acceptable = np.count_nonzero(data, axis=0) > threshold * len(data)
        data = data[:, acceptable.flatten().tolist()[0]]

    if log_data:
        data = np.log1p(data)
        if for_training:
            print("Taking log of data..")

    if normalize_data:
        scaler = StandardScaler()
        training_data = data
        scaler.fit(training_data)
        data = scaler.transform(data)

    return Variable(torch.from_numpy(data).float()).cuda() 


def get_unit_model_output(data_a, data_b, name):

    config = get_config("configs/%s.yaml"%name)
    trainer = Trainer(config)
    last_model_name = get_model_list("outputs/outputs/%s/checkpoints/"%name, "gen")
    state_dict = torch.load(last_model_name)
    trainer.gen_a.load_state_dict(state_dict['a'])
    trainer.gen_b.load_state_dict(state_dict['b'])
    trainer.cuda()
    
    latent_a = trainer.gen_a.enc(data_a).data.cpu().numpy()
    latent_b = trainer.gen_b.enc(data_b).data.cpu().numpy()
    
    return latent_a, latent_b


def main():
    data_a = load_data(isatac=True)
    data_b = load_data(isatac=False)

    names = ["cross_modal_supervision_100"]
    labels = ['Cross-modal autoencoders 100%']

    atac_seq_proj = {}
    expr_seq_proj = {}
    for name, label in zip(names, labels):
        latent_a, latent_b = get_unit_model_output(data_a, data_b, name)
        atac_seq_proj[label] = latent_a
        expr_seq_proj[label] = latent_b
    
    print(atac_seq_proj)
    print(expr_seq_proj)

if __name__ == "__main__":
    main()
