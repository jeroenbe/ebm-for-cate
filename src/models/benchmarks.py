import random
import torch, torchmetrics
import click, wandb
import torch.nn as nn

from collections import defaultdict

import numpy as np
from scipy.stats import norm

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from src.data.data_module import IHDP, LBIDD, Twins, Synth

class AE(pl.LightningModule):
    def __init__(self, 
            input_dim,
            architecture=((20, 20), (20, 20)), 
            K: int=5, lr=.001,):
        super().__init__()
        
        self.architecture = architecture
        self.input_dim = input_dim
        self.K, self.lr = K, lr

        layers_encoder = np.array([[nn.Linear(layer[0], layer[1]), nn.ReLU()] for layer in self.architecture])
        layers_decoder = np.array([[nn.Linear(layer[1], layer[0]), nn.ReLU()] for layer in self.architecture])

        self.enc = nn.Sequential(
            nn.Linear(self.input_dim, self.architecture[0][0]),
            *layers_encoder.flatten(),
            nn.Linear(self.architecture[-1][1], self.K)
        )

        self.dec = nn.Sequential(
            nn.Linear(self.K, self.architecture[-1][1]),
            *np.flip(layers_decoder, axis=0).flatten(),
            nn.Linear(self.architecture[0][0], input_dim)
        )

        def init_weights(m):
            if type(m) == nn.Linear:
                torch.nn.init.normal_(m.weight)
                m.bias.data.fill_(0.01)

        self.enc.apply(init_weights)
        self.dec.apply(init_weights)

        self.loss = torchmetrics.MeanSquaredError()


        self.save_hyperparameters()

    def configure_optimizers(self):
        return torch.optim.Adam(list(self.enc.parameters()) + list(self.dec.parameters()), lr=self.lr)
    
    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, log='train')
        
    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, log='val')

    def shared_step(self, batch, log: str=None):
        X, _, _ = batch
        encoding = self.enc(X)
        decoding = self.dec(encoding)
        
        loss = self.loss(decoding, X)

        if log is not None:
            self.log(f'{log}_loss', loss.item(), on_epoch=True)

        return loss

    def forward(self, x):
        return self.enc(x)
    
