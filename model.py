import math

import torch
import torch.nn as nn


# Generator
class Generator(nn.Module):
    def __init__(self, latent_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_size, latent_size),
            nn.ReLU(),
            nn.Linear(latent_size, latent_size),
            nn.ReLU()
        )
        # initialize the weights
        self.model.apply(self._identity_init)

        # checkpoint dir to save and load the model
        self.checkpoint_dir = './chkpt/generator/'

    def forward(self, input):
        return self.model(input)

    def _identity_init(self, m):
        if type(m) == nn.Linear:
            nn.init.eye_(m.weight)
            m.bias.data.fill_(1e-5)

    def save(self, run):
        print('Saving {} to disk..'.format(self.__class__.__name__))
        torch.save(self.state_dict(), self.checkpoint_dir + str(run))

    def load(self, run):
        checkpoint = torch.load(self.checkpoint_dir + str(run))
        self.load_state_dict(checkpoint)


# Discriminator
class Discriminator(nn.Module):
    def __init__(self, latent_size, data_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_size, math.ceil(math.sqrt(data_size))),
            nn.ReLU(),
            nn.Linear(math.ceil(math.sqrt(data_size)), 1),
            nn.Sigmoid()
        )
        # initiliaze the weights
        self.model.apply(self._xavier_init)

        # checkpoint dir to save and load the model
        self.checkpoint_dir = './chkpt/discriminator/'

    def forward(self, input):
        return self.model(input)

    def _xavier_init(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def save(self, run):
        print('Saving {} to disk..'.format(self.__class__.__name__))
        torch.save(self.state_dict(), self.checkpoint_dir + str(run))

    def load(self, run):
        checkpoint = torch.load(self.checkpoint_dir + str(run))
        self.load_state_dict(checkpoint)
