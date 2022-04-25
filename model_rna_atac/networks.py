"""
Defines models.
This code is based on https://github.com/NVlabs/MUNIT.
"""
from torch import nn
from torch.autograd import Variable
import torch
import torch.nn.functional as F

try:
    from itertools import izip as zip
except ImportError:  # will be 3.x series
    pass


##################################################################################
# Discriminator
##################################################################################

class Discriminator(nn.Module):

    def __init__(self, input_dim, params):
        super(Discriminator, self).__init__()
        self.gan_type = params['gan_type']
        self.dim = params['dim']
        self.norm = params['norm']
        self.input_dim = input_dim
        self.net = self._make_net()

    def _make_net(self):
        return nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.input_dim, self.dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.dim, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.net(x)

    def calc_dis_loss(self, input_fake, input_real):
        # calculate the loss to train D
        outs0 = [self.forward(input_fake)]
        outs1 = [self.forward(input_real)]
        loss = 0

        for it, (out0, out1) in enumerate(zip(outs0, outs1)):
            loss += torch.mean((out0 - 0) ** 2) + torch.mean((out1 - 1) ** 2)
        return loss

    def calc_gen_loss(self, input_fake):
        # calculate the loss to train G
        outs0 = [self.forward(input_fake)]
        loss = 0
        for it, (out0) in enumerate(outs0):
            # 1 = real data
            loss += torch.mean((out0 - 1) ** 2)
        return loss

    def calc_gen_loss_reverse(self, input_real):
        # calculate the loss to train G
        outs0 = [self.forward(input_real)]
        loss = 0
        for it, (out0) in enumerate(outs0):
            # 0 = fake data
            loss += torch.mean((out0 - 0) ** 2)
        return loss

    def calc_gen_loss_half(self, input_fake):
        # calculate the loss to train G
        outs0 = [self.forward(input_fake)]
        loss = 0
        for it, (out0) in enumerate(outs0):
            if self.gan_type == 'lsgan':
                loss += torch.mean((out0 - 0.5) ** 2)
            elif self.gan_type == 'nsgan':
                all1 = Variable(torch.ones_like(out0.data).cuda(), requires_grad=False)
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all1))
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss


##################################################################################
# Generator
##################################################################################

class VAEGen_MORE_LAYERS(nn.Module):
    # VAE architecture
    def __init__(self, input_dim, params, shared_layer=False):
        super(VAEGen_MORE_LAYERS, self).__init__()
        self.dim = params['dim']
        self.latent = params['latent']
        self.input_dim = input_dim

        encoder_layers = [nn.Linear(self.input_dim, self.input_dim),
                          nn.LeakyReLU(0.2, inplace=True),
                          nn.Linear(self.input_dim, self.input_dim),
                          nn.LeakyReLU(0.2, inplace=True),
                          nn.Linear(self.input_dim, self.input_dim),
                          nn.LeakyReLU(0.2, inplace=True),
                          nn.Linear(self.input_dim, self.dim),
                          nn.LeakyReLU(0.2, inplace=True)]

        decoder_layers = [nn.LeakyReLU(0.2, inplace=True),
                          nn.Linear(self.dim, self.input_dim),
                          nn.LeakyReLU(0.2, inplace=True),
                          nn.Linear(self.input_dim, self.input_dim),
                          nn.LeakyReLU(0.2, inplace=True),
                          nn.Linear(self.input_dim, self.input_dim),
                          nn.LeakyReLU(0.2, inplace=True),
                          nn.Linear(self.input_dim, self.input_dim),
                          nn.LeakyReLU(0.2, inplace=True)]

        if shared_layer:
            encoder_layers += [shared_layer["enc"], nn.LeakyReLU(0.2, inplace=True)]
            decoder_layers = [shared_layer["dec"]] + decoder_layers
        else:
            encoder_layers += [nn.Linear(self.dim, self.latent), nn.LeakyReLU(0.2, inplace=True)]
            decoder_layers = [nn.Linear(self.latent, self.dim)] + decoder_layers
        self.enc = nn.Sequential(*encoder_layers)
        self.dec = nn.Sequential(*decoder_layers)

    def forward(self, images):
        # This is a reduced VAE implementation where we assume the outputs are multivariate Gaussian distribution with mean = hiddens and std_dev = all ones.
        hiddens = self.encode(images)
        if self.training == True:
            noise = Variable(torch.randn(hiddens.size()).cuda(hiddens.data.get_device()))
            images_recon = self.decode(hiddens + noise)
        else:
            images_recon = self.decode(hiddens)
        return images_recon, hiddens

    def encode(self, images):
        hiddens = self.enc(images)
        noise = Variable(torch.randn(hiddens.size()).cuda(hiddens.data.get_device()))
        return hiddens, noise

    def decode(self, hiddens):
        images = self.dec(hiddens)
        return images

##################################################################################
# Classifier
##################################################################################

class Classifier(nn.Module):
    def __init__(self, input_dim):
        super(Classifier, self).__init__()
        self.input_dim = input_dim
        self.net = self._make_net()

        self.cel = nn.CrossEntropyLoss()

    def _make_net(self):
        return nn.Sequential(
            nn.Linear(self.input_dim, 3)
        )

    def forward(self, x):
        return self.net(x)

    def class_loss(self, input, target):
        return self.cel(input, target)


