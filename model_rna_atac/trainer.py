"""
Trainer class for model training.
This code is based on https://github.com/NVlabs/MUNIT.
"""
from networks import Discriminator, Classifier
from networks import VAEGen_MORE_LAYERS as VAEGen
from utils import weights_init, get_model_list, get_scheduler
from torch.autograd import Variable
import torch
import torch.nn as nn
import os


class Trainer(nn.Module):
    def __init__(self, hyperparameters):
        super(Trainer, self).__init__()
        lr = hyperparameters['lr']
        # Initiate the networks
        shared_layer = False
        if "shared_layer" in hyperparameters and hyperparameters["shared_layer"]:
            shared_layer = {}
            shared_layer["dec"] = nn.Linear(hyperparameters['gen']['latent'], hyperparameters['gen']['dim'])
            shared_layer["enc"] = nn.Linear(hyperparameters['gen']['dim'], hyperparameters['gen']['latent'])

        self.gen_a = VAEGen(hyperparameters['input_dim_a'], hyperparameters['gen'],
                            shared_layer)  # auto-encoder for domain a
        self.gen_b = VAEGen(hyperparameters['input_dim_b'], hyperparameters['gen'],
                            shared_layer)  # auto-encoder for domain b
        self.dis_latent = Discriminator(hyperparameters['gen']['latent'],
                                        hyperparameters['dis'])  # discriminator for latent space

        self.classifier = Classifier(hyperparameters['gen']['latent'])  # classifier on the latent space

        # Setup the optimizers
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']
        dis_params = list(self.dis_latent.parameters())
        gen_params = list(self.gen_a.parameters()) + list(self.gen_b.parameters()) + list(self.classifier.parameters())
        self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)

        # Network weight initialization
        self.apply(weights_init(hyperparameters['init']))
        self.dis_latent.apply(weights_init('gaussian'))

    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))

    def super_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))

    def forward(self, x_a, x_b):
        self.eval()
        h_a, _ = self.gen_a.encode(x_a)
        h_b, _ = self.gen_b.encode(x_b)
        x_ba = self.gen_a.decode(h_b)
        x_ab = self.gen_b.decode(h_a)
        self.train()
        return x_ab, x_ba

    def gen_update(self, x_a, x_b, a_labels, b_labels, super_a, super_b, hyperparameters, variational=True):
        true_samples = Variable(
            torch.randn(200, hyperparameters['gen']['latent']),
            requires_grad=False
        ).cuda()

        self.gen_opt.zero_grad()
        # encode
        h_a, n_a = self.gen_a.encode(x_a)
        h_b, n_b = self.gen_b.encode(x_b)
        # decode (within domain)
        if variational:
            h_a = h_a + n_a
            h_b = h_b + n_b

        x_a_recon = self.gen_a.decode(h_a)
        x_b_recon = self.gen_b.decode(h_b)

        # decode (cross domain)
        x_ba = self.gen_a.decode(h_b)
        x_ab = self.gen_b.decode(h_a)
        # encode again
        h_b_recon, n_b_recon = self.gen_a.encode(x_ba)
        h_a_recon, n_a_recon = self.gen_b.encode(x_ab)
        # decode again (if needed)
        if variational:
            h_a_recon = h_a_recon + n_a_recon
            h_b_recon = h_b_recon + n_b_recon

        classes_a = self.classifier.forward(h_a)
        classes_b = self.classifier.forward(h_b)

        # reconstruction loss
        self.loss_gen_recon_x_a = self.recon_criterion(x_a_recon, x_a)
        self.loss_gen_recon_x_b = self.recon_criterion(x_b_recon, x_b)

        # GAN loss
        self.loss_latent_a = self.dis_latent.calc_gen_loss(h_a)
        self.loss_latent_b = self.dis_latent.calc_gen_loss_reverse(h_b)

        # Classification Loss
        self.loss_class_a = self.classifier.class_loss(classes_a, a_labels)
        self.loss_class_b = self.classifier.class_loss(classes_b, b_labels)

        # supervision
        s_a, n_a = self.gen_a.encode(super_a)
        s_b, n_b = self.gen_b.encode(super_b)

        self.loss_supervision = self.super_criterion(s_a, s_b)

        class_weight = hyperparameters['gan_w'] if "class_w" not in hyperparameters else hyperparameters["class_w"]

        # total loss
        self.loss_gen_total = hyperparameters['gan_w'] * self.loss_latent_a + \
                              hyperparameters['gan_w'] * self.loss_latent_b + \
                              class_weight * self.loss_class_a + \
                              class_weight * self.loss_class_b + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_a + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_b + \
                              hyperparameters['super_w'] * self.loss_supervision

        if variational:
            self.loss_gen_total += hyperparameters['recon_kl_w'] * self.loss_gen_recon_kl_a + \
                                   hyperparameters['recon_kl_w'] * self.loss_gen_recon_kl_b

        self.loss_gen_total.backward()
        self.gen_opt.step()

    def sample(self, x_a, x_b):
        self.eval()
        x_a_recon, x_b_recon, x_ba, x_ab = [], [], [], []
        for i in range(x_a.size(0)):
            h_a, _ = self.gen_a.encode(x_a[i].unsqueeze(0))
            h_b, _ = self.gen_b.encode(x_b[i].unsqueeze(0))
            x_a_recon.append(self.gen_a.decode(h_a))
            x_b_recon.append(self.gen_b.decode(h_b))
            x_ba.append(self.gen_a.decode(h_b))
            x_ab.append(self.gen_b.decode(h_a))
        x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)
        x_ba = torch.cat(x_ba)
        x_ab = torch.cat(x_ab)
        self.train()
        return x_a, x_a_recon, x_ab, x_b, x_b_recon, x_ba

    def dis_update(self, x_a, x_b, hyperparameters):
        self.dis_opt.zero_grad()
        # encode
        h_a, n_a = self.gen_a.encode(x_a)
        h_b, n_b = self.gen_b.encode(x_b)
        # D loss
        self.loss_dis_latent = self.dis_latent.calc_dis_loss(h_a, h_b)
        self.loss_dis_total = hyperparameters['gan_w'] * (self.loss_dis_latent)
        self.loss_dis_total.backward()
        self.dis_opt.step()

    def update_learning_rate(self):
        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()

    def resume(self, checkpoint_dir, hyperparameters):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen")
        state_dict = torch.load(last_model_name)
        self.gen_a.load_state_dict(state_dict['a'])
        self.gen_b.load_state_dict(state_dict['b'])
        iterations = int(last_model_name[-11:-3])
        # Load discriminators
        last_model_name = get_model_list(checkpoint_dir, "dis")
        state_dict = torch.load(last_model_name)
        self.dis_latent.load_state_dict(state_dict['latent'])
        # Load optimizers
        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
        self.dis_opt.load_state_dict(state_dict['dis'])
        self.gen_opt.load_state_dict(state_dict['gen'])
        # Reinitilize schedulers
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters, iterations)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters, iterations)
        print('Resume from iteration %d' % iterations)
        return iterations

    def save(self, snapshot_dir, iterations):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
        dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
        torch.save(
            {'a': self.gen_a.state_dict(), 'b': self.gen_b.state_dict(), "classifier": self.classifier.state_dict()},
            gen_name)
        torch.save({'latent': self.dis_latent.state_dict()}, dis_name)
        torch.save({'gen': self.gen_opt.state_dict(), 'dis': self.dis_opt.state_dict()}, opt_name)


