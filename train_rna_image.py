import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable

from dataloader import RNA_Dataset
from dataloader import NucleiDatasetNew as NucleiDataset
from model import FC_Autoencoder, FC_Classifier, VAE, FC_VAE, Simple_Classifier

import os
import argparse
import numpy as np
import imageio

torch.manual_seed(1)

#============ PARSE ARGUMENTS =============

def setup_args():

    options = argparse.ArgumentParser()

    # save and directory options
    options.add_argument('-sd', '--save-dir', action="store", dest="save_dir")
    options.add_argument('--save-freq', action="store", dest="save_freq", default=20, type=int)
    options.add_argument('--pretrained-file', action="store")

    # training parameters
    options.add_argument('-bs', '--batch-size', action="store", dest="batch_size", default=32, type=int)
    options.add_argument('-w', '--num-workers', action="store", dest="num_workers", default=10, type=int)
    options.add_argument('-lrAE', '--learning-rate-AE', action="store", dest="learning_rate_AE", default=1e-4, type=float)
    options.add_argument('-lrD', '--learning-rate-D', action="store", dest="learning_rate_D", default=1e-4, type=float)
    options.add_argument('-e', '--max-epochs', action="store", dest="max_epochs", default=1000, type=int)
    options.add_argument('-wd', '--weight-decay', action="store", dest="weight_decay", default=0, type=float)
    options.add_argument('--train-imagenet', action="store_true")
    options.add_argument('--conditional', action="store_true")
    options.add_argument('--conditional-adv', action="store_true")

    # hyperparameters
    options.add_argument('--alpha', action="store", default=0.1, type=float)
    options.add_argument('--beta', action="store", default=1., type=float)
    options.add_argument('--lamb', action="store", default=0.00000001, type=float)
    options.add_argument('--latent-dims', action="store", default=128, type=int)

    # gpu options
    options.add_argument('-gpu', '--use-gpu', action="store_false", dest="use_gpu")

    return options.parse_args()


args = setup_args()
if not torch.cuda.is_available():
    args.use_gpu = False

os.makedirs(args.save_dir, exist_ok=True)

#============= TRAINING INITIALIZATION ==============

# initialize autoencoder
netRNA = FC_VAE(n_input=7633, nz=args.latent_dims)

netImage = VAE(latent_variable_size=args.latent_dims, batchnorm=True)
netImage.load_state_dict(torch.load(args.pretrained_file))
print("Pre-trained model loaded from %s" % args.pretrained_file)

if args.conditional_adv: 
    netClf = FC_Classifier(nz=args.latent_dims+10)
    assert(not args.conditional)
else:
    netClf = FC_Classifier(nz=args.latent_dims)

if args.conditional:
    netCondClf = Simple_Classifier(nz=args.latent_dims)

if args.use_gpu:
    netRNA.cuda()
    netImage.cuda()
    netClf.cuda()
    if args.conditional:
        netCondClf.cuda()

# load data
genomics_dataset = RNA_Dataset(datadir="data/nCD4_gene_exp_matrices/")
image_dataset = NucleiDataset(datadir="data/nuclear_crops_all_experiments", mode='test')

image_loader = torch.utils.data.DataLoader(image_dataset, batch_size=args.batch_size, drop_last=True, shuffle=True)
genomics_loader = torch.utils.data.DataLoader(genomics_dataset, batch_size=args.batch_size, drop_last=True, shuffle=True)

# setup optimizer
opt_netRNA = optim.Adam(list(netRNA.parameters()), lr=args.learning_rate_AE)
opt_netClf = optim.Adam(list(netClf.parameters()), lr=args.learning_rate_D, weight_decay=args.weight_decay)
opt_netImage = optim.Adam(list(netImage.parameters()), lr=args.learning_rate_AE)

if args.conditional:
    opt_netCondClf = optim.Adam(list(netCondClf.parameters()), lr=args.learning_rate_AE)

# loss criteria
criterion_reconstruct = nn.MSELoss()
criterion_classify = nn.CrossEntropyLoss()

# setup logger
with open(os.path.join(args.save_dir, 'log.txt'), 'w') as f:
    print(args, file=f)
    print(netRNA, file=f)
    print(netImage, file=f)
    print(netClf, file=f)
    if args.conditional:
        print(netCondClf, file=f)

# define helper train functions

def compute_KL_loss(mu, logvar):
    if args.lamb>0:
        KLloss = -0.5*torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return args.lamb * KLloss
    return 0

def train_autoencoders(rna_inputs, image_inputs, rna_class_labels=None, image_class_labels=None):
   
    netRNA.train()
    if args.train_imagenet:
        netImage.train()
    else:
        netImage.eval()
    netClf.eval()
    if args.conditional:
        netCondClf.train()
    
    # process input data
    rna_inputs, image_inputs = Variable(rna_inputs), Variable(image_inputs)

    if args.use_gpu:
        rna_inputs, image_inputs = rna_inputs.cuda(), image_inputs.cuda()

    # reset parameter gradients
    netRNA.zero_grad()

    # forward pass
    rna_recon, rna_latents, rna_mu, rna_logvar = netRNA(rna_inputs)
    image_recon, image_latents, image_mu, image_logvar = netImage(image_inputs)

    if args.conditional_adv:
        rna_class_labels, image_class_labels = rna_class_labels.cuda(), image_class_labels.cuda()
        rna_scores = netClf(torch.cat((rna_latents, rna_class_labels.float().view(-1,1).expand(-1,10)), dim=1))
        image_scores = netClf(torch.cat((image_latents, image_class_labels.float().view(-1,1).expand(-1,10)), dim=1))
    else:
        rna_scores = netClf(rna_latents)
        image_scores = netClf(image_latents)

    rna_labels = torch.zeros(rna_scores.size(0),).long()
    image_labels = torch.ones(image_scores.size(0),).long()

    if args.conditional:
        rna_class_scores = netCondClf(rna_latents)
        image_class_scores = netCondClf(image_latents)
    
    if args.use_gpu:
        rna_labels, image_labels = rna_labels.cuda(), image_labels.cuda()
        if args.conditional:
            rna_class_labels, image_class_labels = rna_class_labels.cuda(), image_class_labels.cuda()

    # compute losses
    rna_recon_loss = criterion_reconstruct(rna_inputs, rna_recon)
    image_recon_loss = criterion_reconstruct(image_inputs, image_recon)
    kl_loss = compute_KL_loss(rna_mu, rna_logvar) + compute_KL_loss(image_mu, image_logvar)
    clf_loss = 0.5*criterion_classify(rna_scores, image_labels) + 0.5*criterion_classify(image_scores, rna_labels)
    loss = args.alpha*(rna_recon_loss + image_recon_loss) + clf_loss + kl_loss

    if args.conditional:
        clf_class_loss = 0.5*criterion_classify(rna_class_scores, rna_class_labels) + 0.5*criterion_classify(image_class_scores, image_class_labels)
        loss += args.beta*clf_class_loss

    # backpropagate and update model
    loss.backward()
    opt_netRNA.step()
    if args.conditional:
        opt_netCondClf.step()

    if args.train_imagenet:
        opt_netImage.step()

    summary_stats = {'rna_recon_loss': rna_recon_loss.item()*rna_scores.size(0), 'image_recon_loss': image_recon_loss.item()*image_scores.size(0), 
            'clf_loss': clf_loss.item()*(rna_scores.size(0)+image_scores.size(0))}
    
    if args.conditional:
        summary_stats['clf_class_loss'] = clf_class_loss.item()*(rna_scores.size(0)+image_scores.size(0))

    return summary_stats

def train_classifier(rna_inputs, image_inputs, rna_class_labels=None, image_class_labels=None):
    
    netRNA.eval()
    netImage.eval()
    netClf.train()

    # process input data
    rna_inputs, image_inputs = Variable(rna_inputs), Variable(image_inputs)

    if args.use_gpu:
        rna_inputs, image_inputs = rna_inputs.cuda(), image_inputs.cuda()
    
    # reset parameter gradients
    netClf.zero_grad()

    # forward pass
    _, rna_latents, _, _ = netRNA(rna_inputs)
    _, image_latents, _, _ = netImage(image_inputs)

    if args.conditional_adv:
        rna_class_labels, image_class_labels = rna_class_labels.cuda(), image_class_labels.cuda()
        rna_scores = netClf(torch.cat((rna_latents, rna_class_labels.float().view(-1,1).expand(-1,10)), dim=1))
        image_scores = netClf(torch.cat((image_latents, image_class_labels.float().view(-1,1).expand(-1,10)), dim=1))
    else:
        rna_scores = netClf(rna_latents)
        image_scores = netClf(image_latents)

    rna_labels = torch.zeros(rna_scores.size(0),).long()
    image_labels = torch.ones(image_scores.size(0),).long()
    
    if args.use_gpu:
        rna_labels, image_labels = rna_labels.cuda(), image_labels.cuda()

    # compute losses
    clf_loss = 0.5*criterion_classify(rna_scores, rna_labels) + 0.5*criterion_classify(image_scores, image_labels)

    loss = clf_loss

    # backpropagate and update model
    loss.backward()
    opt_netClf.step()

    summary_stats = {'clf_loss': clf_loss*(rna_scores.size(0)+image_scores.size(0)), 'rna_accuracy': accuracy(rna_scores, rna_labels), 'rna_n_samples': rna_scores.size(0),
            'image_accuracy': accuracy(image_scores, image_labels), 'image_n_samples': image_scores.size(0)}

    return summary_stats

def accuracy(output, target):
    pred = output.argmax(dim=1).view(-1)
    correct = pred.eq(target.view(-1)).float().sum().item()
    return correct

def generate_image(epoch):
    img_dir = os.path.join(args.save_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    netRNA.eval()
    netImage.eval()

    for i in range(5):
        samples = genomics_loader.dataset[np.random.randint(30)]
        rna_inputs = samples['tensor']
        rna_inputs = Variable(rna_inputs.unsqueeze(0))
        samples = image_loader.dataset[np.random.randint(30)]
        image_inputs = samples['image_tensor']
        image_inputs = Variable(image_inputs.unsqueeze(0))
 
        if torch.cuda.is_available():
            rna_inputs = rna_inputs.cuda()
            image_inputs = image_inputs.cuda()
 
        _, rna_latents, _, _ = netRNA(rna_inputs)
        recon_inputs = netImage.decode(rna_latents)
        imageio.imwrite(os.path.join(img_dir, "epoch_%s_trans_%s.jpg" % (epoch, i)), np.uint8(recon_inputs.cpu().data.view(64,64).numpy()*255))
        recon_images, _, _, _ = netImage(image_inputs)
        imageio.imwrite(os.path.join(img_dir, "epoch_%s_recon_%s.jpg" % (epoch, i)), np.uint8(recon_images.cpu().data.view(64,64).numpy()*255))
 
### main training loop
for epoch in range(args.max_epochs):
    print(epoch)

    if epoch % args.save_freq == 0:
        generate_image(epoch)

    recon_rna_loss = 0
    recon_image_loss = 0
    clf_loss = 0
    clf_class_loss = 0
    AE_clf_loss = 0

    n_rna_correct = 0
    n_rna_total = 0
    n_atac_correct = 0
    n_atac_total = 0

    for idx, (rna_samples, image_samples) in enumerate(zip(genomics_loader, image_loader)):
        rna_inputs = rna_samples['tensor']
        image_inputs = image_samples['image_tensor']

        if args.conditional or args.conditional_adv:
            rna_labels = rna_samples['binary_label']
            image_labels = image_samples['binary_label']
            out = train_autoencoders(rna_inputs, image_inputs, rna_labels, image_labels)
        else:
            out = train_autoencoders(rna_inputs, image_inputs)

        recon_rna_loss += out['rna_recon_loss']
        recon_image_loss += out['image_recon_loss']
        AE_clf_loss += out['clf_loss']

        if args.conditional:
            clf_class_loss += out['clf_class_loss']
        
        if args.conditional_adv:
            out = train_classifier(rna_inputs, image_inputs, rna_labels, image_labels)
        else:
            out = train_classifier(rna_inputs, image_inputs)

        clf_loss += out['clf_loss']
        n_rna_correct += out['rna_accuracy']
        n_rna_total += out['rna_n_samples']
        n_atac_correct += out['image_accuracy']
        n_atac_total += out['image_n_samples']

    recon_rna_loss /= n_rna_total
    clf_loss /= n_rna_total+n_atac_total
    AE_clf_loss /= n_rna_total+n_atac_total

    if args.conditional:
        clf_class_loss /= n_rna_total + n_atac_total

    with open(os.path.join(args.save_dir, 'log.txt'), 'a') as f:
        print('Epoch: ', epoch, ', rna recon loss: %.8f' % float(recon_rna_loss), ', image recon loss: %.8f' % float(recon_image_loss),
                ', AE clf loss: %.8f' % float(AE_clf_loss), ', clf loss: %.8f' % float(clf_loss), ', clf class loss: %.8f' % float(clf_class_loss),
                ', clf accuracy RNA: %.4f' % float(n_rna_correct / n_rna_total), ', clf accuracy ATAC: %.4f' % float(n_atac_correct / n_atac_total), file=f)

    # save model
    if epoch % args.save_freq == 0:
        torch.save(netRNA.cpu().state_dict(), os.path.join(args.save_dir,"netRNA_%s.pth" % epoch))
        torch.save(netImage.cpu().state_dict(), os.path.join(args.save_dir,"netImage_%s.pth" % epoch))
        torch.save(netClf.cpu().state_dict(), os.path.join(args.save_dir,"netClf_%s.pth" % epoch))
        if args.conditional:
            torch.save(netCondClf.cpu().state_dict(), os.path.join(args.save_dir,"netCondClf_%s.pth" % epoch))

    if args.use_gpu:
        netRNA.cuda()
        netClf.cuda()
        netImage.cuda()
        if args.conditional:
            netCondClf.cuda()

