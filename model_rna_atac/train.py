"""
Main functionality for starting training.
This code is based on https://github.com/NVlabs/MUNIT.
"""
import torch

torch.cuda.set_device(0)
from utils import get_all_data_loaders, prepare_sub_folder, write_loss, get_config, save_plots, load_supervision, \
    write_knn
import argparse
from torch.autograd import Variable
from trainer import Trainer
import torch.backends.cudnn as cudnn

try:
    from itertools import izip as zip
except ImportError:
    pass
import os
import sys
import tensorboardX
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='.', help="outputs path")
parser.add_argument("--resume", action="store_true")
opts = parser.parse_args()

cudnn.benchmark = True

# Load experiment setting
config = get_config(opts.config)
max_iter = config['max_iter']

trainer = Trainer(config)

trainer.cuda()

train_loader_a, train_loader_b, test_loader_a, test_loader_b = get_all_data_loaders(config)
super_a, super_b = load_supervision(config, supervise=config["supervise"])


# Setup logger and output folders
model_name = os.path.splitext(os.path.basename(opts.config))[0]
train_writer = tensorboardX.SummaryWriter(os.path.join(opts.output_path + "/logs", model_name))
output_directory = os.path.join(opts.output_path + "/outputs", model_name)
checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml'))  # copy config file to output folder

# Start training
iterations = trainer.resume(checkpoint_directory, hyperparameters=config) if opts.resume else 0
num_disc = 1 if "num_disc" not in config else config["num_disc"]
num_gen = 1 if "num_gen" not in config else config["num_gen"]

while True:
    for it, (images_a, images_b) in enumerate(zip(train_loader_a, train_loader_b)):
        trainer.update_learning_rate()
        labels_a, labels_b = Variable(images_a[1]).cuda(), Variable(images_b[1]).cuda()
        images_a, images_b = Variable(images_a[0]).cuda(), Variable(images_b[0]).cuda()
        # Main training code

        for _ in range(num_disc):
            trainer.dis_update(images_a, images_b, config)
        for _ in range(num_gen):
            trainer.gen_update(images_a, images_b, labels_a, labels_b, super_a, super_b, config, variational=False)
        torch.cuda.synchronize()

    # Dump training stats in log file
    if (iterations + 1) % config['log_iter'] == 0:
        # print("Iteration: %08d/%08d" % (iterations + 1, max_iter))
        write_loss(iterations, trainer, train_writer)
        write_knn(trainer, image_directory, str(iterations))

    # Save network weights
    if (iterations + 1) % config['snapshot_save_iter'] == 0:
        trainer.save(checkpoint_directory, iterations)
        save_plots(trainer, image_directory, str(iterations))

    iterations += 1
    if iterations >= max_iter:
        sys.exit('Finish training')
