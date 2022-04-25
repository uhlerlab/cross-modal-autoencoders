"""
Utils for data loading and model training.
This code is based on https://github.com/NVlabs/MUNIT.
"""

from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim import lr_scheduler
import torch

torch.cuda.set_device(0)
import os
import math
import yaml
import numpy as np
import torch.nn.init as init

from scipy import sparse
from scipy.stats import percentileofscore
import torch.utils.data as utils
import pickle
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from random import sample
from sklearn import metrics

CONF = {}
data_a = None
data_b = None


directory_prefix = "processed_data/"
DATA_DIRECTORY = directory_prefix + "transcription_factor/"
TEST_SET = [1652, 1563, 1823, 1252, 1453, 189, 1063, 331, 998, 161, 1103,
            1158, 1595, 1459, 892, 1694, 671, 469, 486, 1537, 1308, 960,
            138, 966, 987, 1448, 1466, 1478, 232, 704, 84, 737, 252,
            256, 62, 1439, 336, 1170, 1786, 1277, 1819, 1096, 508, 462,
            1456, 1129, 240, 352, 1716, 629, 593, 951, 1840, 212, 512,
            1172, 980, 1090, 750, 728, 783, 788, 1000, 498, 5, 569,
            572, 50, 1662, 375, 661, 1778, 235, 1607, 110, 1632, 816,
            209, 1798, 1174, 193, 1362, 310, 342, 98, 1538, 405, 1161,
            1310, 1240, 143, 586, 970, 100, 1679, 604, 700, 549, 1464,
            712, 654, 763, 562, 1323, 1445, 150, 507, 956, 1444, 795,
            394, 1530, 895, 582, 274, 350, 459, 57, 384, 446, 828,
            270, 370, 1510, 300, 1101, 1428, 1561, 1857, 1035, 982, 1276,
            63, 780, 1111, 952, 1347, 268, 421, 1574, 1309, 1168, 1060,
            1566, 804, 769, 1528, 743, 494, 847, 1071, 523, 1011, 914,
            1645, 558, 889, 653, 425, 1863, 844, 812, 1859, 1225, 0,
            1582, 170, 1015, 1242, 1826, 1067, 147, 1651, 884, 1628, 1433,
            165, 976, 45, 1838, 602, 28, 1029, 989, 1725, 1724, 936,
            1082, 1442, 307, 1669, 1791, 1553, 1720, 211, 61, 709, 890,
            86, 148, 1159, 675, 1241, 311, 1254, 1175, 990, 306, 1497,
            385, 1514, 499, 168, 374, 747, 1083, 243, 627, 1869, 1619,
            321, 1012, 1868, 864, 393, 1437, 1806, 25, 320, 111, 1598,
            1526, 873, 972, 59, 217, 434, 341, 557, 1135, 60, 1361,
            117, 1543, 407, 665, 1118, 219, 1251, 713, 688, 1304, 482,
            802, 1380, 349, 1221, 785, 1495, 285, 1208, 1192, 770, 679,
            6, 1021, 518, 305, 1492, 805, 135, 1586, 214, 870, 1421,
            1081, 777, 242, 1274, 1712, 1860, 447, 391, 1661, 766, 840,
            1450, 1220, 1766, 1629, 1653, 191, 1590, 162, 862, 1365, 344,
            87, 1673, 1209, 382, 345, 1069, 1400, 1585, 631, 456, 1267,
            1138, 390, 827, 908, 639, 1649, 1845, 142, 1684, 781, 15,
            301, 1288, 1719, 943, 68, 1298, 626, 1621, 617, 746, 146,
            1482, 745, 403, 556, 899, 220, 471, 651, 1018, 1717]
TEST_SET = list(set(TEST_SET))
TRAINING_SET = [x for x in range(1874) if x not in TEST_SET]


with open(DATA_DIRECTORY + "shared_cells.pkl", 'rb') as f:
    shared = pickle.load(f)
gene_exp_cells = pd.read_csv(DATA_DIRECTORY + 'GSM3271040_RNA_sciCAR_A549_cell.txt', index_col=0)
TREATMENT = gene_exp_cells.loc[shared]["treatment_time"]


def load_data(conf, isatac=False, data_size=1874, for_training=True, supervise=[]):
    print(DATA_DIRECTORY)
    log_data = False if "log_data" not in conf else conf["log_data"]
    normalize_data = False if "normalize_data" not in conf else conf["normalize_data"]
    drop = False if "drop" not in conf else conf["drop"]
    # supervise is indices in dataset to drop

    if isatac:
        f = DATA_DIRECTORY + "diff_atac_shared_cells.npz"
    else:
        f = DATA_DIRECTORY + "diff_expr_shared_cells.npz"

    data = sparse.load_npz(f).T.todense()
    # assert (len(data) == data_size)

    if drop:
        threshold = 0.01 if isatac else 0.1
        # threshold = 0 if isatac else 0.1
        acceptable = np.count_nonzero(data, axis=0) > threshold * len(data)
        data = data[:, acceptable.flatten().tolist()[0]]

    if log_data:
        data = np.log1p(data)
        if for_training:
            print("Taking log of data..")

    if normalize_data:
        scaler = StandardScaler()
        training_data = data[TRAINING_SET, :]
        scaler.fit(training_data)
        if for_training:
            print("Normalizing the data..")
        data = scaler.transform(data)

    if not for_training:
        return Variable(torch.from_numpy(data).float()).cuda()

    elif supervise:
        supervised_data = data[supervise, :]
        assert (len(supervised_data) == len(supervise))
        return Variable(torch.from_numpy(supervised_data).float()).cuda()

    else:
        training_data = data[TRAINING_SET, :]
        test_data = data[TEST_SET, :]
        return torch.from_numpy(training_data).float(), torch.from_numpy(test_data).float()


def load_supervision(conf, supervise=0):
    # supervise is fraction of data to supervise
    s = sample(TRAINING_SET, k=int(supervise * len(TRAINING_SET)))
    supervise_a = load_data(conf, isatac=True, supervise=s)
    supervise_b = load_data(conf, isatac=False, supervise=s)
    return supervise_a, supervise_b


def get_all_data_loaders(conf):
    global CONF
    global data_a
    global data_b
    CONF = conf

    data_a = load_data_for_latent_space_plot(isatac=True)
    # # a is atac
    data_b = load_data_for_latent_space_plot(isatac=False)
    # b is expression

    labels = [i if i != 3 else 2 for i in TREATMENT]
    training_labels = torch.from_numpy(np.array(labels)[TRAINING_SET]).long()
    test_labels = torch.from_numpy(np.array(labels)[TEST_SET]).long()

    assert 1 in training_labels and 2 in training_labels and 0 in training_labels

    train, test = load_data(conf, isatac=True)
    batch_size = conf['batch_size']

    train_dataset = utils.TensorDataset(train, training_labels)
    train_loader_a = utils.DataLoader(train_dataset, batch_size=batch_size)

    test_dataset = utils.TensorDataset(test, test_labels)
    test_loader_a = utils.DataLoader(test_dataset, batch_size=batch_size)

    train, test = load_data(conf, isatac=False)

    train_dataset = utils.TensorDataset(train, training_labels)
    train_loader_b = utils.DataLoader(train_dataset, batch_size=batch_size)

    test_dataset = utils.TensorDataset(test, test_labels)
    test_loader_b = utils.DataLoader(test_dataset, batch_size=batch_size)

    return train_loader_a, train_loader_b, test_loader_a, test_loader_b


def get_config(config):
    # Note need to have pip install pyyaml==5.4.1
    with open(config, 'r') as stream:
        return yaml.load(stream)


def prepare_sub_folder(output_directory):
    image_directory = os.path.join(output_directory, 'images')
    if not os.path.exists(image_directory):
        print("Creating directory: {}".format(image_directory))
        os.makedirs(image_directory)
    checkpoint_directory = os.path.join(output_directory, 'checkpoints')
    if not os.path.exists(checkpoint_directory):
        print("Creating directory: {}".format(checkpoint_directory))
        os.makedirs(checkpoint_directory)
    return checkpoint_directory, image_directory


def write_loss(iterations, trainer, train_writer):
    members = [attr for attr in dir(trainer) \
               if not callable(getattr(trainer, attr)) and not attr.startswith("__") and (
                       'loss' in attr or 'grad' in attr or 'nwd' in attr)]
    for m in members:
        train_writer.add_scalar(m, getattr(trainer, m), iterations + 1)


# Get model list for resume
def get_model_list(dirname, key):
    if os.path.exists(dirname) is False:
        return None
    gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                  os.path.isfile(os.path.join(dirname, f)) and key in f and ".pt" in f]
    if gen_models is None:
        return None
    gen_models.sort()
    last_model_name = gen_models[-1]
    return last_model_name


def get_scheduler(optimizer, hyperparameters, iterations=-1):
    if 'lr_policy' not in hyperparameters or hyperparameters['lr_policy'] == 'constant':
        scheduler = None  # constant scheduler
    elif hyperparameters['lr_policy'] == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=hyperparameters['step_size'],
                                        gamma=hyperparameters['gamma'], last_epoch=iterations)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', hyperparameters['lr_policy'])
    return scheduler


def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            # print m.__class__.__name__
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    return init_fun


# Code for plotting latent space.

def load_data_for_latent_space_plot(isatac=False):
    conf = CONF
    return load_data(conf, isatac=isatac, for_training=False)


def plot_pca(a,b, outname1=None, outname2=None, outname=None, scale=True):
    matrix = np.vstack((b, a))
    pca = PCA(n_components=2)
    scaled = matrix.copy()
    if scale:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(matrix)

    comp = pca.fit_transform(scaled)

    half = len(a)
    fig, ax = matplotlib.pyplot.subplots()
    sc = ax.scatter(comp[:, 0][0:half], comp[:, 1][0:half], c=TREATMENT.values, s=1)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    cbar = fig.colorbar(sc)
    cbar.ax.set_ylabel('Treatment time')
    plt.savefig(outname1)
    plt.close("all")

    fig, ax = matplotlib.pyplot.subplots()
    sc = ax.scatter(comp[:, 0][half:], comp[:, 1][half:], c=TREATMENT.values, s=1)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    cbar = fig.colorbar(sc)
    cbar.ax.set_ylabel('Treatment time')
    plt.savefig(outname2)
    plt.close("all")

    fig, ax = matplotlib.pyplot.subplots()
    # make atac yellow
    colors = ['purple'] * len(TREATMENT.values) + ["yellow"] * len(TREATMENT.values)
    sc = ax.scatter(comp[:, 0], comp[:, 1], c=colors, s=1)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    #plt.legend(('RNA-seq', 'ATAC-seq'))
    plt.savefig(outname)
    plt.close("all")

def plot_pca_both_spaces(a, b, outname, scale=True):
    matrix = np.vstack((b, a))
    pca = PCA(n_components=2)
    scaled = matrix.copy()
    if scale:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(matrix)

    comp = pca.fit_transform(scaled)

    fig, ax = matplotlib.pyplot.subplots()
    # make atac yellow and
    colors = ['purple'] * len(TREATMENT.values) + ["yellow"] * len(TREATMENT.values)
    sc = ax.scatter(comp[:, 0], comp[:, 1], c=colors, s=1)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.legend()
    plt.savefig(outname)
    plt.close("all")


def save_plots(trainer, directory, suffix):
    latent_a = trainer.gen_a.enc(data_a).data.cpu().numpy()
    latent_b = trainer.gen_b.enc(data_b).data.cpu().numpy()

    #plot_pca(latent_a, os.path.join(directory, "_a_" + suffix + ".png"))
    #plot_pca(latent_b, os.path.join(directory, "_b_" + suffix + ".png"))    
    #plot_pca_both_spaces(latent_a, latent_b, os.path.join(directory, "both_" + suffix + ".png"))

    plot_pca(latent_a, latent_b, os.path.join(directory, "_a_" + suffix + ".png"), os.path.join(directory, "_b_" + suffix + ".png"), os.path.join(directory, "both_" + suffix + ".png"))

def write_knn(trainer, directory, suffix):
    latent_a = trainer.gen_a.enc(data_a).data.cpu().numpy()
    latent_b = trainer.gen_b.enc(data_b).data.cpu().numpy()

    for k in [5, 50]:
        accuracy_a_train, accuracy_a_test = knn_accuracy(latent_a, latent_b, k)
        accuracy_b_train, accuracy_b_test = knn_accuracy(latent_b, latent_a, k)
        output = "Iteration: {}\n {}NN accuracy A: train: {} test: {}\n {}NN accuracy B: train: {} test: {}\n".format(
            suffix, str(k), accuracy_a_train, accuracy_a_test, str(k), accuracy_b_train, accuracy_b_test)
        print(output)
        with open(os.path.join(directory, "knn_accuracy.txt"), "a") as myfile:
            myfile.write(output)

def knn_accuracy(A, B, k):
    nn = NearestNeighbors(k, metric="l1")
    nn.fit(A, k)
    transp_nearest_neighbor = nn.kneighbors(B, 1, return_distance=False)
    actual_nn = nn.kneighbors(A, k, return_distance=False)
    train_correct = 0
    test_correct = 0

    for i in range(len(transp_nearest_neighbor)):
        if transp_nearest_neighbor[i] not in actual_nn[i]:
            continue
        elif i in TEST_SET:
            test_correct += 1
        else:
            train_correct += 1

    return train_correct / len(TRAINING_SET), test_correct / len(TEST_SET)