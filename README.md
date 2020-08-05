# Multi-Domain Translation between Single-Cell Imaging and Sequencing Data using Autoencoders

This is the accompanying code for the paper, "Multi-Domain Translation between Single-Cell Imaging and Sequencing Data using Autoencoders" ([bioRxiv](https://www.biorxiv.org/content/10.1101/2019.12.13.875922v1.full))

The preprocessing scripts for the raw data files can be found at this repo ([link](https://github.com/SaradhaVenkatachalapathy/Radial_chromatin_packing_immune_cells)).

The preprocessed data files can be downloaded from Google Drive ([link](https://drive.google.com/file/d/1B7PFrGnosx-Cono9x8G0DCkNuK4UFw1X/view?usp=sharing)).


## 1. Installation instructions

Packages are listed in environment.yml file and can be installed using Anaconda/Miniconda:

```bash
conda env create -f environment.yml
conda activate pytorch
```
This code was tested on NVIDIA GTX 1080TI GPU.

## 2. Usage instructions

Training the image autoencoder with classifier in latent space:

```
python train_ae.py --save-dir <path/to/save/dir> --conditional
```

Integrating the RNA autoencoder with conditional discriminator in latent space:

```
python train_rna_image.py --save-dir <path/to/save/dir> --pretrained-file <path/to/pretrained/image/autoencoder> --conditional-adv
```

## 3. Expected output 

Output is log file and PyTorch checkpoint files when code is run on gene expression and imaging data.
