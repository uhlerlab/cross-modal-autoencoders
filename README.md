# 1. Installation instructions

Packages are listed in environment.yml file and can be installed using Anaconda/Miniconda:

```bash
conda env create -f environment.yml
conda activate pytorch
```
This code was tested on NVIDIA GTX 1080TI GPU.

# 2. Usage instructions

Training the image autoencoder with classifier in latent space:

```
python train_ae.py --save-dir <path/to/save/dir> --conditional
```

Integrating the RNA autoencoder with conditional discriminator in latent space:

```
python train_rna_image.py --save-dir <path/to/save/dir> --pretrained-file <path/to/pretrained/image/autoencoder> --conditional-adv
```

# 3. Expected output 

Output is log file and PyTorch checkpoint files when code is run on gene expression and imaging data.
