# Multi-Domain Translation between Single-Cell Imaging and Sequencing Data using Autoencoders

Code for training RNA-seq - ATAC-seq models.

## Usage instructions

```bash
python train.py --config configs/cross_modal_supervision_100.yaml --output_path outputs
```

## Expected output

PyTorch model checkpoint files in outputs directory that can be loaded as well as logs.

An example for loading the model & data is provided in `load_model.py`
