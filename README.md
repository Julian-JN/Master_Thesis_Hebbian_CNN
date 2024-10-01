# Master Thesis: Hebbian CNN

This repository contains the code used in the experiments presented in the thesis. Below is a guide to the structure and key elements of the project.

## Framework and Dataset

- The code is built using **PyTorch**, a popular deep learning framework.
- The experiments were conducted using the **CIFAR-10** dataset.
- Only additional folders which will be required to be ctreated by the user are a results folder, with a demo folder inside it to store results if required, and a datasets folder with a folder called cifar10, where the Cifar-10 dataset will be automatically downloaded
## Running Experiments

To run the experiments, execute the Python files prefixed with `experiments_`. The suffix of each file indicates the architecture used:
- **Standard Convolutions**
- **Depthwise Separable Convolutions**
- **Residual Blocks**
- **Backpropagation (instead of Hebbian learning)**

### Parameter Configuration

In these files, the following Hebbian learning parameters can be found:

```python
hebb_param = {
    'mode': 'hard', 
    'w_nrm': False, 
    'act': nn.Identity(), 
    'k': 1, 
    'alpha': 1.
}
```

- `mode`: Defines the competition mode for CNN layers.
  - `'hard'`: Hard-WTA (Winner Takes All)
  - `'soft'`: Soft-Hebb learning
  - `'basic'`: no competition mechanisms
  - `'temp'`: Temporal thresholds and hard-WTA
  - `'thresh'`: Statistical thresholding
  - `'BCM'`: BCM learning rule
- `w_nrm`: Controls whether weight normalization is applied.
- `act`: Activation function used.
- `k`: Number of winners.
- `alpha`: Learning rate for Hebbian updates.

### Surround Inhibition and Cosine Similarity

To enable/disable **surround inhibition** or **cosine similarity activation**, you must manually adjust the code in the `hebb.py` file.

## Dale's Principle Layers

Files with the `_abs` suffix indicate layers that respect **Dale's Principle**. If running experiments using these layers, ensure that necessary changes are made in the `receptive_field.py` and `mode_.py` files to import the correct modules.

## Model Architecture

The architecture and code for visualizing the filters of the models are located in the `model_*.py` files. The suffix in these filenames indicates the architecture:
- **Standard Convolutions**
- **Depthwise Separable Convolutions**
- **Residual Blocks**
- **Backpropagation Layers (instead of Hebbian layers)**

## Visualization and Plotting

The **plots** that help to understand Hebbian learning and its dynamics are stored using **Weights & Biases (WanDB)** and declared in the `visualizer.py` and `receptive_field.py` files.

### Types of Plots:
- **Filters/Neurons/Statistics/Distributions**: Created using the **Matplotlib** library.
- **Class Clustering Plots**: Generated using the **UMAP** library.

## Future Work

Future improvements include the ideas discussed in the thesis, along with:
- Creating a `requirements.txt` file.
- Adding an accompanying script to set up a virtual environment for faster integration by researchers.
