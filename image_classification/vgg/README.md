# VGG11

PyTorch implementation of **VGG11**, based on *Very Deep Convolutional Networks for Large-Scale Image Recognition* by Simonyan and Zisserman.

## Architecture

| Layer | Configuration |
|---|---|
| Conv1 | 3 → 64, 3×3 |
| MaxPool | 2×2, stride 2 |
| Conv2 | 64 → 128, 3×3 |
| MaxPool | 2×2, stride 2 |
| Conv3 | 128 → 256, 3×3 |
| Conv4 | 256 → 256, 3×3 |
| MaxPool | 2×2, stride 2 |
| Conv5 | 256 → 512, 3×3 |
| Conv6 | 512 → 512, 3×3 |
| MaxPool | 2×2, stride 2 |
| Conv7 | 512 → 512, 3×3 |
| Conv8 | 512 → 512, 3×3 |
| MaxPool | 2×2, stride 2 |
| FC1 | 25088 → 4096 |
| FC2 | 4096 → 4096 |
| FC3 | 4096 → num_classes |

All convolutional layers use stride `1` and padding `1`. ReLU is applied after every convolutional layer and the first two fully connected layers. Dropout (`p=0.5`) is applied after the first two fully connected layers.

## Key Ideas

- Stacking small 3×3 convolutional filters
- Increasing network depth while keeping convolutional filters small
- 2×2 max pooling for spatial downsampling
- Large fully connected classifier
- Dropout for regularization

## Experiment

The implementation is tested on **STL-10**.

Training setup:

- Loss: Cross-Entropy
- Optimizer: AdamW
- Learning rate: `0.0001`
- Weight decay: `1e-4`
- Training epochs: `50`
- Data augmentation: only resize

A small overfitting test is used to verify that the model and training pipeline are capable of learning meaningful patterns with enough computation power and data.

## Files

```text
vgg/
├── model.py          # VGG11 implementation
├── experiment.ipynb  # Model experimenting
├── paper.pdf         # Original VGG paper
└── README.md