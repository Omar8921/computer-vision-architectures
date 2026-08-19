# AlexNet

PyTorch implementation of **AlexNet**, based on *ImageNet Classification with Deep Convolutional Neural Networks* by Krizhevsky, Sutskever, and Hinton.

## Architecture

| Layer | Configuration |
|---|---|
| Conv1 | 3 → 96, 11×11, stride 4 |
| MaxPool | 3×3, stride 2 |
| Conv2 | 96 → 256, 5×5, groups=2 |
| MaxPool | 3×3, stride 2 |
| Conv3 | 256 → 384, 3×3 |
| Conv4 | 384 → 384, 3×3, groups=2 |
| Conv5 | 384 → 256, 3×3, groups=2 |
| MaxPool | 3×3, stride 2 |
| FC1 | 9216 → 4096 |
| FC2 | 4096 → 4096 |
| FC3 | 4096 → num_classes |

ReLU is used throughout the network. Local Response Normalization (LRN) follows Conv1 and Conv2, and dropout (`p=0.5`) is applied after the first two fully connected layers.

## Key Ideas

- ReLU activations
- Local Response Normalization
- Overlapping max pooling
- Grouped convolutions to reproduce the original two-GPU connectivity
- Dropout for regularization
- Original AlexNet weight and bias initialization

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
alexnet/
├── model.py          # AlexNet and custom LRN implementation
├── experiment.ipynb  # Model experimenting
├── paper.pdf         # Original AlexNet paper
└── README.md