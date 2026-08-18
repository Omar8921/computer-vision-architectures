# Computer Vision Architectures

A personal repository for studying important computer vision architectures through **notes, implementations, and small experiments**.

The goal is to understand the key ideas behind major architectures and have a reference I can easily revisit later.

## Structure

```text
computer-vision-architectures/
├── image_classification/
│   ├── alexnet/
│   ├── vgg/
│   └── resnet/
├── object_detection/
│   ├── faster_rcnn/
│   ├── yolo/
│   └── detr/
├── segmentation/
│   ├── unet/
│   └── mask_rcnn/
└── vision_transformers/
    └── vit/
```

Each architecture may contain:

```text
architecture/
├── README.md        # Architecture notes
├── model.py         # Implementation
└── experiment.ipynb # Experiments / visualization
└── paper.pdf        # Published research paper
```

## Study Approach

For each architecture:

1. Understand the problem it solves.
2. Read the original paper.
3. Document the architecture and its main idea.
4. Implement the important components.
5. Run a small experiment.
6. Be able to explain how and why it works.

The focus is on **understanding important architectural ideas rather than learning every model that exists**.
