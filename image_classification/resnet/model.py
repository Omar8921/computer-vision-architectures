import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        kernel_size: int, 
        stride: int,
        padding: int
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                      kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),

            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, 
                      kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(num_features=out_channels),
        )

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=stride
                ),
                nn.BatchNorm2d(num_features=out_channels)
            )
        else:
            self.shortcut = nn.Identity()

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.block(x)
        shortcut = self.shortcut(x)

        out = out + shortcut
        out = self.relu(out)

        return out


class ResNet34(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64,
                      kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ResidualBlock(in_channels=64, out_channels=64,
                          kernel_size=3, stride=1, padding=1),
            ResidualBlock(in_channels=64, out_channels=64,
                          kernel_size=3, stride=1, padding=1),
            ResidualBlock(in_channels=64, out_channels=64,
                          kernel_size=3, stride=1, padding=1)
        )

        self.conv3 = nn.Sequential(
            ResidualBlock(in_channels=64, out_channels=128,
                          kernel_size=3, stride=2, padding=1),
            ResidualBlock(in_channels=128, out_channels=128,
                          kernel_size=3, stride=1, padding=1),
            ResidualBlock(in_channels=128, out_channels=128,
                          kernel_size=3, stride=1, padding=1),
            ResidualBlock(in_channels=128, out_channels=128,
                          kernel_size=3, stride=1, padding=1)
        )

        self.conv4 = nn.Sequential(
            ResidualBlock(in_channels=128, out_channels=256,
                          kernel_size=3, stride=2, padding=1),
            ResidualBlock(in_channels=256, out_channels=256,
                          kernel_size=3, stride=1, padding=1),
            ResidualBlock(in_channels=256, out_channels=256,
                          kernel_size=3, stride=1, padding=1),
            ResidualBlock(in_channels=256, out_channels=256,
                          kernel_size=3, stride=1, padding=1),
            ResidualBlock(in_channels=256, out_channels=256,
                          kernel_size=3, stride=1, padding=1),
            ResidualBlock(in_channels=256, out_channels=256,
                          kernel_size=3, stride=1, padding=1)
        )

        self.conv5 = nn.Sequential(
            ResidualBlock(in_channels=256, out_channels=512,
                          kernel_size=3, stride=2, padding=1),
            ResidualBlock(in_channels=512, out_channels=512,
                          kernel_size=3, stride=1, padding=1),
            ResidualBlock(in_channels=512, out_channels=512,
                          kernel_size=3, stride=1, padding=1)
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 1 * 1, num_classes)

        self._initialize_weights()

    def forward(self, x):
        # Shape: (B, 3, 224, 224)
        x = self.conv1(x)

        # Shape: (B, 64, 112, 112)
        x = self.conv2(x)

        # Shape: (B, 64, 56, 56)
        x = self.conv3(x)

        # Shape: (B, 128, 28, 28)
        x = self.conv4(x)

        # Shape: (B, 256, 14, 14)
        x = self.conv5(x)

        # Shape: (B, 512, 7, 7)
        x = self.avg_pool(x)

        # Shape: (B, 512, 1, 1)
        x = torch.flatten(x, start_dim=1)

        # Shape: (B, 512 * 1 * 1)
        x = self.fc(x)

        return x

    def _initialize_weights(self):
        for layer in self.modules():

            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(
                    layer.weight,
                    mode="fan_out",
                    nonlinearity="relu"
                )

                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)

            elif isinstance(layer, nn.Linear):
                nn.init.normal_(
                    layer.weight,
                    mean=0.0,
                    std=0.01
                )

                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)