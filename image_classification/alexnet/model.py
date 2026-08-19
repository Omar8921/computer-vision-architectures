import torch
import torch.nn as nn
import torch.nn.functional as F

class LocalResponseNormalization(nn.Module):
    def __init__(self, k: int, n: int, alpha: float, beta: float):
        super().__init__()
        self.k = k
        self.n = n
        self.alpha = alpha
        self.beta = beta

    def forward(self, x: torch.Tensor):
        # Input Shape: (B, C, H, W)
        b, c, h, w = x.shape

        squared = x.square()

        # (B, C, H, W)
        #      ↓
        # (B, H, W, C)
        #      ↓
        # (B*H*W, 1, C)
        squared = squared.permute(0, 2, 3, 1).reshape(-1, 1, c)

        kernel = torch.ones(
            1, 1, self.n,
            device=x.device,
            dtype=x.dtype
        )

        # Back to (B, C, H, W)
        local_sum = F.conv1d(
            squared, 
            kernel, 
            padding=self.n // 2
        )

        local_sum = local_sum.reshape(b, h, w, c).permute(0, 3, 1, 2)

        denominator = (self.k + self.alpha * local_sum).pow(self.beta)

        return x / denominator

class AlexNet(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        self.overlapping_pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.relu = nn.ReLU()
        self.lrn = LocalResponseNormalization(k=2, n=5, alpha=1e-4, beta=0.75)
        self.dropout = nn.Dropout(p=0.5)

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2, groups=2)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1, groups=2)
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1, groups=2)

        self.fc1 = nn.Linear(6 * 6 * 256, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)

        self._initialize_weights()

    def forward(self, x: torch.Tensor):
        # Input Shape: (B, 3, 224, 224)
        x = self.relu(self.conv1(x))
        x = self.lrn(x)
        x = self.overlapping_pool(x)

        # Input Shape: (B, 96, 27, 27)
        x = self.relu(self.conv2(x))
        x = self.lrn(x)
        x = self.overlapping_pool(x)

        # Input Shape: (B, 256, 13, 13)
        x = self.relu(self.conv3(x))

        # Input Shape: (B, 384, 13, 13)
        x = self.relu(self.conv4(x))

        # Input Shape: (B, 384, 13, 13)
        x = self.relu(self.conv5(x))
        x = self.overlapping_pool(x)

        # Input Shape: (B, 256, 6, 6)
        x = x.view(x.shape[0], -1)
        x = self.dropout(self.relu(self.fc1(x)))        
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)

        return x

    def _initialize_weights(self):
        # All weights: N(0, 0.01²)
        for layer in [
            self.conv1, self.conv2, self.conv3,
            self.conv4, self.conv5,
            self.fc1, self.fc2, self.fc3
        ]:
            nn.init.normal_(layer.weight, mean=0.0, std=0.01)

        # Bias = 0
        nn.init.constant_(self.conv1.bias, 0)
        nn.init.constant_(self.conv3.bias, 0)
        nn.init.constant_(self.fc3.bias, 0)

        # Bias = 1
        nn.init.constant_(self.conv2.bias, 1)
        nn.init.constant_(self.conv4.bias, 1)
        nn.init.constant_(self.conv5.bias, 1)
        nn.init.constant_(self.fc1.bias, 1)
        nn.init.constant_(self.fc2.bias, 1)