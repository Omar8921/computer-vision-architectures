import torch
import torch.nn as nn

class VGG11(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, 
                               kernel_size=3, stride=1, padding=1)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, 
                               kernel_size=3, stride=1, padding=1)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, 
                               kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, 
                               kernel_size=3, stride=1, padding=1)

        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, 
                               kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(in_channels=512, out_channels=512, 
                               kernel_size=3, stride=1, padding=1)

        self.conv7 = nn.Conv2d(in_channels=512, out_channels=512, 
                               kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(in_channels=512, out_channels=512, 
                               kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(512 * 7 * 7, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)

        self._initialize_weights()
    
    def forward(self, x):
        # Shape: (B, 3, 224, 224)
        x = self.relu(self.conv1(x))
        x = self.pool(x)

        # Shape: (B, 64, 112, 112)
        x = self.relu(self.conv2(x))
        x = self.pool(x)

        # Shape: (B, 128, 56, 56)
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.pool(x)

        # Shape: (B, 256, 28, 28)
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        x = self.pool(x)

        # Shape: (B, 512, 14, 14)
        x = self.relu(self.conv7(x))
        x = self.relu(self.conv8(x))
        x = self.pool(x)

        # Shape: (B, 512, 7, 7)
        x = torch.flatten(x, start_dim=1)

        # Shape: (B, 512 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)

        # Shape: (B, 4096)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)

        # Shape: (B, 4096)
        x = self.fc3(x)

        # Shape: (B, num_classes)
        return x

    def _initialize_weights(self):
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0)