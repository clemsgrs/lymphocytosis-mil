import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self, num_input_channels=3, num_classes=1):
        super(SimpleCNN, self).__init__()
        self.layer1 = nn.Conv2d(num_input_channels, 32, kernel_size=3)
        self.pool1 = nn.AvgPool2d(2)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool2 = nn.AvgPool2d(2)
        self.bn2 = nn.BatchNorm2d(64)
        self.layer3 = nn.Conv2d(64, 128, kernel_size=3)
        self.pool3 = nn.AvgPool2d(2)
        self.bn3 = nn.BatchNorm2d(128)

        self.final_pool = nn.AdaptiveAvgPool2d(1)
        self.final_layer = nn.Conv2d(128, num_classes, kernel_size=1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.pool1(out)
        out = nn.functional.relu(out)
        out = self.bn1(out)

        out = self.layer2(out)
        out = self.pool2(out)
        out = nn.functional.relu(out)
        out = self.bn2(out)

        out = self.layer3(out)
        out = self.pool3(out)
        out = nn.functional.relu(out)
        out = self.bn3(out)

        out = self.final_pool(out)
        out = self.final_layer(out)
        return out.view(x.size(0), -1)