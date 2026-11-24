from torch import nn


class BaselineCNN(nn.Module):
    """Three convolutional stages for 32 x 32 RGB images."""

    def __init__(self):
        super().__init__()
        layers = []
        in_channels = 3
        for width in (32, 64, 128):
            for _ in range(2):
                layers.extend([
                    nn.Conv2d(in_channels, width, 3, padding=1, bias=False),
                    nn.BatchNorm2d(width),
                    nn.ReLU(inplace=True),
                ])
                in_channels = width
            layers.append(nn.MaxPool2d(2))
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, 10)
        )

    def forward(self, images):
        return self.classifier(self.features(images))
