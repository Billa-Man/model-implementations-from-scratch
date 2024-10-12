import torch
import torch.nn as nn

class VGG19(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(VGG19, self).__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes

        self.conv_layers = nn.Sequential(
            self.conv_block(in_channels, 64, 2),
            self.conv_block(64, 128, 2),
            self.conv_block(128, 256, 4),
            self.conv_block(256, 512, 4),
            self.conv_block(512, 512, 4),
        )

        self.fcnn = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(),
            nn.Dropout(p = 0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p = 0.5),
            nn.Linear(4096, num_classes)
        )

        self.flatten = nn.Flatten()


    def conv_block(self, in_channels, out_channels, num_convs):

        layers = []

        for _ in range(num_convs):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            in_channels = out_channels
            
        layers.append(nn.MaxPool2d(kernel_size = 2, stride = 2))

        return nn.Sequential(*layers)


    def forward(self, x):

        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.fcnn(x)

        return x
