import torch
import torch.nn as nn

class FCN(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super(FCN, self).__init__()

        # Encoder blocks
        self.encoder_1 = self.encoder_block_2(in_channels, 64)  
        self.encoder_2 = self.encoder_block_2(64, 128)          
        self.encoder_3 = self.encoder_block_3(128, 256)         
        self.encoder_4 = self.encoder_block_3(256, 512)        

        # Mid block
        self.mid = self.mid_block(512, 1024)                    

        # Upsampling layers
        self.conv_t_32s = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.conv_t_16s = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.conv_t_8s  = nn.ConvTranspose2d(768, 256, 4, 4)

        # 1x1 convolutions for skip connections
        self.x3_conv_1x1 = nn.Conv2d(256, 512, 1)
        self.x2_conv_1x1 = nn.Conv2d(128, 256, 1)

        # Output layer
        self.output = nn.Conv2d(256, out_channels, 1)
        # self.activation = nn.Sigmoid()

    def encoder_block_2(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 'same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 'same'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

    def encoder_block_3(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 'same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 'same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 'same'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

    def mid_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 'same'),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),
            nn.Conv2d(out_channels, out_channels, 3, 1, 'same'),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),
            nn.Conv2d(out_channels, 1024, 3, 1, 'same'),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, output_stride=None):
        # Encoder
        x1 = self.encoder_1(x)
        x2 = self.encoder_2(x1)
        x3 = self.encoder_3(x2)
        x4 = self.encoder_4(x3)

        # 1x1 convolutions
        x3_1x1 = self.x3_conv_1x1(x3)
        x2_1x1 = self.x2_conv_1x1(x2)

        # Mid-block
        x_mid = self.mid(x4)

        # FCN-32s path
        x4 = self.conv_t_32s(x_mid)
        if output_stride == '32s':
            return self.output(x4)

        # FCN-16s path
        x5 = torch.cat([x4, x3_1x1], dim=1)
        x5 = self.conv_t_16s(x5)
        if output_stride == '16s':
            return self.output(x5)

        # FCN-8s path
        x6 = torch.cat([x5, x2_1x1], dim=1)
        x6 = self.conv_t_8s(x6)

        # Final output layer
        return self.output(x6)
