import torch
import torch.nn as nn


class WiAR_CNN(nn.Module):
    def __init__(self, shrinkage_ratio):
        super(WiAR_CNN, self).__init__()
        self.shrinkage_ratio = shrinkage_ratio

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=int(16 * self.shrinkage_ratio),
                kernel_size=7, stride=6, padding=1
            ),
            nn.BatchNorm2d(int(16 * self.shrinkage_ratio)),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=int(16 * self.shrinkage_ratio),
                out_channels=int(32 * self.shrinkage_ratio),
                kernel_size=7, stride=(2, 6), padding=1
            ),
            nn.BatchNorm2d(int(32 * self.shrinkage_ratio)),
        )

        self.linear = nn.Linear(int(32 * self.shrinkage_ratio) * 6 * 7, 15)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, 1, 90, 250)

        conv_output = torch.relu(self.conv1(x))
        conv_output = torch.relu(self.conv2(conv_output))

        conv_output = conv_output.view(batch_size, int(32 * self.shrinkage_ratio) * 6 * 7)
        output = self.linear(conv_output)

        return output
