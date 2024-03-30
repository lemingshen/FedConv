import torch
import torch.nn as nn
from torch.nn import functional as F


class ResNet18(nn.Module):
    def __init__(self, shrinkage_ratio):
        super(ResNet18, self).__init__()

        self.shrinkage_ratio = shrinkage_ratio
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=int(64 * self.shrinkage_ratio),
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(int(64 * self.shrinkage_ratio)),
        )

        # block1
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=int(64 * self.shrinkage_ratio),
                out_channels=int(64 * self.shrinkage_ratio),
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(int(64 * self.shrinkage_ratio)),
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=int(64 * self.shrinkage_ratio),
                out_channels=int(64 * self.shrinkage_ratio),
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(int(64 * self.shrinkage_ratio)),
        )

        # block2
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=int(64 * self.shrinkage_ratio),
                out_channels=int(64 * self.shrinkage_ratio),
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(int(64 * self.shrinkage_ratio)),
        )
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=int(64 * self.shrinkage_ratio),
                out_channels=int(64 * self.shrinkage_ratio),
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(int(64 * self.shrinkage_ratio)),
        )

        # block3
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=int(64 * self.shrinkage_ratio),
                out_channels=int(128 * self.shrinkage_ratio),
                kernel_size=1,
                stride=2,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(int(128 * self.shrinkage_ratio)),
        )
        self.conv3_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=int(64 * self.shrinkage_ratio),
                out_channels=int(128 * self.shrinkage_ratio),
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(int(128 * self.shrinkage_ratio)),
        )
        self.conv3_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=int(128 * self.shrinkage_ratio),
                out_channels=int(128 * self.shrinkage_ratio),
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(int(128 * self.shrinkage_ratio)),
        )

        # block4
        self.conv4_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=int(128 * self.shrinkage_ratio),
                out_channels=int(128 * self.shrinkage_ratio),
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(int(128 * self.shrinkage_ratio)),
        )
        self.conv4_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=int(128 * self.shrinkage_ratio),
                out_channels=int(128 * self.shrinkage_ratio),
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(int(128 * self.shrinkage_ratio)),
        )

        # block5
        self.conv5 = nn.Sequential(
            nn.Conv2d(
                in_channels=int(128 * self.shrinkage_ratio),
                out_channels=int(256 * self.shrinkage_ratio),
                kernel_size=1,
                stride=2,
                padding=0,
            ),
            nn.BatchNorm2d(int(256 * self.shrinkage_ratio)),
        )
        self.conv5_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=int(128 * self.shrinkage_ratio),
                out_channels=int(256 * self.shrinkage_ratio),
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(int(256 * self.shrinkage_ratio)),
        )
        self.conv5_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=int(256 * self.shrinkage_ratio),
                out_channels=int(256 * self.shrinkage_ratio),
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(int(256 * self.shrinkage_ratio)),
        )

        # block6
        self.conv6_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=int(256 * self.shrinkage_ratio),
                out_channels=int(256 * self.shrinkage_ratio),
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(int(256 * self.shrinkage_ratio)),
        )
        self.conv6_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=int(256 * self.shrinkage_ratio),
                out_channels=int(256 * self.shrinkage_ratio),
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(int(256 * self.shrinkage_ratio)),
        )

        # block7
        self.conv7 = nn.Sequential(
            nn.Conv2d(
                in_channels=int(256 * self.shrinkage_ratio),
                out_channels=int(512 * self.shrinkage_ratio),
                kernel_size=1,
                stride=2,
                padding=0,
            ),
            nn.BatchNorm2d(int(512 * self.shrinkage_ratio)),
        )
        self.conv7_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=int(256 * self.shrinkage_ratio),
                out_channels=int(512 * self.shrinkage_ratio),
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(int(512 * self.shrinkage_ratio)),
        )
        self.conv7_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=int(512 * self.shrinkage_ratio),
                out_channels=int(512 * self.shrinkage_ratio),
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(int(512 * self.shrinkage_ratio)),
        )

        # block8
        self.conv8_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=int(512 * self.shrinkage_ratio),
                out_channels=int(512 * self.shrinkage_ratio),
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(int(512 * self.shrinkage_ratio)),
        )
        self.conv8_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=int(512 * self.shrinkage_ratio),
                out_channels=int(512 * self.shrinkage_ratio),
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(int(512 * self.shrinkage_ratio)),
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.linear = nn.Linear(int(512 * self.shrinkage_ratio), 10)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # the original shape of x is (batch_size, 3, 32, 32)
        self.batch_size = x.size(0)
        x = x.view(self.batch_size, 3, 32, 32)

        feature_map = self.conv1(x)
        # feature_map.register_hook(extract)
        output = torch.max_pool2d(feature_map, 2)

        # block 1
        identity = output
        output = torch.relu(self.conv1_1(output))
        output = self.conv1_2(output)
        output = output + identity
        output = torch.relu(output)

        # block 2
        identity = output
        output = torch.relu(self.conv2_1(output))
        output = self.conv2_2(output)
        output = output + identity
        output = torch.relu(output)

        # block 3
        identity = self.conv3(output)
        output = torch.relu(self.conv3_1(output))
        output = self.conv3_2(output)
        output = output + identity
        output = torch.relu(output)

        # block4
        identity = output
        output = torch.relu(self.conv4_1(output))
        output = self.conv4_2(output)
        output = output + identity
        output = torch.relu(output)

        # block 5
        identity = self.conv5(output)
        output = torch.relu(self.conv5_1(output))
        output = self.conv5_2(output)
        output = output + identity
        output = torch.relu(output)

        # block 6
        identity = output
        output = torch.relu(self.conv6_1(output))
        output = self.conv6_2(output)
        output = output + identity
        output = torch.relu(output)

        # block7
        identity = self.conv7(output)
        output = torch.relu(self.conv7_1(output))
        output = self.conv7_2(output)
        output = output + identity
        output = torch.relu(output)

        # block 8
        identity = output
        output = torch.relu(self.conv8_1(output))
        output = self.conv8_2(output)
        output = output + identity
        output = torch.relu(output)

        output = self.avg_pool(output)
        output = torch.flatten(output, 1)
        output = self.linear(output)

        return output
