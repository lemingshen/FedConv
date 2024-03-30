import torch
import torch.nn as nn


class GoogLeNet(nn.Module):
    def __init__(self, shrinkage_ratio):
        super(GoogLeNet, self).__init__()

        self.shrinkage_ratio = shrinkage_ratio

        self.conv = nn.Sequential(
            nn.Conv2d(
                3, int(192 * self.shrinkage_ratio), kernel_size=3, padding=1, stride=1
            ),
            nn.BatchNorm2d(int(192 * self.shrinkage_ratio)),
            nn.ReLU(inplace=True),
        )

        self.conv1_1 = nn.Sequential(
            nn.Conv2d(
                int(192 * self.shrinkage_ratio),
                int(64 * self.shrinkage_ratio),
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(int(64 * self.shrinkage_ratio)),
            nn.ReLU(inplace=True),
        )
        self.conv1_21 = nn.Sequential(
            nn.Conv2d(
                int(192 * self.shrinkage_ratio),
                int(96 * self.shrinkage_ratio),
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(int(96 * self.shrinkage_ratio)),
            nn.ReLU(inplace=True),
        )
        self.conv1_22 = nn.Sequential(
            nn.Conv2d(
                int(96 * self.shrinkage_ratio),
                int(128 * self.shrinkage_ratio),
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(int(128 * self.shrinkage_ratio)),
            nn.ReLU(inplace=True),
        )
        self.conv1_31 = nn.Sequential(
            nn.Conv2d(
                int(192 * self.shrinkage_ratio),
                int(16 * self.shrinkage_ratio),
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(int(16 * self.shrinkage_ratio)),
            nn.ReLU(inplace=True),
        )
        self.conv1_32 = nn.Sequential(
            nn.Conv2d(
                int(16 * self.shrinkage_ratio),
                int(32 * self.shrinkage_ratio),
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(int(32 * self.shrinkage_ratio)),
            nn.ReLU(inplace=True),
        )
        self.conv1_33 = nn.Sequential(
            nn.Conv2d(
                int(32 * self.shrinkage_ratio),
                int(32 * self.shrinkage_ratio),
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(int(32 * self.shrinkage_ratio)),
            nn.ReLU(inplace=True),
        )
        self.conv1_4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(
                int(192 * self.shrinkage_ratio),
                int(32 * self.shrinkage_ratio),
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(int(32 * self.shrinkage_ratio)),
            nn.ReLU(inplace=True),
        )

        self.conv2_1 = nn.Sequential(
            nn.Conv2d(
                int(256 * self.shrinkage_ratio),
                int(128 * self.shrinkage_ratio),
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(int(128 * self.shrinkage_ratio)),
            nn.ReLU(inplace=True),
        )
        self.conv2_21 = nn.Sequential(
            nn.Conv2d(
                int(256 * self.shrinkage_ratio),
                int(128 * self.shrinkage_ratio),
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(int(128 * self.shrinkage_ratio)),
            nn.ReLU(inplace=True),
        )
        self.conv2_22 = nn.Sequential(
            nn.Conv2d(
                int(128 * self.shrinkage_ratio),
                int(192 * self.shrinkage_ratio),
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(int(192 * self.shrinkage_ratio)),
            nn.ReLU(inplace=True),
        )
        self.conv2_31 = nn.Sequential(
            nn.Conv2d(
                int(256 * self.shrinkage_ratio),
                int(32 * self.shrinkage_ratio),
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(int(32 * self.shrinkage_ratio)),
            nn.ReLU(inplace=True),
        )
        self.conv2_32 = nn.Sequential(
            nn.Conv2d(
                int(32 * self.shrinkage_ratio),
                int(96 * self.shrinkage_ratio),
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(int(96 * self.shrinkage_ratio)),
            nn.ReLU(inplace=True),
        )
        self.conv2_33 = nn.Sequential(
            nn.Conv2d(
                int(96 * self.shrinkage_ratio),
                int(96 * self.shrinkage_ratio),
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(int(96 * self.shrinkage_ratio)),
            nn.ReLU(inplace=True),
        )
        self.conv2_4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(
                int(256 * self.shrinkage_ratio),
                int(64 * self.shrinkage_ratio),
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(int(64 * self.shrinkage_ratio)),
            nn.ReLU(inplace=True),
        )

        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv3_1 = nn.Sequential(
            nn.Conv2d(
                int(480 * self.shrinkage_ratio),
                int(192 * self.shrinkage_ratio),
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(int(192 * self.shrinkage_ratio)),
            nn.ReLU(inplace=True),
        )
        self.conv3_21 = nn.Sequential(
            nn.Conv2d(
                int(480 * self.shrinkage_ratio),
                int(96 * self.shrinkage_ratio),
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(int(96 * self.shrinkage_ratio)),
            nn.ReLU(inplace=True),
        )
        self.conv3_22 = nn.Sequential(
            nn.Conv2d(
                int(96 * self.shrinkage_ratio),
                int(208 * self.shrinkage_ratio),
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(int(208 * self.shrinkage_ratio)),
            nn.ReLU(inplace=True),
        )
        self.conv3_31 = nn.Sequential(
            nn.Conv2d(
                int(480 * self.shrinkage_ratio),
                int(16 * self.shrinkage_ratio),
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(int(16 * self.shrinkage_ratio)),
            nn.ReLU(inplace=True),
        )
        self.conv3_32 = nn.Sequential(
            nn.Conv2d(
                int(16 * self.shrinkage_ratio),
                int(48 * self.shrinkage_ratio),
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(int(48 * self.shrinkage_ratio)),
            nn.ReLU(inplace=True),
        )
        self.conv3_33 = nn.Sequential(
            nn.Conv2d(
                int(48 * self.shrinkage_ratio),
                int(48 * self.shrinkage_ratio),
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(int(48 * self.shrinkage_ratio)),
            nn.ReLU(inplace=True),
        )
        self.conv3_4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(
                int(480 * self.shrinkage_ratio),
                int(64 * self.shrinkage_ratio),
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(int(64 * self.shrinkage_ratio)),
            nn.ReLU(inplace=True),
        )

        self.conv4_1 = nn.Sequential(
            nn.Conv2d(
                int(512 * self.shrinkage_ratio),
                int(160 * self.shrinkage_ratio),
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(int(160 * self.shrinkage_ratio)),
            nn.ReLU(inplace=True),
        )
        self.conv4_21 = nn.Sequential(
            nn.Conv2d(
                int(512 * self.shrinkage_ratio),
                int(112 * self.shrinkage_ratio),
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(int(112 * self.shrinkage_ratio)),
            nn.ReLU(inplace=True),
        )
        self.conv4_22 = nn.Sequential(
            nn.Conv2d(
                int(112 * self.shrinkage_ratio),
                int(224 * self.shrinkage_ratio),
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(int(224 * self.shrinkage_ratio)),
            nn.ReLU(inplace=True),
        )
        self.conv4_31 = nn.Sequential(
            nn.Conv2d(
                int(512 * self.shrinkage_ratio),
                int(24 * self.shrinkage_ratio),
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(int(24 * self.shrinkage_ratio)),
            nn.ReLU(inplace=True),
        )
        self.conv4_32 = nn.Sequential(
            nn.Conv2d(
                int(24 * self.shrinkage_ratio),
                int(64 * self.shrinkage_ratio),
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(int(64 * self.shrinkage_ratio)),
            nn.ReLU(inplace=True),
        )
        self.conv4_33 = nn.Sequential(
            nn.Conv2d(
                int(64 * self.shrinkage_ratio),
                int(64 * self.shrinkage_ratio),
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(int(64 * self.shrinkage_ratio)),
            nn.ReLU(inplace=True),
        )
        self.conv4_4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(
                int(512 * self.shrinkage_ratio),
                int(64 * self.shrinkage_ratio),
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(int(64 * self.shrinkage_ratio)),
            nn.ReLU(inplace=True),
        )

        self.conv5_1 = nn.Sequential(
            nn.Conv2d(
                int(512 * self.shrinkage_ratio),
                int(128 * self.shrinkage_ratio),
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(int(128 * self.shrinkage_ratio)),
            nn.ReLU(inplace=True),
        )
        self.conv5_21 = nn.Sequential(
            nn.Conv2d(
                int(512 * self.shrinkage_ratio),
                int(128 * self.shrinkage_ratio),
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(int(128 * self.shrinkage_ratio)),
            nn.ReLU(inplace=True),
        )
        self.conv5_22 = nn.Sequential(
            nn.Conv2d(
                int(128 * self.shrinkage_ratio),
                int(256 * self.shrinkage_ratio),
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(int(256 * self.shrinkage_ratio)),
            nn.ReLU(inplace=True),
        )
        self.conv5_31 = nn.Sequential(
            nn.Conv2d(
                int(512 * self.shrinkage_ratio),
                int(24 * self.shrinkage_ratio),
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(int(24 * self.shrinkage_ratio)),
            nn.ReLU(inplace=True),
        )
        self.conv5_32 = nn.Sequential(
            nn.Conv2d(
                int(24 * self.shrinkage_ratio),
                int(64 * self.shrinkage_ratio),
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(int(64 * self.shrinkage_ratio)),
            nn.ReLU(inplace=True),
        )
        self.conv5_33 = nn.Sequential(
            nn.Conv2d(
                int(64 * self.shrinkage_ratio),
                int(64 * self.shrinkage_ratio),
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(int(64 * self.shrinkage_ratio)),
            nn.ReLU(inplace=True),
        )
        self.conv5_4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(
                int(512 * self.shrinkage_ratio),
                int(64 * self.shrinkage_ratio),
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(int(64 * self.shrinkage_ratio)),
            nn.ReLU(inplace=True),
        )

        self.conv6_1 = nn.Sequential(
            nn.Conv2d(
                int(512 * self.shrinkage_ratio),
                int(112 * self.shrinkage_ratio),
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(int(112 * self.shrinkage_ratio)),
            nn.ReLU(inplace=True),
        )
        self.conv6_21 = nn.Sequential(
            nn.Conv2d(
                int(512 * self.shrinkage_ratio),
                int(144 * self.shrinkage_ratio),
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(int(144 * self.shrinkage_ratio)),
            nn.ReLU(inplace=True),
        )
        self.conv6_22 = nn.Sequential(
            nn.Conv2d(
                int(144 * self.shrinkage_ratio),
                int(288 * self.shrinkage_ratio),
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(int(288 * self.shrinkage_ratio)),
            nn.ReLU(inplace=True),
        )
        self.conv6_31 = nn.Sequential(
            nn.Conv2d(
                int(512 * self.shrinkage_ratio),
                int(32 * self.shrinkage_ratio),
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(int(32 * self.shrinkage_ratio)),
            nn.ReLU(inplace=True),
        )
        self.conv6_32 = nn.Sequential(
            nn.Conv2d(
                int(32 * self.shrinkage_ratio),
                int(64 * self.shrinkage_ratio),
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(int(64 * self.shrinkage_ratio)),
            nn.ReLU(inplace=True),
        )
        self.conv6_33 = nn.Sequential(
            nn.Conv2d(
                int(64 * self.shrinkage_ratio),
                int(64 * self.shrinkage_ratio),
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(int(64 * self.shrinkage_ratio)),
            nn.ReLU(inplace=True),
        )
        self.conv6_4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(
                int(512 * self.shrinkage_ratio),
                int(64 * self.shrinkage_ratio),
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(int(64 * self.shrinkage_ratio)),
            nn.ReLU(inplace=True),
        )

        self.conv7_1 = nn.Sequential(
            nn.Conv2d(
                int(528 * self.shrinkage_ratio),
                int(256 * self.shrinkage_ratio),
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(int(256 * self.shrinkage_ratio)),
            nn.ReLU(inplace=True),
        )
        self.conv7_21 = nn.Sequential(
            nn.Conv2d(
                int(528 * self.shrinkage_ratio),
                int(160 * self.shrinkage_ratio),
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(int(160 * self.shrinkage_ratio)),
            nn.ReLU(inplace=True),
        )
        self.conv7_22 = nn.Sequential(
            nn.Conv2d(
                int(160 * self.shrinkage_ratio),
                int(320 * self.shrinkage_ratio),
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(int(320 * self.shrinkage_ratio)),
            nn.ReLU(inplace=True),
        )
        self.conv7_31 = nn.Sequential(
            nn.Conv2d(
                int(528 * self.shrinkage_ratio),
                int(32 * self.shrinkage_ratio),
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(int(32 * self.shrinkage_ratio)),
            nn.ReLU(inplace=True),
        )
        self.conv7_32 = nn.Sequential(
            nn.Conv2d(
                int(32 * self.shrinkage_ratio),
                int(128 * self.shrinkage_ratio),
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(int(128 * self.shrinkage_ratio)),
            nn.ReLU(inplace=True),
        )
        self.conv7_33 = nn.Sequential(
            nn.Conv2d(
                int(128 * self.shrinkage_ratio),
                int(128 * self.shrinkage_ratio),
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(int(128 * self.shrinkage_ratio)),
            nn.ReLU(inplace=True),
        )
        self.conv7_4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(
                int(528 * self.shrinkage_ratio),
                int(128 * self.shrinkage_ratio),
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(int(128 * self.shrinkage_ratio)),
            nn.ReLU(inplace=True),
        )

        self.conv8_1 = nn.Sequential(
            nn.Conv2d(
                int(832 * self.shrinkage_ratio),
                int(256 * self.shrinkage_ratio),
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(int(256 * self.shrinkage_ratio)),
            nn.ReLU(inplace=True),
        )
        self.conv8_21 = nn.Sequential(
            nn.Conv2d(
                int(832 * self.shrinkage_ratio),
                int(160 * self.shrinkage_ratio),
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(int(160 * self.shrinkage_ratio)),
            nn.ReLU(inplace=True),
        )
        self.conv8_22 = nn.Sequential(
            nn.Conv2d(
                int(160 * self.shrinkage_ratio),
                int(320 * self.shrinkage_ratio),
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(int(320 * self.shrinkage_ratio)),
            nn.ReLU(inplace=True),
        )
        self.conv8_31 = nn.Sequential(
            nn.Conv2d(
                int(832 * self.shrinkage_ratio),
                int(32 * self.shrinkage_ratio),
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(int(32 * self.shrinkage_ratio)),
            nn.ReLU(inplace=True),
        )
        self.conv8_32 = nn.Sequential(
            nn.Conv2d(
                int(32 * self.shrinkage_ratio),
                int(128 * self.shrinkage_ratio),
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(int(128 * self.shrinkage_ratio)),
            nn.ReLU(inplace=True),
        )
        self.conv8_33 = nn.Sequential(
            nn.Conv2d(
                int(128 * self.shrinkage_ratio),
                int(128 * self.shrinkage_ratio),
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(int(128 * self.shrinkage_ratio)),
            nn.ReLU(inplace=True),
        )
        self.conv8_4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(
                int(832 * self.shrinkage_ratio),
                int(128 * self.shrinkage_ratio),
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(int(128 * self.shrinkage_ratio)),
            nn.ReLU(inplace=True),
        )

        self.conv9_1 = nn.Sequential(
            nn.Conv2d(
                int(832 * self.shrinkage_ratio),
                int(384 * self.shrinkage_ratio),
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(int(384 * self.shrinkage_ratio)),
            nn.ReLU(inplace=True),
        )
        self.conv9_21 = nn.Sequential(
            nn.Conv2d(
                int(832 * self.shrinkage_ratio),
                int(192 * self.shrinkage_ratio),
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(int(192 * self.shrinkage_ratio)),
            nn.ReLU(inplace=True),
        )
        self.conv9_22 = nn.Sequential(
            nn.Conv2d(
                int(192 * self.shrinkage_ratio),
                int(384 * self.shrinkage_ratio),
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(int(384 * self.shrinkage_ratio)),
            nn.ReLU(inplace=True),
        )
        self.conv9_31 = nn.Sequential(
            nn.Conv2d(
                int(832 * self.shrinkage_ratio),
                int(48 * self.shrinkage_ratio),
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(int(48 * self.shrinkage_ratio)),
            nn.ReLU(inplace=True),
        )
        self.conv9_32 = nn.Sequential(
            nn.Conv2d(
                int(48 * self.shrinkage_ratio),
                int(128 * self.shrinkage_ratio),
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(int(128 * self.shrinkage_ratio)),
            nn.ReLU(inplace=True),
        )
        self.conv9_33 = nn.Sequential(
            nn.Conv2d(
                int(128 * self.shrinkage_ratio),
                int(128 * self.shrinkage_ratio),
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(int(128 * self.shrinkage_ratio)),
            nn.ReLU(inplace=True),
        )
        self.conv9_4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(
                int(832 * self.shrinkage_ratio),
                int(128 * self.shrinkage_ratio),
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(int(128 * self.shrinkage_ratio)),
            nn.ReLU(inplace=True),
        )

        self.avg_pool = nn.AvgPool2d(kernel_size=16, stride=1)
        self.linear = nn.Linear(int(1024 * self.shrinkage_ratio), 10)

    def forward(self, x):
        output = self.conv(x)

        output1 = self.conv1_1(output)
        output2 = self.conv1_21(output)
        output2 = self.conv1_22(output2)
        output3 = self.conv1_31(output)
        output3 = self.conv1_32(output3)
        output3 = self.conv1_33(output3)
        output4 = self.conv1_4(output)
        output = torch.cat([output1, output2, output3, output4], dim=1)

        output1 = self.conv2_1(output)
        output2 = self.conv2_21(output)
        output2 = self.conv2_22(output2)
        output3 = self.conv2_31(output)
        output3 = self.conv2_32(output3)
        output3 = self.conv2_33(output3)
        output4 = self.conv2_4(output)
        output = torch.cat([output1, output2, output3, output4], dim=1)

        output = self.max_pool(output)

        output1 = self.conv3_1(output)
        output2 = self.conv3_21(output)
        output2 = self.conv3_22(output2)
        output3 = self.conv3_31(output)
        output3 = self.conv3_32(output3)
        output3 = self.conv3_33(output3)
        output4 = self.conv3_4(output)
        output = torch.cat([output1, output2, output3, output4], dim=1)

        output1 = self.conv4_1(output)
        output2 = self.conv4_21(output)
        output2 = self.conv4_22(output2)
        output3 = self.conv4_31(output)
        output3 = self.conv4_32(output3)
        output3 = self.conv4_33(output3)
        output4 = self.conv4_4(output)
        output = torch.cat([output1, output2, output3, output4], dim=1)

        output1 = self.conv5_1(output)
        output2 = self.conv5_21(output)
        output2 = self.conv5_22(output2)
        output3 = self.conv5_31(output)
        output3 = self.conv5_32(output3)
        output3 = self.conv5_33(output3)
        output4 = self.conv5_4(output)
        output = torch.cat([output1, output2, output3, output4], dim=1)

        output1 = self.conv6_1(output)
        output2 = self.conv6_21(output)
        output2 = self.conv6_22(output2)
        output3 = self.conv6_31(output)
        output3 = self.conv6_32(output3)
        output3 = self.conv6_33(output3)
        output4 = self.conv6_4(output)
        output = torch.cat([output1, output2, output3, output4], dim=1)

        output1 = self.conv7_1(output)
        output2 = self.conv7_21(output)
        output2 = self.conv7_22(output2)
        output3 = self.conv7_31(output)
        output3 = self.conv7_32(output3)
        output3 = self.conv7_33(output3)
        output4 = self.conv7_4(output)
        output = torch.cat([output1, output2, output3, output4], dim=1)

        output1 = self.conv8_1(output)
        output2 = self.conv8_21(output)
        output2 = self.conv8_22(output2)
        output3 = self.conv8_31(output)
        output3 = self.conv8_32(output3)
        output3 = self.conv8_33(output3)
        output4 = self.conv8_4(output)
        output = torch.cat([output1, output2, output3, output4], dim=1)

        output1 = self.conv9_1(output)
        output2 = self.conv9_21(output)
        output2 = self.conv9_22(output2)
        output3 = self.conv9_31(output)
        output3 = self.conv9_32(output3)
        output3 = self.conv9_33(output3)
        output4 = self.conv9_4(output)
        output = torch.cat([output1, output2, output3, output4], dim=1)

        output = self.avg_pool(output)
        output = output.view(x.size(0), -1)
        output = self.linear(output)

        return output
