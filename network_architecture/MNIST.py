import torch
import torch.nn as nn


class MNIST(nn.Module):
    def __init__(self, shrinkage_ratio):
        super(MNIST, self).__init__()
        self.shrinkage_ratio = shrinkage_ratio

        self.conv1 = nn.Conv2d(1, int(16 * self.shrinkage_ratio), kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(int(16 * self.shrinkage_ratio), int(32 * self.shrinkage_ratio), kernel_size=3, stride=1,
                               padding=1)
        self.linear = nn.Linear(int(32 * self.shrinkage_ratio) * 7 * 7, 10)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, 1, 28, 28)

        conv_output = self.conv1(x)
        conv_output = torch.max_pool2d(torch.relu(conv_output), 2)
        conv_output = self.conv2(conv_output)
        conv_output = torch.max_pool2d(torch.relu(conv_output), 2)

        conv_output = conv_output.view(batch_size, int(32 * self.shrinkage_ratio) * 7 * 7)
        output = self.linear(conv_output)

        return output
