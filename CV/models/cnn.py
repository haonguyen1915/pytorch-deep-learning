from torch import nn
import torch


class CNNNet(nn.Module):
    def __init__(self, num_lass):
        super(CNNNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 5, 3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(5, 10, 3, stride=1, padding=1)
        self.fc1 = nn.Linear(10 * 3 * 4, 200)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(200, num_lass)

    def forward(self, x):
        """

        Args:
            x: (batch, channel, width, height)

        Returns:

        """
        out = self.conv1(x)
        out = self.pool(out)
        out = self.conv2(out)
        out = out.view(-1, 10 * 3 * 4)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out


if __name__ == "__main__":
    x = torch.rand((2, 1, 6, 8))
    net = CNNNet(10)
    outs = net(x)
