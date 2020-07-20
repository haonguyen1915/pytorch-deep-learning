import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
# from torchsummary import summary
from torchsummary import summary


class NeuralNet(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        """
        The full connected nearal network.
        Args:
            input_size:
            hidden_size:
            output_size:
        """
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        # self.fc2 = nn.Linear(500, 400)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Sigmoid()

    def forward(self, x):
        """
        The data need to feed forward.
        Its value must be FloatTensor
        Args:
            x: tensor with the shape must be: (batch, input_size)

        Returns:

        """
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.softmax(out)
        return out


if __name__ == "__main__":
    net = NeuralNet(3, 5, 10)

    x = [[1, 2, 3]]
    x_tensor = torch.tensor(x, dtype=torch.float)
    out = net(x_tensor)
    print(x_tensor.shape)
    print(out.type())
