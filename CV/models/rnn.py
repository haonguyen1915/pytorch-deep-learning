from torch import nn
import torch


class RecurrentNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RecurrentNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """

        Args:
            x: shape: batch_size, sequence_length, input_size

        Returns:

        """
        # Set initial hidden and cell states
        # tensor of shape (num_layers, batch_size, hidden_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        # Forward propagate LSTM
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


if __name__ == "__main__":
    # Hyper-parameters
    sequence_length = 28
    input_size = 28
    hidden_size = 200
    num_layers = 2
    num_classes = 10
    batch_size = 100
    model = RecurrentNet(input_size, hidden_size, num_layers, num_classes)
    x = torch.rand((batch_size, sequence_length, input_size))
    y = model(x)
