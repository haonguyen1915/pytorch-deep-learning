from CV.data_generator.calculator_generator import CalculatorGenerator
from CV.models.rnn import RecurrentNet
import torch
# from magic.lib_ai.torch_utils import save_checkpoints, load_checkpoint
from magic.lib_cm import get_cfd
import torch.nn as nn
import os

PRJ_DIR = os.getcwd()


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


def evaluation(data_generator):
    model = torch.load(f"{PRJ_DIR}/weights/RecurrentNet.pt")
    model.eval()
    inputs, labels = data_generator.generator_calculator_batch(500)
    inputs_new, labels = process_inputs(inputs, labels)

    outputs = model(inputs_new)
    preds = torch.argmax(outputs, dim=1)
    num_correct = torch.sum(preds == labels)

    visualize_sample(inputs, preds, data_generator)
    print(f"Evaluation result: {num_correct}/{inputs.shape[0]} "
          f"({(float(num_correct) / labels.shape[0]) * 100}%)")


def visualize_sample(inputs, preds, data_generator):
    counter = 0
    for inp, pred in zip(inputs, preds):
        if counter > 100:
            break
        inp = inp
        preds = preds
        output = data_generator.decode(inp, pred.item())
        print(f"Samples output: {output}")
        counter += 1


def process_inputs(inputs, labels):
    """

    Args:
        inputs:
        labels:

    Returns:

    """
    inputs = torch.from_numpy(inputs).float()
    labels = torch.from_numpy(labels).long()
    inputs_padded = torch.zeros((inputs.shape[0], 48))
    inputs_padded[:, :inputs.shape[1]] = inputs[:, :inputs.shape[1]]
    inputs_padded = inputs_padded.view(inputs.shape[0], 6, 8)
    return inputs_padded, labels


def train(model, data_generator, num_iterator):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for ite in range(num_iterator):
        inputs, labels = data_generator.generator_calculator_batch(10)
        inputs, labels = process_inputs(inputs, labels)

        outputs = model(inputs)

        # Calculate the loss.
        # outputs: a FloatTensor, labels: LongTensor
        loss = criterion(outputs, labels)

        # Update gradient
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if ite % 100 == 0:
            print(outputs.shape)
            preds = torch.argmax(outputs, dim=1)
            num_correct = torch.sum(preds == labels)
            print(
                f"Iterator: {ite}, loss: {loss.item()}, accuracy: {num_correct / labels.shape[0]}")
    torch.save(model, f"{PRJ_DIR}/weights/{str(model.__class__.__name__)}.pt")


if __name__ == "__main__":
    ops = ("+", "*")
    cal_generator = CalculatorGenerator(ops)

    sequence_length = 6
    input_size = 8
    hidden_size = 128
    num_layers = 2
    num_classes = cal_generator.num_output
    batch_size = 100

    net = RecurrentNet(input_size, hidden_size, num_layers, num_classes)
    train(net, cal_generator, num_iterator=2000)
    # evaluation(cal_generator)
