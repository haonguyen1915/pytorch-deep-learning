from data_generator.calculator_generator import CalculatorGenerator
from models.full_connected import NeuralNet
import torch
from haolib.lib_ai import save_checkpoints, load_checkpoint
from haolib.lib_cm import get_cfd

PRJ_DIR = get_cfd()


def evaluation(data_generator):
    model = torch.load(f"{PRJ_DIR}/weights/NeuralNet.pt")
    model.eval()
    inputs, labels = data_generator.generator_calculator_batch(500)

    inputs = torch.from_numpy(inputs).float()
    labels = torch.from_numpy(labels).long()

    outputs = model(inputs)
    preds = torch.argmax(outputs, dim=1)
    num_correct = torch.sum(preds == labels)

    visualize_sample(inputs, preds, data_generator)
    print(f"Evaluation result: {num_correct}/{inputs.shape[0]} "
          f"({(num_correct / labels.shape[0]) * 100}%)")


def visualize_sample(inputs, preds, data_generator):
    counter = 0
    for inp, pred in zip(inputs, preds):
        if counter > 100:
            break
        inp = inp.numpy()
        preds = preds
        output = data_generator.decode(inp, pred.item())
        print(f"Samples output: {output}")
        counter += 1


def train(model, data_generator, num_iterator):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for ite in range(num_iterator):
        inputs, labels = data_generator.generator_calculator_batch(10)
        inputs = torch.from_numpy(inputs).float()
        labels = torch.from_numpy(labels).long()
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
    net = NeuralNet(input_size=cal_generator.num_feature,
                    hidden_size=200,
                    output_size=cal_generator.num_output)
    train(net, cal_generator, num_iterator=2000)
    evaluation(cal_generator)
