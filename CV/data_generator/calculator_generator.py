import numpy as np
import random
from typing import *

PLUS = "+"
MINUS = "-"
MULTIPLE = "*"
OPERATORS = [PLUS, MINUS, MULTIPLE]


class CalculatorGenerator:
    def __init__(self, ops=(PLUS, MULTIPLE), max_number=10, max_length=10):
        self.max_number = max_number
        self.max_length = max_length
        self.ops = ops
        self.num_feature = len(ops) + max_length * 2
        self.num_output = 50

    def generator_calculator_batch(self, bs=10):
        """
        Generate batch data

        Args:
            bs: batch size

        Returns: features shape: (bs, num_feature), labels: (bs, )

        """
        features = []
        labels = []
        for _ in range(bs):
            X, y = self.generator_calculator_single()
            features.append(X)
            labels.append(y)
        features = np.stack(features, axis=0)
        labels = np.array(labels)
        return features, labels

    def generator_calculator_single(self):
        """
        Generate a single data

        Args:

        Returns:

        """
        a = random.randint(0, self.max_number)
        b = random.randint(0, self.max_number)
        op = random.choice(self.ops)
        features, label = self.encode(a, op, b)
        return features, label

    def encode(self, a: int, op: int, b: int):
        """
        Generate feature for calculator
        Args:
            a:
            b:
            op: The operator

        Returns:

        """
        if op == PLUS:
            result = a + b
        elif op == MINUS:
            result = a - b
        elif op == MULTIPLE:
            result = a * b
        else:
            raise ValueError(f"No support operator: {op}")
        a_feature = self._convert_int_2_binary(a, self.max_length)
        b_feature = self._convert_int_2_binary(b, self.max_length)

        operator_feature = self.one_hot_encode(self.ops.index(op), len(self.ops))
        features = np.concatenate(
            (a_feature, operator_feature, b_feature),
            axis=0)
        label = result if result < self.num_output else self.num_output-1
        return features, label

    def decode(self, featurure: np, label: int = None):
        """
        Decode the calculator feature

        Args:
            featurure: numpy
            label:

        Returns:

        """
        features = list(featurure.astype(np.int))
        a_feature = features[: self.max_length]
        op_feature = features[self.max_length: len(self.ops) + self.max_length]
        b_feature = features[self.max_length + len(self.ops):]
        print(op_feature)
        op_index = op_feature.index(1)
        out_str = f"{str(self.convert_binary_2_int(a_feature))}" \
                  f"{self.ops[op_index]}" \
                  f"{str(self.convert_binary_2_int(b_feature))}" \
                  f"=" \
                  f"{str(label)}"
        return out_str

    @staticmethod
    def _convert_int_2_binary(number: int, length=10):
        """
        Convert a integer to binary feature
        example: self._convert_int_2_binary(4, 6) -> [0 0 0 1 0 1]
        Args:
            number:
            length:

        Returns:l

        """
        binary = format(number, f'0{length}b')
        feature = [int(n) for n in binary]
        feature = np.array(feature)
        return feature

    @staticmethod
    def one_hot_encode(number: int, length: int = 10):
        """
        Create one hot encode
        example: one_hot_encode(4, 5) -> [0. 0. 0. 0. 1.]
        Args:
            number: number will be encoded
            length: max len

        Returns:

        """
        if length < number + 1:
            raise ValueError(f"lenghth: {length} must be greater than number: {number + 1}")
        one_hot = np.zeros(length)
        one_hot[number] = 1
        # print(one_hot)
        return one_hot

    @staticmethod
    def convert_binary_2_int(binary: List):
        """
        Convert the bin
        Args:
            binary: list

        Returns:

        """
        binary_str = ""
        for b in binary:
            binary_str += str(b)
        result = int(binary_str, 2)
        return result


if __name__ == "__main__":
    calculator_generator = CalculatorGenerator(max_length=10)
    features_, label = calculator_generator.generator_calculator_single()
    output = calculator_generator.decode(features_, label)
    print(output)
