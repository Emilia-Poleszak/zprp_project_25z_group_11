import argparse
import random

from zprp_project_25z_group_11.config import RAW_DATA_DIR, ADDING_SEQUENCE_LENGTH, ADDING_DATA_FILENAME, \
    MULTIPLICATION_DATA_FILENAME, ADDING_RANGE, MULTIPLICATION_RANGE, MULTIPLICATION_SEQUENCE_LENGTH


class Components:

    def __init__(self, length: int, value_range: tuple[float, float]):
        """Data generator for adding and multiplication experiment

        :param length: number of samples in sequence
        :param value_range: value range for first component of element in sequence
        """
        self.length = length
        self.low, self.high = value_range

    def generate_adding(self) -> tuple[list[tuple[float, float]], float]:
        """Generate one sequence for adding/multiplication experiment

        :return: list of components and target value
        """
        first = [random.uniform(self.low, self.high) for _ in range(self.length)]
        second = [0.0] * self.length

        marked_one_idx = random.randint(0, min(9, self.length - 1))
        second[marked_one_idx] = 1.0
        x1 = first[marked_one_idx]

        possible_indices = [i for i in range(self.length) if i != marked_one_idx]
        marked_two_idx = random.choice(possible_indices[:self.length // 2 - 1])
        second[marked_two_idx] = 1.0
        x2 = first[marked_two_idx]

        if x1 == first[0]:
            x1 = 0.0
        elif x2 == first[0]:
            x2 = 0.0

        second[0] = -1.0
        second[-1] = -1.0

        target = 0.5 + (x1 + x2) / 4.0

        return list(zip(first, second)), target

    def generate_multiplication(self) -> tuple[list[tuple[float, float]], float]:
        """Generate one sequence for multiplication experiment

        :return: list of components and target value
        """
        first = [random.uniform(self.low, self.high) for _ in range(self.length)]
        second = [0.0] * self.length

        second[0] = -1.0
        second[-1] = -1.0

        marked_one_idx = random.randint(0, min(9, self.length - 1))
        possible_indices = [i for i in range(self.length // 2 - 1) if i != marked_one_idx]
        marked_two_idx = random.choice(possible_indices) if possible_indices else marked_one_idx

        if marked_one_idx == 0:
            x1 = 1.0
        else:
            x1 = first[marked_one_idx]
            second[marked_one_idx] = 1.0

        second[marked_two_idx] = 1.0
        x2 = first[marked_two_idx]

        target = x1 * x2

        return list(zip(first, second)), target

    def save(self, n_sequences: int, output_file: str) -> None:
        """Save sequences in file

        :param n_sequences: number of sequences to generate
        :param output_file: name of the output file
        """
        if output_file == ADDING_DATA_FILENAME:
            with open(RAW_DATA_DIR / output_file, "w") as f:
                for _ in range(n_sequences):
                    seq, target = self.generate_adding()
                    for a, b in seq:
                        f.write(f"{a:.6f} {b:.1f}\n")
                    f.write(f"# target {target:.6f}\n\n")
        elif output_file == MULTIPLICATION_DATA_FILENAME:
            with open(RAW_DATA_DIR / output_file, "w") as f:
                for _ in range(n_sequences):
                    seq, target = self.generate_multiplication()
                    for a, b in seq:
                        f.write(f"{a:.6f} {b:.1f}\n")
                    f.write(f"{target:.6f}\n")

    def get_data(self, start_line_number: int, filename: str) -> tuple[list[tuple[float, float]], float, int]:
        """
        Reads a batch of training data from source file.
        :param start_line_number: number of the first line of a new data sequence, int.
        :param filename: name of the file to read.
        :return: input data, target value and next starting line number.
        """
        with open(RAW_DATA_DIR / filename, "r") as f:
            for _ in range(start_line_number):
                next(f)

            data = []

            for line in f:
                start_line_number += 1
                elements = line.split()
                if len(elements) == 1:
                    target = float(line)
                    break
                else:
                    value, marker = map(float, elements)
                    data.append((value, marker))

            return data, target, start_line_number


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, choices=["adding", "multiplication"])
    parser.add_argument("--num_sequences", type=int, required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.task == "adding":
        gen = Components(length=ADDING_SEQUENCE_LENGTH, value_range=ADDING_RANGE)
        gen.save(args.num_sequences, ADDING_DATA_FILENAME)

    elif args.task == "multiplication":
        gen = Components(length=MULTIPLICATION_SEQUENCE_LENGTH, value_range=MULTIPLICATION_RANGE)
        gen.save(args.num_sequences, MULTIPLICATION_DATA_FILENAME)
