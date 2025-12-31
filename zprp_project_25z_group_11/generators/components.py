import random

from zprp_project_25z_group_11.config import RAW_DATA_DIR, ADDING_DATA_FILENAME, ADDING_SEQUENCES, MULTIPLICATION_DATA_FILENAME, MULTIPLICATION_SEQUENCES

class Components:

    def __init__(self, min_no_samples: int, value_range: tuple[float, float]):
        """Data generator for adding and multiplication experiment

        :param min_no_samples: minimal number of samples in sequence
        :param value_range: value range for first component of element in sequence
        """
        self.min_no_samples = min_no_samples
        self.low, self.high = value_range


    def generate_adding(self) -> tuple[list[tuple[float, float]], float]:
        """Generate one sequence for adding experiment

        :return: list of components and target value
        """
        length = random.randint(self.min_no_samples, int(self.min_no_samples + self.min_no_samples / 10))

        first = [random.uniform(self.low, self.high) for _ in range(length)]
        second = [0.0] * length

        marked_one_idx = random.randint(0, min(9, length - 1))
        possible_indices = [i for i in range(length // 2 - 1) if i != marked_one_idx]
        marked_two_idx = random.choice(possible_indices) if possible_indices else marked_one_idx

        second[marked_one_idx] = 1.0
        second[marked_two_idx] = 1.0

        if marked_one_idx == 0:
            first[0] = 0.0
        else:
            second[0] = -1.0
        second[-1] = -1.0

        x1 = first[marked_one_idx]
        x2 = first[marked_two_idx]
        target = 0.5 + (x1 + x2) / 4.0

        return list(zip(first, second)), target

    def generate_multiplication(self) -> tuple[list[tuple[float, float]], float]:
        """Generate one sequence for multiplication experiment

        :return: list of components and target value
        """
        length = random.randint(self.min_no_samples, int(self.min_no_samples + self.min_no_samples / 10))

        first = [random.uniform(self.low, self.high) for _ in range(length)]
        second = [0.0] * length

        second[0] = -1.0
        second[-1] = -1.0

        marked_one_idx = random.randint(0, min(9, length - 1))
        possible_indices = [i for i in range(length // 2 - 1) if i != marked_one_idx]
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

    def save(self, n_sequences, output_file):
        """
        Save training data in txt file.
        :param n_sequences: number of sequences
        :param output_file: specified name of output file
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

    def get_data(self, start_line_number):
        """
        Reads a batch of training data from source file.
        :param start_line_number: number of the first line of a new data sequence, int.
        :return: input data, target value and next starting line number.
        """
        with open(RAW_DATA_DIR/MULTIPLICATION_DATA_FILENAME, "r") as f:
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


if __name__ == '__main__':
    gen4 = Components(min_no_samples=100, value_range=(-1.0, 1.0))
    gen4.save(ADDING_SEQUENCES, ADDING_DATA_FILENAME)

    gen5 = Components(min_no_samples=100, value_range=(0.0, 1.0))
    gen5.save(MULTIPLICATION_SEQUENCES, MULTIPLICATION_DATA_FILENAME)


