import random

class Components:

    def __init__(self, min_no_samples: int, value_range: tuple[float, float]):
        """Data generator for adding and multiplication experiment

        :param min_no_samples: minimal number of samples in sequence
        :param value_range: value range for first component of element in sequence
        """
        self.min_no_samples = min_no_samples
        self.low, self.high = value_range


    def generate(self) -> tuple[list[tuple[float, float]], float]:
        """Generate one sequence for adding/multiplication experiment

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
        target = 0.5 + (x1 + x2) / 4.0 # needs modification cause multiplication has different target

        return list(zip(first, second)), target

if __name__ == '__main__':
    gen4 = Components(min_no_samples=100, value_range=(-1.0, 1.0))
