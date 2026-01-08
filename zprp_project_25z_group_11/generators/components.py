import random

from zprp_project_25z_group_11.config import RAW_DATA_DIR

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
        # length = random.randint(self.min_no_samples, int(self.min_no_samples + self.min_no_samples / 10))
        length = self.min_no_samples

        first = [random.uniform(self.low, self.high) for _ in range(length)]
        second = [0.0] * length

        marked_one_idx = random.randint(0, min(9, length - 1))
        second[marked_one_idx] = 1.0
        x1 = first[marked_one_idx]

        possible_indices = [i for i in range(length // 2 - 1) if i != marked_one_idx]
        marked_two_idx = random.choice(possible_indices[:self.min_no_samples // 2 - 1]) if possible_indices else marked_one_idx
        second[marked_two_idx] = 1.0
        x2 = first[marked_two_idx]

        if marked_one_idx == 0:
            x1 = 0.0

        second[0] = -1.0
        second[-1] = -1.0

        target = 0.5 + (x1 + x2) / 4.0

        self.save(list(zip(first, second)), target)
        return list(zip(first, second)), target

    def save(self, seq, target):
        """Saves single sequence to file."""
        filename = 'adding' + str(self.min_no_samples) + '.txt'
        with open(RAW_DATA_DIR / filename, "a") as f:
            for a, b in seq:
                f.write(f"{a:.6f} {b:.1f}\n")
            f.write(f"# target {target:.6f}\n\n")

if __name__ == '__main__':
    gen4 = Components(min_no_samples=100, value_range=(-1.0, 1.0))
