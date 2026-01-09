import random

from zprp_project_25z_group_11.config import RAW_DATA_DIR

class Components:

    def __init__(self, length: int, value_range: tuple[float, float]):
        """Data generator for adding and multiplication experiment

        :param length: number of samples in sequence
        :param value_range: value range for first component of element in sequence
        """
        self.length = length
        self.low, self.high = value_range


    def generate(self, save: bool) -> tuple[list[tuple[float, float]], float]:
        """Generate one sequence for adding/multiplication experiment

        :param save: indicator whether to save the generated sequence
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

        if save:
            self.save(list(zip(first, second)), target)
        return list(zip(first, second)), target

    def save(self, seq: list[tuple[float, float]], target: float):
        """Saves single sequence to file.

        :param seq: sequence of components
        :param target: target value of the sequence
        """
        filename = 'adding' + str(self.length) + '.txt'
        with open(RAW_DATA_DIR / filename, "a") as f:
            for a, b in seq:
                f.write(f"{a:.6f} {b:.1f}\n")
            f.write(f"# target {target:.6f}\n\n")

if __name__ == '__main__':
    gen4 = Components(length=100, value_range=(-1.0, 1.0))
