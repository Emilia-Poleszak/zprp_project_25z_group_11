import random

from zprp_project_25z_group_11.config import RAW_DATA_DIR, ADDING_DATA_FILENAME, ADDING_SEQUENCES, MULTIPLICATION_DATA_FILENAME, MULTIPLICATION_SEQUENCES

class Components:

    def __init__(self, min_no_samples: int, value_range: tuple[float, float]):
        self.min_no_samples = min_no_samples
        self.low, self.high = value_range


    def generate(self):
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

    def save(self, n_sequences, output_file):
        with open(RAW_DATA_DIR / output_file, "w") as f:
            for _ in range(n_sequences):
                seq, target = self.generate()
                for a, b in seq:
                    f.write(f"{a:.6f} {b:.1f}\n")
                f.write(f"# target {target:.6f}\n\n")

if __name__ == '__main__':
    gen4 = Components(min_no_samples=100, value_range=(-1.0, 1.0))
    gen4.save(ADDING_SEQUENCES, ADDING_DATA_FILENAME)

    gen5 = Components(min_no_samples=100, value_range=(0.0, 1.0))
    gen5.save(MULTIPLICATION_SEQUENCES, MULTIPLICATION_DATA_FILENAME)


