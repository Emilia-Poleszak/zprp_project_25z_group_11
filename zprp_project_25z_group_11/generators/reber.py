from random import choice

from zprp_project_25z_group_11.config import RAW_DATA_DIR, REBER_SAMPLES

class EmbeddedReberGrammar:
    def __init__(self):
        """ Data generator for embedded Reber Grammar experiment.
        """
        self.grammar = { 0: [('B', 1)],
                         1: [('T', 2), ('P', 3)],
                         2: [('S', 2), ('X', 4)],
                         3: [('T', 3), ('V', 5)],
                         4: [('X', 3), ('S', 6)],
                         5: [('P', 4), ('V', 6)],
                         6: [('E', 7)]}
        pass

    def save(self, no_samples: int):
        """ Save declared number of Embedded Reber Grammar sequences into raw data file.

        :param no_samples: Number of Embedded Reber Grammar sequences to save
        """
        with open(RAW_DATA_DIR / 'reber.txt', 'w') as file:
            for i in range(no_samples):
                file.write(self.generate() + '\n')

    def generate(self) -> str:
        """ Generate a single embedded Reber Grammar transition.

        :return: list of letters
        """
        sequence = 'B' + choice(['T', 'P'])
        state = 0
        while state < len(self.grammar):
            transition, state = choice(self.grammar[state])
            sequence += transition
        sequence += sequence[1] + 'E'
        return sequence


if __name__ == '__main__':
    grammar = EmbeddedReberGrammar()
    grammar.save(REBER_SAMPLES)