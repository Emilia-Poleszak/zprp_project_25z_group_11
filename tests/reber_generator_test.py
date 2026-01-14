from unittest.mock import patch, mock_open, ANY

import pytest

from zprp_project_25z_group_11.generators.reber import EmbeddedReberGrammar

@pytest.fixture
def generator():
    return EmbeddedReberGrammar()

def test_generator_grammar_structure(generator):
    minimum_length = 6
    for _ in range(100):
        sequence = generator.generate()

        assert len(sequence) >= minimum_length
        assert sequence.startswith('B')
        assert sequence.endswith('E')

        assert sequence[1] == sequence[-2]
        assert sequence[1] in ['T', 'P']

        inner_seq = sequence[2:-2]
        current_state = 0
        valid_path = True

        for char in inner_seq:
            possible_transitions = generator.grammar.get(current_state)

            next_state = None
            if possible_transitions:
                for trans_char, target_state in possible_transitions:
                    if trans_char == char:
                        next_state = target_state
                        break

            if next_state is None:
                valid_path = False
                break

            current_state = next_state

        assert valid_path

def test_save_creates_file(generator):
    num_samples = 10
    m = mock_open()

    with patch("builtins.open", m):
        generator.save(num_samples)

    m.assert_called_once_with(ANY, "w")

    handle = m()
    assert handle.write.call_count == num_samples