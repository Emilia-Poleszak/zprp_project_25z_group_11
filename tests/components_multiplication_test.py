import random

from zprp_project_25z_group_11.generators.components import Components


def test_sizes():
    rng = random.Random(42)
    generator = Components(100, (0, 1), rng)
    seq, target = generator.generate_multiplication()

    assert len(seq) == 100


def test_values_range():
    rng = random.Random(42)
    generator = Components(100, (0, 1), rng)
    seq, target = generator.generate_multiplication()

    values = [x for x, _ in seq]

    for v in values:
        assert 0.0 <= v <= 1.0


def test_types():
    rng = random.Random(42)
    generator = Components(100, (0, 1), rng)
    seq, target = generator.generate_multiplication()
    values = [x for x, _ in seq]
    markers = [y for _, y in seq]

    assert all(isinstance(v, float) for v in values)
    assert all(isinstance(m, float) for m in markers)
    assert isinstance(target, float)


def test_reproducibility():
    rng1 = random.Random(42)
    generator1 = Components(100, (0, 1), rng1)
    rng2 = random.Random(42)
    generator2 = Components(100, (0, 1), rng2)

    s1, t1 = generator1.generate_multiplication()
    s2, t2 = generator2.generate_multiplication()

    assert s1 == s2
    assert t1 == t2


def test_not_constant():
    rng = random.Random(42)
    generator = Components(100, (0, 1), rng)
    s1, t1 = generator.generate_multiplication()
    s2, t2 = generator.generate_multiplication()

    values1 = [x1 for x1, _ in s1]
    markers1 = [y1 for _, y1 in s1]
    values2 = [x2 for x2, _ in s2]
    markers2 = [y2 for _, y2 in s2]

    assert not values1 == values2
    assert not markers1 == markers2
    assert not t1 == t2


def test_target_value_and_markers():
    rng = random.Random(42)
    generator = Components(100, (0, 1), rng)
    seq, target = generator.generate_multiplication()

    marked = [v for v, m in seq if m == 1.0]

    assert seq[0][1] == -1.0
    assert seq[-1][1] == -1.0

    assert 1 <= len(marked) <= 2

    if len(marked) == 1:
        assert abs(marked[0] - target) < 1e-9
    else:
        assert abs(marked[0] * marked[1] - target) < 1e-9
