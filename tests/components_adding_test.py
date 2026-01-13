import random

from zprp_project_25z_group_11.generators.components import Components


def test_sizes():
    rng = random.Random(42)
    generator = Components(100, (-1, 1), rng)
    seq, target = generator.generate_adding()
    assert len(seq) == 100


def test_values_range():
    rng = random.Random(42)
    generator = Components(100, (-1, 1), rng)
    seq, target = generator.generate_adding()
    values = [x for x, _ in seq]

    for v in values:
        assert -1.0 <= v <= 1.0


def test_types():
    rng = random.Random(42)
    generator = Components(100, (-1, 1), rng)
    seq, target = generator.generate_adding()
    values = [x for x, _ in seq]
    markers = [y for _, y in seq]

    assert all(isinstance(v, float) for v in values)
    assert all(isinstance(m, float) for m in markers)
    assert isinstance(target, float)


def test_reproducibility():
    rng1 = random.Random(42)
    generator1 = Components(100, (-1, 1), rng1)
    rng2 = random.Random(42)
    generator2 = Components(100, (-1, 1), rng2)

    s1, t1 = generator1.generate_adding()
    s2, t2 = generator2.generate_adding()

    assert s1 == s2
    assert t1 == t2


def test_not_constant():
    rng = random.Random(42)
    generator = Components(100, (-1, 1), rng)
    s1, t1 = generator.generate_adding()
    s2, t2 = generator.generate_adding()

    values1 = [x1 for x1, _ in s1]
    markers1 = [y1 for _, y1 in s1]
    values2 = [x2 for x2, _ in s2]
    markers2 = [y2 for _, y2 in s2]

    assert not values1 == values2
    assert not markers1 == markers2
    assert not t1 == t2


