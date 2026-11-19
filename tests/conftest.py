import itertools

import pytest

# (S, T, rd, rf, sigma) spanning short/long tenors, rate differentials of both signs
MARKETS = [
    (1.10, 1 / 52, 0.053, 0.035, 0.07),
    (1.10, 0.25, 0.053, 0.035, 0.08),
    (1.10, 1.0, 0.045, 0.025, 0.09),
    (1.25, 2.0, -0.005, 0.03, 0.12),  # negative domestic rate, rf > rd
    (0.95, 5.0, 0.04, 0.01, 0.15),
    (150.0, 0.5, 0.001, 0.05, 0.11),  # USDJPY-like scale
]
MONEYNESS = [0.80, 0.95, 1.0, 1.05, 1.20]


def _cases():
    for (S, T, rd, rf, vol), m, is_call in itertools.product(
        MARKETS, MONEYNESS, (True, False)
    ):
        F = S * pow(2.718281828459045, (rd - rf) * T)
        yield (S, F * m, T, rd, rf, vol, is_call)


@pytest.fixture(params=list(_cases()), ids=lambda c: f"S{c[0]}-K{c[1]:.4g}-T{c[2]:.3g}-{'C' if c[6] else 'P'}")
def case(request):
    return request.param
