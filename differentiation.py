def firstDerivativeO2(values, h):
    """Approximates f' with O(h^2) error from table values only:
    formula (4) at x0, formula (3) at internal points, formula (5) at xm.
    Returns a list of the table length; None where the formula
    would need function values outside the table."""
    m = len(values) - 1
    result = [None] * (m + 1)

    if m < 2:
        return result

    result[0] = firstDerivativeFormula4(h, values[0], values[1], values[2])

    for k in range(1, m):
        result[k] = (values[k + 1] - values[k - 1]) / (2 * h)

    result[m] = (3 * values[m] - 4 * values[m - 1] + values[m - 2]) / (2 * h)

    return result


def firstDerivativeO4(values, h):
    """Approximates f' with O(h^4) error from table values only:
    formula (7) at x0, (8) at x1, (9) at internal points, (10) at x_{m-1},
    (11) at xm. Returns a list of the table length; None where the formula
    would need function values outside the table."""
    m = len(values) - 1
    result = [None] * (m + 1)

    if m < 4:
        return result

    result[0] = (
        -25 * values[0]
        + 48 * values[1]
        - 36 * values[2]
        + 16 * values[3]
        - 3 * values[4]
    ) / (12 * h)
    result[1] = (
        -3 * values[0] - 10 * values[1] + 18 * values[2] - 6 * values[3] + values[4]
    ) / (12 * h)

    for k in range(2, m - 1):
        result[k] = (
            values[k - 2] - 8 * values[k - 1] + 8 * values[k + 1] - values[k + 2]
        ) / (12 * h)

    result[m - 1] = (
        3 * values[m]
        + 10 * values[m - 1]
        - 18 * values[m - 2]
        + 6 * values[m - 3]
        - values[m - 4]
    ) / (12 * h)
    result[m] = (
        25 * values[m]
        - 48 * values[m - 1]
        + 36 * values[m - 2]
        - 16 * values[m - 3]
        + 3 * values[m - 4]
    ) / (12 * h)

    return result


def secondDerivativeO2(values, h):
    """Approximates f'' with O(h^2) error from table values only:
    formula (6) at internal points. Returns a list of the table length;
    None at the first and the last table points."""
    m = len(values) - 1
    result = [None] * (m + 1)

    if m < 2:
        return result

    for k in range(1, m):
        result[k] = (values[k + 1] - 2 * values[k] + values[k - 1]) / (h * h)

    return result


def firstDerivativeFormula4(h, f0, f1, f2):
    """Basic sum of formula (4) from three values of f taken
    at x, x+h, x+2h; used by the optimal h experiment."""
    return (-3 * f0 + 4 * f1 - f2) / (2 * h)
