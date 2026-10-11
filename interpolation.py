import math
import random

NODE_TOLERANCE = 1e-12
MAX_RANDOM_ATTEMPTS = 100000
MAX_NODES_WARN = 1000


def arePairwiseDistinct(nodes, tolerance=NODE_TOLERANCE):
    ordered = sorted(nodes)

    return all(
        ordered[i + 1] - ordered[i] > tolerance for i in range(len(ordered) - 1)
    )


def buildEquidistantNodes(segment, count):
    if count < 2:
        raise ValueError(f"the number of nodes must be >= 2, got {count}")

    step = segment.len() / (count - 1)
    nodes = [segment.left + i * step for i in range(count)]

    if not arePairwiseDistinct(nodes):
        raise ValueError(
            "The equidistant nodes are not pairwise distinct: "
            f"the interval [{segment.left}, {segment.right}] "
            f"(width {segment.len()}) is too narrow for {count} nodes "
            f"at the tolerance {NODE_TOLERANCE}"
        )

    return nodes


def buildRandomNodes(segment, count):
    if count < 2:
        raise ValueError(f"the number of nodes must be >= 2, got {count}")

    spacing = max(NODE_TOLERANCE, math.ulp(segment.left), math.ulp(segment.right))

    if segment.len() / spacing + 1 < count:
        raise ValueError(
            f"The interval [{segment.left}, {segment.right}] "
            f"(width {segment.len()}) is too narrow for {count} random nodes: "
            f"the minimum node spacing is {spacing}"
        )

    nodes = []
    attempts = 0

    while len(nodes) < count:
        if attempts == MAX_RANDOM_ATTEMPTS:
            raise ValueError(
                "Random node placement did not converge: "
                f"placed only {len(nodes)} of {count} nodes "
                f"within {MAX_RANDOM_ATTEMPTS} attempts "
                f"(the nodes must stay at least {NODE_TOLERANCE} apart)"
            )

        attempts = attempts + 1
        node = random.uniform(segment.left, segment.right)

        if all(abs(node - other) > NODE_TOLERANCE for other in nodes):
            nodes.append(node)

    return sorted(nodes)


def sortTableByDistance(nodes, values, x):
    """Sort the table, nodes nearest to x go first"""
    pairs = sorted(zip(nodes, values), key=lambda pair: abs(pair[0] - x))

    return [pair[0] for pair in pairs], [pair[1] for pair in pairs]


def lagrangeCoefficients(nodes, x):
    """ ] Q_k(x) = Π_{i!=k}(x - x_i)
    ω'(x) = 1·Q_k(x) + (x - x_k)·Q_k'(x)
    ω'(x_k) = Q_k(x_k) = Π_{i≠k}(x_k - x_i)
    lk(x) = prod_{i != k} (x - xi) / (xk - xi), k = 0..n"""
    coefficients = []

    for k, node in enumerate(nodes):
        coefficient = 1.0

        for i, other in enumerate(nodes):
            if i != k:
                coefficient *= (x - other) / (node - other)

        coefficients.append(coefficient)

    return coefficients


def lagrangeValue(nodes, values, x):
    """Pn(x) in the Lagrange form; also returns the coefficients lk(x)"""
    coefficients = lagrangeCoefficients(nodes, x)
    value = sum(c * v for c, v in zip(coefficients, values))

    return value, coefficients


def lebesgueValue(coefficients):
    """Lebesgue function: lambda_n(x) = sum |lk(x)|"""
    return sum(abs(c) for c in coefficients)


def dividedDifferenceTable(nodes, values):
    """table[order][k] = f[x_k, ..., x_{k + order}], order = 0..n"""
    table = [list(values)]

    for order in range(1, len(nodes)):
        prev = table[order - 1]
        table.append(
            [
                (prev[i + 1] - prev[i]) / (nodes[i + order] - nodes[i])
                for i in range(len(prev) - 1)
            ]
        )

    return table


def newtonCoefficients(table):
    """Ak = f[x0, ..., xk] = table[k][0]"""
    return [row[0] for row in table]


def newtonValue(nodes, coefficients, x):
    """Pn(x) in the Newton form:
    Pn(x) = A0 + A1*(x-x0) + A2*(x-x0)(x-x1) + ..."""
    value = coefficients[0]
    product = 1.0

    for i in range(1, len(coefficients)):
        product *= x - nodes[i - 1]
        value += coefficients[i] * product

    return value
