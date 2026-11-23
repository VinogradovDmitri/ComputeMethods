import math

import sympy as sp

from methods import tryEvaluate

X = sp.symbols("x")

RUNGE_TOLERANCE = 1e-10
RUNGE_MAX_DOUBLINGS = 18


class RungeResult(object):
    def __init__(self, value, previous, errorEstimate, intervals, converged):
        self.value = value
        self.previous = previous
        self.errorEstimate = errorEstimate
        self.intervals = intervals
        self.converged = converged


def evaluateAt(func, point):
    value = tryEvaluate(func, point)

    if not math.isfinite(value):
        raise ValueError(f"the integrand is not finite at x = {point:.12g}")

    return value


def simpsonIntegrate(func, a, b, n):
    """Composite Simpson formula with n subintervals (n is made even)"""
    if not a < b:
        raise ValueError(f"the interval must satisfy a < b, got a = {a}, b = {b}")

    if n < 2:
        raise ValueError(f"the number of subintervals must be >= 2, got {n}")

    if n % 2 != 0:
        n = n + 1

    h = (b - a) / n
    terms = [evaluateAt(func, a), evaluateAt(func, b)]

    for i in range(1, n, 2):
        terms.append(4.0 * evaluateAt(func, a + i * h))

    for i in range(2, n, 2):
        terms.append(2.0 * evaluateAt(func, a + i * h))

    return math.fsum(terms) * h / 3.0


def rungeIntegrate(
    func, a, b, tolerance=RUNGE_TOLERANCE, maxDoublings=RUNGE_MAX_DOUBLINGS
):
    """Simpson values with h and h/2; the error of I_{h/2} is estimated
    as |I_{h/2} - I_h| / 15 (Runge rule). Returns RungeResult"""
    if not a < b:
        raise ValueError(f"the interval must satisfy a < b, got a = {a}, b = {b}")

    if maxDoublings < 1:
        raise ValueError("maxDoublings must be >= 1")

    n = 2
    previous = simpsonIntegrate(func, a, b, n)
    lastPrevious = previous
    estimate = float("nan")

    for _ in range(maxDoublings):
        n = n * 2
        current = simpsonIntegrate(func, a, b, n)
        estimate = abs(current - previous) / 15.0

        if estimate <= tolerance * max(1.0, abs(current)):
            return RungeResult(current, previous, estimate, n, True)

        lastPrevious = previous
        previous = current

    return RungeResult(previous, lastPrevious, estimate, n, False)


def exactIntegral(expr, a, b):
    """Value of the definite integral by sympy, or None if sympy cannot
    produce a finite real value"""
    try:
        result = sp.integrate(expr, (X, a, b))

        if result.has(sp.Integral, sp.nan, sp.zoo, sp.oo, -sp.oo):
            return None

        value = sp.N(result, 30)
        real = float(sp.re(value))
        imaginary = float(sp.im(value))
    except Exception:
        return None

    if not math.isfinite(real) or not math.isfinite(imaginary):
        return None

    if abs(imaginary) > 1e-10 * max(1.0, abs(real)):
        return None

    return real


def buildVandermonde(nodes):
    """V[j][k] = x_k^j, j = 0..N-1, k = 0..N-1"""
    return [[node**j for node in nodes] for j in range(len(nodes))]


def solveQuadratureSystem(nodes, moments):
    """Coefficients A_k from sum_k A_k*x_k^j = mu_j, j = 0..N-1;
    returns <coefficients, max residual>"""
    n = len(nodes)

    if n == 0:
        raise ValueError("there must be at least one node")

    if len(moments) != n:
        raise ValueError(
            f"the number of moments ({len(moments)}) must equal "
            f"the number of nodes ({n})"
        )

    matrixValues = buildVandermonde(nodes)

    try:
        matrix = sp.Matrix(
            [[sp.Float(value, 30) for value in row] for row in matrixValues]
        )
        rhs = sp.Matrix([sp.Float(value, 30) for value in moments])
        solution = matrix.LUsolve(rhs)
    except Exception as error:
        raise ValueError(f"cannot solve the linear system: {error}")

    coefficients = []

    for value in solution:
        try:
            real = float(sp.re(value))
            imaginary = float(sp.im(value))
        except Exception as error:
            raise ValueError(f"cannot read a coefficient from the solution: {error}")

        if not math.isfinite(real):
            raise ValueError("the solution contains a non-finite coefficient")

        if abs(imaginary) > 1e-8 * max(1.0, abs(real)):
            raise ValueError("the solution contains a complex coefficient")

        coefficients.append(real)

    residual = max(
        abs(
            math.fsum(matrixValues[j][k] * coefficients[k] for k in range(n))
            - moments[j]
        )
        for j in range(n)
    )

    return coefficients, residual


def quadratureValue(coefficients, values):
    """Approximate integral sum_k A_k*f(x_k)"""
    return math.fsum(a * value for a, value in zip(coefficients, values))
