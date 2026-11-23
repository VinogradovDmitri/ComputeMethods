import math

AST_TOLERANCE = 1e-12
MAX_AST_DEGREE = 8


def leftRectangle(func, a, b):
    """Left rectangle formula on [a, b]"""
    return (b - a) * func(a)


def rightRectangle(func, a, b):
    """Right rectangle formula on [a, b]"""
    return (b - a) * func(b)


def middleRectangle(func, a, b):
    """Middle rectangle formula on [a, b]"""
    return (b - a) * func((a + b) / 2.0)


def trapezoid(func, a, b):
    """Trapezoid formula on [a, b]"""
    return (b - a) * (func(a) + func(b)) / 2.0


def simpson(func, a, b):
    """Simpson formula on [a, b] with one middle node"""
    middle = (a + b) / 2.0
    return (b - a) / 6.0 * (func(a) + 4.0 * func(middle) + func(b))


SIMPLE_FORMULAS = [
    ("Left rectangle", leftRectangle),
    ("Right rectangle", rightRectangle),
    ("Middle rectangle", middleRectangle),
    ("Trapezoid", trapezoid),
    ("Simpson", simpson),
]


def exactMonomialIntegral(a, b, degree):
    """Exact value of int_a^b x^degree dx"""
    return (b ** (degree + 1) - a ** (degree + 1)) / (degree + 1.0)


def algebraicDegreeOfAccuracy(formula, a, b, maxDegree=MAX_AST_DEGREE):
    """Largest d such that the formula is exact for x^0, ..., x^d.
    Returns -1 if the formula is not exact even for constants"""
    degree = -1

    for power in range(maxDegree + 1):

        def monomial(x, exponent=power):
            return x**exponent

        try:
            exact = exactMonomialIntegral(a, b, power)
        except (OverflowError, ZeroDivisionError):
            break

        try:
            approx = formula(monomial, a, b)
        except Exception:
            break

        tolerance = AST_TOLERANCE * max(1.0, abs(exact))

        if not math.isfinite(approx) or abs(exact - approx) > tolerance:
            break

        degree = power

    return degree
