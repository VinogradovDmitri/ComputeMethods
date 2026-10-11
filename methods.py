import math
import sympy as sp

from segment import Segment

MAX_STEPS = 10000
M1_SAMPLES = 100
PROBE_POINTS = (-1000.0, -10.0, -2.0, -1.0, 0.0, 1.0, 2.0, 10.0, 1000.0)


class ApproximationInfo(object):
    def __init__(
        self, name, approximation, stepsCount, errorEstimate=None, error=None
    ):
        self.name = name
        self.approximation = approximation
        self.stepsCount = stepsCount
        self.errorEstimate = errorEstimate
        self.error = error


def tryEvaluate(func, x):
    try:
        value = func(x)
        return value if math.isfinite(value) else float("nan")
    except Exception:
        return float("nan")


def anyProbeFinite(func):
    return any(math.isfinite(tryEvaluate(func, point)) for point in PROBE_POINTS)


def differentiate(expr, x, times=1):
    try:
        for _ in range(times):
            expr = sp.diff(expr, x)
        func = sp.lambdify(x, expr, "math")
    except Exception:
        return None

    return func if anyProbeFinite(func) else None


def buildFunction(expression):
    if expression == "":
        raise ValueError("Please enter a function, e.g. x^2 - 2*sin(x)")

    x = sp.symbols("x")

    try:
        expr = sp.sympify(expression)
        func = sp.lambdify(x, expr, "math")
    except Exception as error:
        raise ValueError(
            f'Cannot parse the expression "{expression}": {error}'
        )

    if not anyProbeFinite(func):
        raise ValueError(
            f'The function "{expression}" cannot be evaluated: '
            f"it is not finite at any probe point x in {PROBE_POINTS}"
        )

    return func, differentiate(expr, x), differentiate(expr, x, 2)


def polynomialInfo(expression):
    """<degree, leading coefficient> if the expression is a polynomial in x,
    otherwise (None, None). Used for the divided difference control"""
    x = sp.symbols("x")

    try:
        poly = sp.sympify(expression).as_poly(x)

        if poly is None or poly.degree() < 0:
            return None, None

        return int(poly.degree()), float(poly.LC())
    except Exception:
        return None, None


def separateRoots(func, segment, stepFactor):
    if stepFactor <= 0:
        raise ValueError("stepFactor must be positive")

    signChangeSegments = []
    h = segment.len() / stepFactor
    left = segment.left

    while left < segment.right:
        right = min(left + h, segment.right)
        valueAtStart = tryEvaluate(func, left)
        valueAtEnd = tryEvaluate(func, right)
        rootAtNode = valueAtEnd == 0 or (valueAtStart == 0 and left == segment.left)

        if valueAtStart * valueAtEnd < 0 or rootAtNode:
            signChangeSegments.append(Segment(left, right))

        left = right

    return signChangeSegments


def keepsConstantSign(func, segment):
    step = segment.len() / (M1_SAMPLES - 1)
    values = [tryEvaluate(func, segment.left + i * step) for i in range(M1_SAMPLES)]
    finite = [value for value in values if math.isfinite(value)]

    if not finite:
        return False

    return min(finite) >= 0 or max(finite) <= 0


def estimateError(func, derivative, x, segment):
    step = segment.len() / (M1_SAMPLES - 1)
    samples = (
        abs(tryEvaluate(derivative, segment.left + i * step))
        for i in range(M1_SAMPLES)
    )
    finite = [value for value in samples if math.isfinite(value)]
    m1 = min(finite) if finite else None

    if m1 is None or m1 == 0:
        return None

    return abs(tryEvaluate(func, x)) / m1


def approximateByBisection(func, segment, eps):
    stepCounter = 0
    currSegment = segment.copy()
    error = None

    while currSegment.len() > 2 * eps:
        if stepCounter == MAX_STEPS:
            error = (
                f"no convergence in {MAX_STEPS} steps "
                f"(the segment is still [{currSegment.left}, {currSegment.right}])"
            )
            break

        stepCounter = stepCounter + 1
        prevCenter = currSegment.center()
        valueAtStart = tryEvaluate(func, currSegment.left)
        valueAtCenter = tryEvaluate(func, prevCenter)

        if not math.isfinite(valueAtStart) or not math.isfinite(valueAtCenter):
            error = (
                "the function cannot be evaluated: it is not finite at "
                f"x = {prevCenter} or x = {currSegment.left} "
                "(inside the current bracketing segment)"
            )
            break

        if valueAtStart * valueAtCenter <= 0:
            currSegment.right = prevCenter
        else:
            currSegment.left = prevCenter

    if error is not None:
        return ApproximationInfo("Bisection Method", None, stepCounter, error=error)

    approximation = currSegment.center()

    return ApproximationInfo(
        "Bisection Method", approximation, stepCounter, currSegment.len()
    )


def approximateByNewton(func, derivative, segment, eps, fixedSlope=False):
    name = "Modified Newton's Method" if fixedSlope else "Newton's Method"
    stepCounter = 0
    prevX = segment.center()
    currX = prevX
    slope = None
    error = None

    while stepCounter < MAX_STEPS:
        stepCounter = stepCounter + 1

        if slope is None or not fixedSlope:
            slope = tryEvaluate(derivative, prevX)

            if not math.isfinite(slope):
                error = (
                    "f'(x) cannot be evaluated at x = "
                    f"{prevX} (the derivative does not exist or is not finite there)"
                )
                break

            if slope == 0:
                error = f"f'(x) = 0 at x = {prevX}, the Newton step is undefined"
                break

        valueAtPrev = tryEvaluate(func, prevX)

        if not math.isfinite(valueAtPrev):
            error = f"the function is not finite at x = {prevX}"
            break

        currX = prevX - valueAtPrev / slope

        if abs(currX - prevX) < eps:
            break

        prevX = currX
    else:
        error = (
            f"no convergence in {MAX_STEPS} steps "
            f"(the last iterate is x = {prevX}, |f(x)| = {abs(tryEvaluate(func, prevX)):.3e})"
        )

    if error is not None:
        return ApproximationInfo(name, None, stepCounter, error=error)

    return ApproximationInfo(name, currX, stepCounter, abs(currX - prevX))


def approximateBySecant(func, derivative, segment, eps):
    stepCounter = 0
    prevX = segment.left
    currX = segment.right
    error = None
    nextX = currX

    while stepCounter < MAX_STEPS:
        stepCounter = stepCounter + 1
        valueAtCurr = tryEvaluate(func, currX)
        valueAtPrev = tryEvaluate(func, prevX)

        if not math.isfinite(valueAtCurr) or not math.isfinite(valueAtPrev):
            error = (
                "the function cannot be evaluated: it is not finite at "
                f"x = {prevX} or x = {currX}"
            )
            break

        denominator = valueAtCurr - valueAtPrev

        if denominator == 0:
            error = (
                f"equal function values at the last two points: "
                f"f({prevX}) = f({currX}) = {valueAtPrev:.6e}, "
                "the secant slope is zero"
            )
            break

        nextX = currX - valueAtCurr * (currX - prevX) / denominator

        if abs(nextX - currX) < eps:
            break

        prevX = currX
        currX = nextX
    else:
        error = (
            f"no convergence in {MAX_STEPS} steps "
            f"(the last iterate is x = {currX}, |f(x)| = {abs(tryEvaluate(func, currX)):.3e})"
        )

    if error is not None:
        return ApproximationInfo("Secant Method", None, stepCounter, error=error)

    return ApproximationInfo(
        "Secant Method",
        nextX,
        stepCounter,
        estimateError(func, derivative, nextX, segment),
    )
