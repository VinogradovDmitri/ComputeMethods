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
        raise ValueError("Please enter a function")

    x = sp.symbols("x")

    try:
        expr = sp.sympify(expression)
        func = sp.lambdify(x, expr, "math")
    except Exception:
        raise ValueError("Cannot parse the expression")

    if not anyProbeFinite(func):
        raise ValueError("The function cannot be evaluated")

    return func, differentiate(expr, x), differentiate(expr, x, 2)


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
            error = f"no convergence in {MAX_STEPS} steps"
            break

        stepCounter = stepCounter + 1
        prevCenter = currSegment.center()
        valueAtStart = tryEvaluate(func, currSegment.left)
        valueAtCenter = tryEvaluate(func, prevCenter)

        if not math.isfinite(valueAtStart) or not math.isfinite(valueAtCenter):
            error = "the function cannot be evaluated at the current point"
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
                error = "the derivative cannot be evaluated at the current point"
                break

            if slope == 0:
                error = "the derivative is zero at the current point"
                break

        valueAtPrev = tryEvaluate(func, prevX)

        if not math.isfinite(valueAtPrev):
            error = "the function cannot be evaluated at the current point"
            break

        currX = prevX - valueAtPrev / slope

        if abs(currX - prevX) < eps:
            break

        prevX = currX
    else:
        error = f"no convergence in {MAX_STEPS} steps"

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
            error = "the function cannot be evaluated at the current point"
            break

        denominator = valueAtCurr - valueAtPrev

        if denominator == 0:
            error = "equal function values at the last two points"
            break

        nextX = currX - valueAtCurr * (currX - prevX) / denominator

        if abs(nextX - currX) < eps:
            break

        prevX = currX
        currX = nextX
    else:
        error = f"no convergence in {MAX_STEPS} steps"

    if error is not None:
        return ApproximationInfo("Secant Method", None, stepCounter, error=error)

    return ApproximationInfo(
        "Secant Method",
        nextX,
        stepCounter,
        estimateError(func, derivative, nextX, segment),
    )
