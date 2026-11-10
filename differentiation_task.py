import math

import sympy as sp

from differentiation import (
    firstDerivativeFormula4,
    firstDerivativeO2,
    firstDerivativeO4,
    secondDerivativeO2,
)
from methods import buildFunction, differentiate, tryEvaluate
from reader import reader
from style import (
    clearScreen,
    bold,
    dim,
    red,
    yellow,
    section,
)

DEFAULT_DIGITS = 5
MAX_HALVINGS = 25
SAMPLES_COUNT = 101
WIDTH = 16
ERROR_WIDTH = 18

DIFFERENTIATION_FUNCTIONS = [
    ("f(x) = x^2 - 2*x + 5", "x**2 - 2*x + 5", {"count": 11, "x0": 0.5, "h": 0.1}),
    ("f(x) = 2*x^3 - 3*x + 1", "2*x**3 - 3*x + 1", {"count": 11, "x0": 0.5, "h": 0.1}),
    ("f(x) = exp(4*x)", "exp(4*x)", {"count": 11, "x0": 0.5, "h": 0.05}),
    (
        "f(x) = sin(2*x) - 1.25*x^2 + 0.35",
        "sin(2*x) - 1.25*x**2 + 0.35",
        {"count": 11, "x0": 0.5, "h": 0.1},
    ),
]


def thirdDerivativeFunction(expression):
    """d3(x) as a callable, or None if f''' cannot be built"""
    x = sp.symbols("x")

    try:
        return differentiate(sp.sympify(expression), x, 3)
    except Exception:
        return None


def roundTo(value, digits):
    """Rounds the value to `digits` decimal places like when it is
    written down with that many digits"""
    return float(f"{value:.{digits}f}")


def finiteOrNone(value):
    return value if value is not None and math.isfinite(value) else None


MANUAL_TABLE_NAME = "values y_k entered by hand at the nodes"


def chooseDifferentiationFunction():
    """name, func, d1, d2, d3, defaults; func = None means a manual table
    of values y_k given only at the nodes"""
    while True:
        section("Choose a function to differentiate")

        for i, (name, _, _) in enumerate(DIFFERENTIATION_FUNCTIONS, 1):
            print(f"{i}. {name}")

        print(f"{len(DIFFERENTIATION_FUNCTIONS) + 1}. Custom expression")
        print(
            f"{len(DIFFERENTIATION_FUNCTIONS) + 2}. "
            "Custom table (y_k values given at the nodes)"
        )
        print(dim("0. Back to the main menu"))

        index = reader.value("Function: ", int)

        if index == 0:
            clearScreen()
            return None

        if 1 <= index <= len(DIFFERENTIATION_FUNCTIONS):
            name, expression, defaults = DIFFERENTIATION_FUNCTIONS[index - 1]
        elif index == len(DIFFERENTIATION_FUNCTIONS) + 1:
            expression = reader.line("f(x) = ").strip()
            name = f"f(x) = {expression}"
            defaults = None
        elif index == len(DIFFERENTIATION_FUNCTIONS) + 2:
            clearScreen()

            return (
                MANUAL_TABLE_NAME,
                None,
                None,
                None,
                None,
                None,
            )
        else:
            print(
                red(
                    f"Unknown function: {index} "
                    f"(enter 0..{len(DIFFERENTIATION_FUNCTIONS) + 2})"
                )
            )
            continue

        try:
            func, derivative, secondDerivative = buildFunction(expression)
        except ValueError as error:
            print(red(f'Cannot differentiate f(x) = "{expression}": {error}'))
            continue

        clearScreen()

        return (
            name,
            func,
            derivative,
            secondDerivative,
            thirdDerivativeFunction(expression),
            defaults,
        )


def printNodeTable(nodes, values):
    header = f"{'k':>4} {'x_k':>18} {'f(x_k)':>18}"

    section("Function table")
    print(bold(header))
    print("-" * len(header))

    for k, (node, value) in enumerate(zip(nodes, values)):
        print(f"{k:>4} {node:>18.12f} {value:>18.12e}")


def buildDifferentiationTable(func, defaults):
    """nodes, values, h: m+1 equidistant table points x_k = x0 + k*h, k = 0..m;
    func = None -> the values y_k are entered by hand (f has no analytic form)"""
    while True:
        if defaults is not None:
            count = reader.optionalValue(
                "m+1 (number of points)", int, defaults["count"]
            )
            x0 = reader.optionalValue("x0", float, defaults["x0"])
            h = reader.optionalValue("h", float, defaults["h"])
        else:
            count = reader.value("m+1 (number of points): ", int)
            x0 = reader.value("x0: ", float)
            h = reader.value("h: ", float)

        if count < 2:
            print(red(f"m+1 must be >= 2, got {count}"))
            continue

        if not math.isfinite(x0) or not math.isfinite(h):
            print(red(f"x0 and h must be finite numbers: got x0 = {x0}, h = {h}"))
            continue

        if h <= 0:
            print(red(f"h must be positive, got {h}"))
            continue

        nodes = [x0 + k * h for k in range(count)]
        collided = any(nodes[i + 1] == nodes[i] for i in range(count - 1))

        if collided:
            print(
                red(
                    f"h = {h} is too small at x0 = {x0}: "
                    "neighboring nodes x_k coincide, increase h"
                )
            )
            continue

        if func is None:
            values = []
            print(f"Enter {count} values of y_k at the nodes x_k:")

            while len(values) < count:
                k = len(values)
                y = reader.value(f"y{k} = ", float)

                if not math.isfinite(y):
                    print(red(f"y{k} must be a finite number, got {y}"))
                    continue

                values.append(y)
        else:
            values = [tryEvaluate(func, node) for node in nodes]
            badNodes = [
                node for node, value in zip(nodes, values) if not math.isfinite(value)
            ]

            if badNodes:
                print(
                    red(
                        "f(x) is not finite at these nodes (log/sqrt/division "
                        "by zero or similar): "
                        + ", ".join(f"{node:.6g}" for node in badNodes)
                    )
                )
                continue

        printNodeTable(nodes, values)

        return nodes, values, h


def cell(value):
    if value is None or not math.isfinite(value):
        return f"{'-':>{WIDTH}}"

    return f"{value:>{WIDTH}.7e}"


def errorCell(exact, approx):
    if exact is None or approx is None:
        return f"{'-':>{ERROR_WIDTH}}"

    value = abs(exact - approx)

    if not math.isfinite(value):
        return f"{'-':>{ERROR_WIDTH}}"

    return f"{value:>{ERROR_WIDTH}.3e}"


def printResultsTable(nodes, values, h, derivative, secondDerivative, manual=False):
    """Results in a single table: nodes, function values, exact derivatives
    (analytic f only), approximations by the formulas and the actual
    inaccuracies (absolute values of exact minus approximate)"""
    m = len(nodes) - 1
    approxD1O2 = firstDerivativeO2(values, h)
    approxD1O4 = firstDerivativeO4(values, h)
    approxD2O2 = secondDerivativeO2(values, h)

    exactD1 = [
        finiteOrNone(tryEvaluate(derivative, node)) if derivative is not None else None
        for node in nodes
    ]
    exactD2 = [None] * (m + 1)

    if secondDerivative is not None:
        for k in range(1, m):
            exactD2[k] = finiteOrNone(tryEvaluate(secondDerivative, nodes[k]))

    labelExactD1 = "f'_T"
    labelApproxO2 = "~f' O(h^2)"
    labelInaccO2 = "inaccuracy O(h^2)"
    labelApproxO4 = "~f' O(h^4)"
    labelInaccO4 = "inaccuracy O(h^4)"
    labelExactD2 = "f''_T"
    labelApproxD2 = "~f'' O(h^2)"

    header = f"{'k':>4} {'x_k':>16} {'y_k':>16}"
    header += f" {labelExactD1:>16} {labelApproxO2:>16} {labelInaccO2:>18}"
    header += f" {labelApproxO4:>16} {labelInaccO4:>18}"
    header += f" {labelExactD2:>16} {labelApproxD2:>16} {labelInaccO2:>18}"

    section("Results")
    print(bold(header))
    print("-" * len(header))

    for k in range(m + 1):
        row = f"{k:>4} {nodes[k]:>16.12f} {values[k]:>16.12e}"
        row += f" {cell(exactD1[k])} {cell(approxD1O2[k])}"
        row += f" {errorCell(exactD1[k], approxD1O2[k])}"
        row += f" {cell(approxD1O4[k])} {errorCell(exactD1[k], approxD1O4[k])}"
        row += f" {cell(exactD2[k])} {cell(approxD2O2[k])}"
        row += f" {errorCell(exactD2[k], approxD2O2[k])}"
        print(row)

    print()
    print(
        dim(
            "Legend: f'_T, f''_T are exact (analytic) values of the derivatives; "
            "~f', ~f'' are approximations computed from the table values only; "
            "inaccuracy = |exact - approximate|; '-' = the formula is not "
            "applicable at the point"
        )
    )

    if m < 2:
        print(
            dim(
                "m+1 < 3: no three-point formula fits this table (need at least 3 points)"
            )
        )
        return

    if manual:
        print(
            yellow(
                "the table is defined by its values only (f has no analytic "
                "form here): no exact f'_T and f''_T exist, so the exact "
                "columns and every inaccuracy show '-'"
            )
        )
    elif derivative is None:
        print(
            yellow(
                "f'(x) does not exist or cannot be computed for this expression: "
                "the exact f'_T column shows '-'"
            )
        )

    if secondDerivative is None and not manual:
        print(
            yellow(
                "f''(x) does not exist or cannot be computed for this expression: "
                "the exact f''_T column shows '-'"
            )
        )

    print(
        dim(
            "~f' O(h^2): 3-point one-sided (x, x+h, x+2h) at the first row, "
            "3-point central (x-h, x, x+h) at interior rows, 3-point one-sided "
            "(x-2h, x-h, x) at the last row"
        )
    )
    print(
        dim(
            "~f'' O(h^2): 3-point central second difference (x-h, x, x+h) "
            "at interior rows"
        )
    )

    if m < 4:
        print(
            yellow(
                "~f' O(h^4) is not computed: "
                "the 5-point formulas need at least 5 points in the table"
            )
        )
        return

    print(
        dim(
            "~f' O(h^4): 5-point one-sided (x..x+4h) at the first two rows, "
            "5-point central (x-2h..x+2h) at interior rows, 5-point one-sided "
            "(x-4h..x) at the last two rows"
        )
    )


def sampleMaxAbs(func, a, b):
    """max |f| on [a, b] by uniform sampling, None if no finite sample"""
    if func is None or not a < b:
        return None

    step = (b - a) / (SAMPLES_COUNT - 1)
    samples = (abs(tryEvaluate(func, a + i * step)) for i in range(SAMPLES_COUNT))
    finite = [value for value in samples if math.isfinite(value)]

    return max(finite) if finite else None


def runOptimalStepExperiment(func, derivative, thirdDerivative, defaultX, defaultH):
    """Lecture experiment: f'(x) by formula (4) from rounded values of f,
    h is halved until the actual error starts to grow; the empirical
    h_opt is compared with the theoretical one"""
    section("Optimal h experiment (~f', one-sided 3-point O(h^2) formula)")
    print(
        "Every value of f is rounded to D decimal places "
        "(the input error is eps = 0.5 * 10^-D), then h halves "
        "until the actual error (inaccuracy) starts to grow."
    )

    while True:
        x = reader.optionalValue("x (point of differentiation)", float, defaultX)

        if math.isfinite(x):
            break

        print(red(f"x must be a finite number, got {x}"))

    while True:
        h = reader.optionalValue("initial h", float, defaultH)

        if math.isfinite(h) and h > 0:
            break

        print(red(f"h must be a positive finite number, got {h}"))

    while True:
        digits = reader.optionalValue("D (rounding digits)", int, DEFAULT_DIGITS)

        if 0 <= digits <= 15:
            break

        print(red(f"D must be between 0 and 15, got {digits}"))

    exact = finiteOrNone(tryEvaluate(derivative, x)) if derivative is not None else None

    if exact is None:
        print(red(f"f'(x) is not finite at x = {x}: the experiment is impossible"))
        return

    eps = 0.5 * 10 ** (-digits)
    print()
    print(f"f'_T(x) = {exact:.12e}, eps = {eps:.2e}")
    print()

    rows = []
    stopReason = None
    current = h
    prevError = None

    while len(rows) < MAX_HALVINGS:
        samples = (x, x + current, x + 2 * current)
        values = [tryEvaluate(func, point) for point in samples]
        badPoints = [
            point for point, value in zip(samples, values) if not math.isfinite(value)
        ]

        if badPoints:
            stopReason = "f is not finite at " + ", ".join(
                f"{point:.6g}" for point in badPoints
            )
            break

        rounded = [roundTo(value, digits) for value in values]
        approx = firstDerivativeFormula4(current, rounded[0], rounded[1], rounded[2])
        error = abs(exact - approx)
        rows.append((current, rounded, approx, error))

        if prevError is not None and error > prevError:
            stopReason = "the error started to grow"
            break

        prevError = error
        current = current / 2

        if current == 0:
            stopReason = "h underflowed to 0"
            break

    if not rows:
        print(red("No approximation was computed: the experiment is impossible here"))
        return

    labelApprox = "~f' O(h^2)"
    header = (
        f"{'h':>10} {'f(x) table':>16} {'f(x+h) table':>16} {'f(x+2h) table':>16}"
        f" {labelApprox:>16} {'inaccuracy':>12}"
    )
    print(
        dim(
            'the three "table" values are the rounded values of f and the '
            "only numbers ~f' is computed from; inaccuracy = |f'_T(x) - ~f'|"
        )
    )
    print(bold(header))
    print("-" * len(header))

    for step, rounded, approx, error in rows:
        row = f"{step:>10.7g} {rounded[0]:>16.6e} {rounded[1]:>16.6e}"
        row += f" {rounded[2]:>16.6e} {approx:>16.6e} {error:>12.3e}"
        print(row)

    if stopReason is not None:
        print(dim(f"Stopped: {stopReason}"))
    else:
        print(
            dim(
                f"Stopped: the error kept decreasing for {MAX_HALVINGS} halvings, "
                "try a smaller initial h"
            )
        )

    best = min(rows, key=lambda row: row[3])
    print()
    print(f"Empirical h_opt = {best[0]:.7g} (actual error {best[3]:.3e})")

    m3 = sampleMaxAbs(thirdDerivative, x, x + 2 * h)

    if m3 is None:
        print(
            dim(
                "M3 = max|f'''| is not available for this f: "
                "the theoretical part is skipped"
            )
        )
        return

    if m3 == 0:
        print(
            dim(
                "M3 = max|f'''| on [x, x+2h] = 0: the truncation error term "
                "is absent, the smaller h the smaller the truncation error "
                "of ~f' (until the rounding error takes over)"
            )
        )
        return

    hOptTheor = (6 * eps / m3) ** (1.0 / 3.0)
    minBound = 4 * eps / hOptTheor + m3 / 3 * hOptTheor**2
    print()
    print(f"epsilon = {eps:.2e}, M3 = max|f'''| on [x, x+2h] = {m3:.4e}")
    print(
        f"Theoretical h_opt = (6*eps/M3)^(1/3) = {hOptTheor:.7g}, "
        f"|R(eps)| <= phi(h_opt) = {minBound:.3e}"
    )


def readAnalysisAction(allowExperiment=True):
    while True:
        print()
        print("What next:")

        if allowExperiment:
            print("1. Run the optimal h experiment for ~f'")
        else:
            print(
                dim(
                    "1. Optimal h experiment "
                    "(impossible: f is known at the table nodes only)"
                )
            )

        print("2. Build a new table for the same function")
        print("3. Choose another function")
        print(dim("0. Back to the main menu"))

        action = reader.value("Action: ", int)

        if 0 <= action <= 3:
            if action != 1 or allowExperiment:
                return action

            print(
                red(
                    "the optimal h experiment evaluates f at x, x+h, x+2h "
                    "outside the table nodes: an analytic f is required"
                )
            )
            continue

        print(red(f"Unknown action: {action} (enter 0..3)"))


def runDifferentiationTask():
    while True:
        chosen = chooseDifferentiationFunction()

        if chosen is None:
            return

        name, func, derivative, secondDerivative, thirdDerivative, defaults = chosen
        manual = func is None

        while True:
            subject = "table" if manual else "function"
            print(bold(f"Chosen {subject}: {name}"))
            nodes, values, h = buildDifferentiationTable(func, defaults)
            printResultsTable(nodes, values, h, derivative, secondDerivative, manual)

            while True:
                action = readAnalysisAction(allowExperiment=not manual)

                if action == 1:
                    runOptimalStepExperiment(
                        func, derivative, thirdDerivative, nodes[0], h
                    )
                    continue

                if action == 0:
                    clearScreen()
                    return

                break

            if action == 3:
                clearScreen()
                break
