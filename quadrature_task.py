import math

import sympy as sp

from methods import tryEvaluate
from quadrature import (
    X,
    buildVandermonde,
    exactIntegral,
    quadratureValue,
    rungeIntegrate,
    solveQuadratureSystem,
)
from reader import reader
from style import (
    clearScreen,
    bold,
    dim,
    red,
    green,
    yellow,
    section,
)

NODE_TOLERANCE = 1e-12
MAX_NODES_WARN = 12

DEFAULT_VARIANT = {"a": 0.0, "b": 1.0, "N": 3}

QUADRATURE_WEIGHTS = [
    ("p(x) = exp(x)", "exp(x)", DEFAULT_VARIANT),
    ("p(x) = 1", "1", None),
    ("p(x) = x^2", "x**2", None),
    ("p(x) = 1/(1+x^2)", "1/(1+x**2)", None),
]

QUADRATURE_FUNCTIONS = [
    ("f(x) = sin(x)", "sin(x)"),
    ("f(x) = cos(x)", "cos(x)"),
    ("f(x) = exp(x)", "exp(x)"),
    ("f(x) = x^2 + 2*x + 1  (polynomial of degree 2)", "x**2 + 2*x + 1"),
    ("f(x) = x^3 - x + 2  (polynomial of degree 3)", "x**3 - x + 2"),
    ("f(x) = x^4 + x^2 + 1  (polynomial of degree 4)", "x**4 + x**2 + 1"),
]


def prettyExpression(expr):
    return str(expr).replace("**", "^")


def parseExpression(expression, label):
    if expression == "":
        raise ValueError(f"please enter {label}, e.g. x^2 + exp(x)")

    try:
        expr = sp.sympify(expression)
    except Exception as error:
        raise ValueError(f'cannot parse "{expression}": {error}')

    if not isinstance(expr, sp.Basic):
        raise ValueError("the expression must be a scalar expression in x")

    extraSymbols = expr.free_symbols - {X}

    if extraSymbols:
        names = ", ".join(sorted(str(symbol) for symbol in extraSymbols))
        raise ValueError(f"only the variable x is allowed, found: {names}")

    try:
        func = sp.lambdify(X, expr, "math")
    except Exception as error:
        raise ValueError(f"cannot build a numeric function: {error}")

    return expr, func


def chooseWeight():
    """name, sympy expression, callable, defaults"""
    while True:
        section("Choose a weight function")

        for i, (name, _, _) in enumerate(QUADRATURE_WEIGHTS, 1):
            print(f"{i}. {name}")

        print(f"{len(QUADRATURE_WEIGHTS) + 1}. Custom weight expression")
        print(dim("0. Back to the main menu"))

        index = reader.optionalValue("Weight", int, 1)

        if index == 0:
            clearScreen()
            return None

        if 1 <= index <= len(QUADRATURE_WEIGHTS):
            name, expression, defaults = QUADRATURE_WEIGHTS[index - 1]
        elif index == len(QUADRATURE_WEIGHTS) + 1:
            print(dim("Use sympy syntax, e.g. exp(x), 1/(1+x^2), sqrt(x)"))
            expression = reader.line("ρ(x) = ").strip()
            name = f"p(x) = {expression}"
            defaults = None
        else:
            print(
                red(
                    f"Unknown weight: {index} "
                    f"(enter 0..{len(QUADRATURE_WEIGHTS) + 1})"
                )
            )
            continue

        try:
            expr, func = parseExpression(expression, "the weight function")
        except ValueError as error:
            print(red(f'Cannot use weight ρ(x) = "{expression}": {error}'))
            continue

        clearScreen()

        return name, expr, func, defaults


def defaultNode(a, b, count, index):
    if count == 1:
        return (a + b) / 2.0

    return a + index * (b - a) / (count - 1)


def weightIsFiniteOnInterval(func, a, b):
    step = (b - a) / 100.0

    return any(math.isfinite(tryEvaluate(func, a + i * step)) for i in range(101))


def readNodes(a, b, count, useDefaults):
    nodes = []
    print(f"Enter {count} node(s) x_k:")

    while len(nodes) < count:
        k = len(nodes)

        if useDefaults:
            node = reader.optionalValue(f"x{k}", float, defaultNode(a, b, count, k))
        else:
            node = reader.value(f"x{k}: ", float)

        if not math.isfinite(node):
            print(red(f"Node x{k} must be a finite number, got {node}"))
            continue

        duplicate = next((other for other in nodes if node == other), None)

        if duplicate is not None:
            print(
                red(
                    f"Node x{k} = {node} duplicates the earlier node {duplicate}: "
                    "the nodes must be pairwise distinct"
                )
            )
            continue

        close = next(
            (other for other in nodes if 0 < abs(node - other) <= NODE_TOLERANCE),
            None,
        )

        if close is not None:
            print(
                yellow(
                    f"Warning: x{k} = {node} is very close to {close} "
                    f"(|diff| <= {NODE_TOLERANCE}); the Vandermonde system "
                    "may be ill-conditioned"
                )
            )

        nodes.append(node)

    outside = [node for node in nodes if node < a or node > b]

    if outside:
        print(
            yellow(
                "Note: these nodes lie outside [a, b]: "
                + ", ".join(f"{node:.12g}" for node in outside)
                + ". The formula is still built, but the usual quadrature "
                "interpretation assumes nodes inside the interval"
            )
        )

    return nodes


def readParameters(weightName, weightFunc, defaults):
    """a, b, N, nodes; defaults are used only for the variant weight"""
    section("Integration interval and nodes")

    while True:
        if defaults is not None:
            a = reader.optionalValue("a", float, defaults["a"])
            b = reader.optionalValue("b", float, defaults["b"])
        else:
            a = reader.value("a: ", float)
            b = reader.value("b: ", float)

        if not math.isfinite(a) or not math.isfinite(b):
            print(red(f"a and b must be finite numbers: got a = {a}, b = {b}"))
            continue

        if b <= a:
            print(red(f"b must be greater than a: got a = {a}, b = {b}"))
            continue

        if not weightIsFiniteOnInterval(weightFunc, a, b):
            print(
                red(
                    f"{weightName} is not finite anywhere on [{a}, {b}]: "
                    "choose another interval or weight"
                )
            )
            continue

        if defaults is not None:
            count = reader.optionalValue(
                "N (number of nodes, >= 1)", int, defaults["N"]
            )
        else:
            count = reader.value("N (number of nodes, >= 1): ", int)

        if count < 1:
            print(red(f"N must be >= 1, got {count}"))
            continue

        if count > MAX_NODES_WARN:
            print(
                yellow(
                    f"Warning: N = {count} is large, the Vandermonde matrix "
                    "may be ill-conditioned and slow to solve"
                )
            )

        nodes = readNodes(a, b, count, defaults is not None)

        return a, b, count, nodes


def printParametersSummary(weightName, a, b, count, nodes):
    print(bold(f"Chosen weight: {weightName}"))
    print(f"Interval: [{a}, {b}]")
    print(f"Number of nodes: N = {count}")
    print("Nodes: " + ", ".join(f"x{k} = {node:.12g}" for k, node in enumerate(nodes)))


def computeMoments(weightExpr, weightFunc, a, b, count):
    """moments mu_j, their sources ('exact' / 'Runge') and Runge estimates"""
    moments = []
    sources = []
    estimates = []

    for j in range(count):
        exact = exactIntegral(weightExpr * X**j, a, b)

        if exact is not None:
            moments.append(exact)
            sources.append("exact")
            estimates.append(None)
            continue

        try:
            result = rungeIntegrate(lambda t, power=j: weightFunc(t) * t**power, a, b)
        except ValueError as error:
            raise ValueError(f"cannot compute moment μ_{j}: {error}")

        if not result.converged:
            print(
                yellow(
                    f"Runge rule did not reach the tolerance for μ_{j}: "
                    f"the last estimate is {result.errorEstimate:.3e}"
                )
            )

        moments.append(result.value)
        sources.append("Runge")
        estimates.append(result.errorEstimate)

    return moments, sources, estimates


def printMoments(moments, sources, estimates):
    header = f"{'j':>4} {'μ_j':>18} {'source':>10} {'error estimate':>18}"

    section("Weight moments")
    print(dim("μ_j = ∫_a^b ρ(x) x^j dx"))
    print(bold(header))
    print("-" * len(header))

    for j, moment in enumerate(moments):
        estimate = estimates[j]
        estimateText = "-" if estimate is None else f"{estimate:.3e}"
        print(f"{j:>4} {moment:>18.9e} {sources[j]:>10} {estimateText:>18}")

    if any(source == "Runge" for source in sources):
        print(
            dim(
                "Runge estimates are |I_{h/2} - I_h| / 15; "
                "the corresponding moments are numerical"
            )
        )


def printSystem(matrix, moments):
    count = len(moments)
    header = f"{'j':>4} {'μ_j':>18}"
    header += "".join(f" {'x_' + str(k) + '^j':>18}" for k in range(count))

    section("System for the coefficients")
    print(dim("V[j][k] = x_k^j; the system is Σ_k A_k x_k^j = μ_j, j = 0..N-1"))
    print(bold(header))
    print("-" * len(header))

    for j in range(count):
        row = f"{j:>4} {moments[j]:>18.9e}"

        for k in range(count):
            row += f" {matrix[j][k]:>18.9e}"

        print(row)


def printCoefficients(nodes, coefficients):
    header = f"{'k':>4} {'x_k':>18} {'A_k':>18} {'|A_k|':>18}"

    section("Quadrature coefficients")
    print(bold(header))
    print("-" * len(header))

    for k, (node, coefficient) in enumerate(zip(nodes, coefficients)):
        print(
            f"{k:>4} {node:>18.12f} {coefficient:>18.12e} "
            f"{abs(coefficient):>18.12e}"
        )

    sumAbs = math.fsum(abs(coefficient) for coefficient in coefficients)
    sumAll = math.fsum(coefficients)
    print()
    print(bold(f"Σ |A_k| = {sumAbs:.12e}"))
    print(f"Σ A_k   = {sumAll:.12e} {dim('(must equal μ_0)')}")


def polynomialExpression(count):
    if count == 1:
        return sp.Integer(1)

    return X ** (count - 1) + 2 * X + 1


def checkExactness(weightExpr, weightFunc, a, b, nodes, coefficients):
    count = len(nodes)
    pExpr = polynomialExpression(count)
    pFunc = sp.lambdify(X, pExpr, "math")

    exact = exactIntegral(weightExpr * pExpr, a, b)
    source = "exact (sympy)"
    estimate = None

    if exact is None:
        try:
            result = rungeIntegrate(lambda t: weightFunc(t) * pFunc(t), a, b)
        except ValueError as error:
            print(yellow(f"Exactness check skipped: {error}"))
            return

        exact = result.value
        source = "Runge"
        estimate = result.errorEstimate

        if not result.converged:
            print(
                yellow(
                    "Runge rule did not reach the tolerance for the exactness "
                    f"check: the last estimate is {result.errorEstimate:.3e}"
                )
            )

    approx = quadratureValue(coefficients, [pFunc(node) for node in nodes])
    difference = abs(exact - approx)

    section("Exactness check on a polynomial of degree N-1")
    print(f"p(x) = {prettyExpression(pExpr)}")
    print(f"degree = {count - 1}")
    print(f"∫_a^b ρ(x) p(x) dx = {exact:.12e} ({source})")

    if estimate is not None:
        print(dim(f"Runge estimate = {estimate:.3e}"))

    print(f"Σ A_k p(x_k)       = {approx:.12e}")
    print(f"|difference|       = {difference:.3e}")

    threshold = 1e-7 * max(1.0, abs(exact))

    if difference <= threshold:
        print(
            green(
                "OK: the formula is exact on this polynomial " "(up to numerical error)"
            )
        )
    else:
        print(
            yellow(
                "The difference is larger than expected: the Vandermonde "
                "system may be ill-conditioned"
            )
        )


def chooseIntegrand(count, nodes):
    """name, sympy expression, callable, values at the nodes; None = back"""
    if count == 1:
        generatedExpr = "1"
        generatedName = "f(x) = 1  (polynomial of degree N-1 = 0)"
    else:
        generatedExpr = f"x**{count - 1} + 2*x + 1"
        generatedName = (
            f"f(x) = {prettyExpression(sp.sympify(generatedExpr))}  "
            f"(polynomial of degree N-1 = {count - 1})"
        )

    while True:
        section("Choose a function to integrate")

        for i, (name, _) in enumerate(QUADRATURE_FUNCTIONS, 1):
            print(f"{i}. {name}")

        print(f"{len(QUADRATURE_FUNCTIONS) + 1}. {generatedName}")
        print(f"{len(QUADRATURE_FUNCTIONS) + 2}. Custom expression")
        print(dim("0. Back to new parameters"))

        index = reader.optionalValue("Function", int, 1)

        if index == 0:
            clearScreen()
            return None

        if 1 <= index <= len(QUADRATURE_FUNCTIONS):
            name, expression = QUADRATURE_FUNCTIONS[index - 1]
        elif index == len(QUADRATURE_FUNCTIONS) + 1:
            name, expression = generatedName, generatedExpr
        elif index == len(QUADRATURE_FUNCTIONS) + 2:
            print(dim("Use sympy syntax, e.g. sin(x), exp(x), x^3 - x + 2"))
            expression = reader.line("f(x) = ").strip()
            name = f"f(x) = {expression}"
        else:
            print(
                red(
                    f"Unknown function: {index} "
                    f"(enter 0..{len(QUADRATURE_FUNCTIONS) + 2})"
                )
            )
            continue

        try:
            expr, func = parseExpression(expression, "the function f(x)")
        except ValueError as error:
            print(red(f'Cannot use f(x) = "{expression}": {error}'))
            continue

        values = [tryEvaluate(func, node) for node in nodes]
        badNodes = [
            node for node, value in zip(nodes, values) if not math.isfinite(value)
        ]

        if badNodes:
            print(
                red(
                    "f(x) is not finite at these nodes: "
                    + ", ".join(f"{node:.12g}" for node in badNodes)
                    + ". Choose another function"
                )
            )
            continue

        clearScreen()
        print(bold(f"Chosen function: {name}"))

        return name, expr, func, values


def integrateAndPrint(weightExpr, weightFunc, a, b, nodes, coefficients, integrand):
    _, fExpr, fFunc, values = integrand
    terms = [a_k * value for a_k, value in zip(coefficients, values)]
    approx = quadratureValue(coefficients, values)

    header = f"{'k':>4} {'x_k':>18} {'f(x_k)':>18} {'A_k':>18} " f"{'A_k f(x_k)':>20}"

    section("Values at the nodes")
    print(bold(header))
    print("-" * len(header))

    for k, (node, value, coefficient, term) in enumerate(
        zip(nodes, values, coefficients, terms)
    ):
        print(
            f"{k:>4} {node:>18.12f} {value:>18.12e} "
            f"{coefficient:>18.12e} {term:>20.12e}"
        )

    print()
    print(bold(f"~I = Σ A_k f(x_k) = {approx:.12e}"))

    section("Reference value and inaccuracy")
    exact = exactIntegral(weightExpr * fExpr, a, b)
    estimate = None

    if exact is not None:
        reference = exact
        source = "exact (sympy)"
        referenceLabel = "I_exact"
        print(f"I_exact = {reference:.12e} ({source})")
    else:
        try:
            result = rungeIntegrate(lambda t: weightFunc(t) * fFunc(t), a, b)
        except ValueError as error:
            print(red(f"Cannot compute a reference value: {error}"))
            return

        reference = result.value
        estimate = result.errorEstimate
        source = "Runge"
        referenceLabel = "I_ref"

        print(
            bold(
                "The analytic integral was not found: "
                "using the Runge rule for the reference value"
            )
        )
        print(f"I_h     = {result.previous:.12e}")
        print(f"I_{{h/2}} = {result.value:.12e}")
        print("Runge estimate = |I_{h/2} - I_h| / 15 = " f"{result.errorEstimate:.3e}")
        print(f"n = {result.intervals}")

        if not result.converged:
            print(
                yellow(
                    "Runge rule did not reach the tolerance: "
                    "the reference value may be inaccurate"
                )
            )

        print(f"I_ref = {reference:.12e} ({source})")

    absoluteError = abs(reference - approx)

    print(f"absolute inaccuracy = |{referenceLabel} - ~I| = " f"{absoluteError:.6e}")

    if reference == 0:
        print(f"relative inaccuracy = undefined ({referenceLabel} = 0)")
    else:
        print(
            f"relative inaccuracy = |{referenceLabel} - ~I| / "
            f"|{referenceLabel}| = {absoluteError / abs(reference):.6e}"
        )

    if estimate is not None:
        print(
            dim(f"note: I_ref is numerical, its Runge estimate is " f"{estimate:.3e}")
        )


def readAfterResult():
    while True:
        print()
        print("What next:")
        print("1. Choose another f(x) for the same formula")
        print("2. Enter new parameters (weight, a, b, N, nodes)")
        print(dim("0. Back to the main menu"))

        action = reader.value("Action: ", int)

        if 0 <= action <= 2:
            return action

        print(red(f"Unknown action: {action} (enter 0..2)"))


def runQuadratureTask():
    while True:
        chosenWeight = chooseWeight()

        if chosenWeight is None:
            return

        weightName, weightExpr, weightFunc, defaults = chosenWeight
        a, b, count, nodes = readParameters(weightName, weightFunc, defaults)
        printParametersSummary(weightName, a, b, count, nodes)

        try:
            moments, sources, estimates = computeMoments(
                weightExpr, weightFunc, a, b, count
            )
        except ValueError as error:
            print(red(str(error)))
            continue

        printMoments(moments, sources, estimates)

        matrix = buildVandermonde(nodes)
        printSystem(matrix, moments)

        try:
            coefficients, residual = solveQuadratureSystem(nodes, moments)
        except ValueError as error:
            print(red(f"Cannot build the quadrature formula: {error}"))
            continue

        printCoefficients(nodes, coefficients)

        residualLimit = 1e-8 * max(1.0, max(abs(moment) for moment in moments))

        if residual > residualLimit:
            print(
                yellow(
                    f"Warning: the system residual is {residual:.3e}: "
                    "the Vandermonde system may be ill-conditioned"
                )
            )

        checkExactness(weightExpr, weightFunc, a, b, nodes, coefficients)
        print(green("The quadrature formula is built."))

        while True:
            integrand = chooseIntegrand(count, nodes)

            if integrand is None:
                break

            integrateAndPrint(
                weightExpr,
                weightFunc,
                a,
                b,
                nodes,
                coefficients,
                integrand,
            )

            action = readAfterResult()

            if action == 1:
                continue

            if action == 2:
                clearScreen()
                break

            clearScreen()
            return
