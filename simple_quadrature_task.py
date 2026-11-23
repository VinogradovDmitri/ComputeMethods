import math

import sympy as sp

from methods import anyProbeFinite, tryEvaluate
from quadrature import X, exactIntegral, rungeIntegrate
from reader import reader
from simple_quadrature import SIMPLE_FORMULAS, algebraicDegreeOfAccuracy
from style import clearScreen, bold, dim, red, yellow, section

DEFAULT_FUNCTION_INDEX = 5

# name, sympy expression, sympy antiderivative
# exact integrals on [0, 1]: 1, 3/2, 7/3, 7/4, 1-cos(1),
# sin(1), e-1, 23/15
TEST_FUNCTIONS = [
    ("f0(x) = 1", sp.Integer(1), X),
    ("f1(x) = x + 1", X + 1, X**2 / 2 + X),
    ("f2(x) = x^2 + 2*x + 1", X**2 + 2 * X + 1, X**3 / 3 + X**2 + X),
    ("f3(x) = x^3 - x + 2", X**3 - X + 2, X**4 / 4 - X**2 / 2 + 2 * X),
    ("f4(x) = sin(x)", sp.sin(X), -sp.cos(X)),
    ("f5(x) = cos(x)", sp.cos(X), sp.sin(X)),
    ("f6(x) = exp(x)", sp.exp(X), sp.exp(X)),
    ("f7(x) = x^4 + x^2 + 1", X**4 + X**2 + 1, X**5 / 5 + X**3 / 3 + X),
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


def chooseFunction():
    """name, sympy expression, callable, antiderivative or None; None = back"""
    while True:
        section("Choose a function to integrate")

        for i, (name, _, _) in enumerate(TEST_FUNCTIONS, 1):
            print(f"{i}. {name}")

        print(f"{len(TEST_FUNCTIONS) + 1}. Custom expression")
        print(dim("0. Back to the main menu"))

        index = reader.optionalValue("Function", int, DEFAULT_FUNCTION_INDEX)

        if index == 0:
            clearScreen()
            return None

        if 1 <= index <= len(TEST_FUNCTIONS):
            name, expr, antiderivative = TEST_FUNCTIONS[index - 1]
            func = sp.lambdify(X, expr, "math")
        elif index == len(TEST_FUNCTIONS) + 1:
            print(dim("Use sympy syntax, e.g. sin(x), exp(x), x^3 - x + 2"))
            expression = reader.line("f(x) = ").strip()

            try:
                expr, func = parseExpression(expression, "the function f(x)")
            except ValueError as error:
                print(red(f'Cannot use f(x) = "{expression}": {error}'))
                continue

            if not anyProbeFinite(func):
                print(
                    red(
                        f'f(x) = "{expression}" cannot be evaluated '
                        "at any probe point"
                    )
                )
                continue

            name = f"f(x) = {prettyExpression(expr)}"
            antiderivative = None
        else:
            print(
                red(
                    f"Unknown function: {index} "
                    f"(enter 0..{len(TEST_FUNCTIONS) + 1})"
                )
            )
            continue

        clearScreen()
        return name, expr, func, antiderivative


def functionIsFiniteOnInterval(func, a, b):
    step = (b - a) / 100.0

    return all(math.isfinite(tryEvaluate(func, a + i * step)) for i in range(101))


def readInterval(func):
    """a, b; None means back to the function menu"""
    section("Integration interval")

    while True:
        a = reader.optionalValue("a", float, 0.0)
        b = reader.optionalValue("b", float, 1.0)

        if not math.isfinite(a) or not math.isfinite(b):
            print(red(f"a and b must be finite numbers: got a = {a}, b = {b}"))
            continue

        if b <= a:
            print(red(f"b must be greater than a: got a = {a}, b = {b}"))
            continue

        if not functionIsFiniteOnInterval(func, a, b):
            print(red(f"f(x) is not finite everywhere on [{a}, {b}]"))

            while True:
                action = reader.value(
                    "Action (0 = choose another function, " "1 = another interval): ",
                    int,
                )

                if action in (0, 1):
                    break

                print(red(f"Unknown action: {action} (enter 0 or 1)"))

            if action == 0:
                return None

            continue

        return a, b


def evaluateAntiderivative(antiderivative, a, b):
    try:
        value = sp.N(antiderivative.subs(X, b) - antiderivative.subs(X, a), 30)
        real = float(sp.re(value))
        imaginary = float(sp.im(value))
    except Exception as error:
        raise ValueError(f"cannot evaluate the antiderivative: {error}")

    if not math.isfinite(real) or not math.isfinite(imaginary):
        raise ValueError("the exact integral is not finite")

    if abs(imaginary) > 1e-10 * max(1.0, abs(real)):
        raise ValueError("the exact integral is not real")

    return real


def exactValue(expr, func, antiderivative, a, b):
    """value, source, error estimate, Runge result or None"""
    if antiderivative is not None:
        value = evaluateAntiderivative(antiderivative, a, b)
        return value, "exact (Newton-Leibniz)", None, None

    exact = exactIntegral(expr, a, b)

    if exact is not None:
        return exact, "exact (sympy)", None, None

    try:
        result = rungeIntegrate(func, a, b)
    except ValueError as error:
        raise ValueError(f"cannot compute a reference value: {error}")

    return result.value, "Runge reference", result.errorEstimate, result


def computeAndPrint(name, expr, func, antiderivative, a, b):
    print(bold(f"Chosen function: {name}"))
    print(f"Interval: [{a:.12g}, {b:.12g}]")
    print("Weight: ρ(x) = 1")

    try:
        exact, source, _, runge = exactValue(expr, func, antiderivative, a, b)
    except ValueError as error:
        print(red(f"Cannot compute the exact value: {error}"))
        return

    if source == "exact (Newton-Leibniz)":
        section("Exact value (Newton-Leibniz)")
    elif source.startswith("exact"):
        section("Exact value (sympy)")
    else:
        section("Reference value (Runge rule)")

    if antiderivative is not None:
        print(f"F(x) = {prettyExpression(antiderivative)}")
        print(f"I_exact = F({b:.12g}) - F({a:.12g}) = {exact:.12e}")
    else:
        label = "I_exact" if source.startswith("exact") else "I_ref"
        print(f"{label} = {exact:.12e}")

    if runge is not None:
        print(f"I_h     = {runge.previous:.12e}")
        print(f"I_{{h/2}} = {runge.value:.12e}")
        print("Runge estimate = |I_{h/2} - I_h| / 15 = " f"{runge.errorEstimate:.3e}")
        print(f"n = {runge.intervals}")

        if not runge.converged:
            print(
                yellow(
                    "Runge rule did not reach the tolerance: "
                    "the reference value may be inaccurate"
                )
            )

    referenceLabel = "I_exact" if source.startswith("exact") else "I_ref"
    errorHeader = f"R = {referenceLabel} - ~I"

    section("Simple quadrature formulas")

    header = (
        f"{'formula':<20} {'approx':>20} {errorHeader:>20} "
        f"{'|R|':>14} {'rel.':>12} {'AST':>4}"
    )
    print(bold(header))
    print("-" * len(header))

    for formulaName, formula in SIMPLE_FORMULAS:
        try:
            approx = formula(func, a, b)
        except Exception as error:
            print(red(f"{formulaName}: cannot evaluate: {error}"))
            continue

        if not math.isfinite(approx):
            print(red(f"{formulaName}: the result is not finite"))
            continue

        error = exact - approx
        absError = abs(error)
        relativeText = "-" if exact == 0 else f"{absError / abs(exact):.6e}"
        ast = algebraicDegreeOfAccuracy(formula, a, b)

        print(
            f"{formulaName:<20} {approx:>20.12e} {error:>20.12e} "
            f"{absError:>14.6e} {relativeText:>12} {ast:>4}"
        )


def readAfterResult():
    while True:
        print()
        print("What next:")
        print("1. Choose another f(x) for the same interval")
        print("2. Enter new a and b for the same f(x)")
        print(dim("0. Back to the main menu"))

        action = reader.value("Action: ", int)

        if 0 <= action <= 2:
            return action

        print(red(f"Unknown action: {action} (enter 0..2)"))


def runSimpleQuadratureTask():
    while True:
        chosen = chooseFunction()

        if chosen is None:
            return

        name, expr, func, antiderivative = chosen
        interval = readInterval(func)

        if interval is None:
            continue

        a, b = interval

        while True:
            computeAndPrint(name, expr, func, antiderivative, a, b)
            action = readAfterResult()

            if action == 0:
                clearScreen()
                return

            if action == 1:
                chosen = chooseFunction()

                if chosen is None:
                    clearScreen()
                    return

                name, expr, func, antiderivative = chosen

                if not functionIsFiniteOnInterval(func, a, b):
                    print(red(f"{name} is not finite on " f"[{a:.12g}, {b:.12g}]"))
                    interval = readInterval(func)

                    if interval is None:
                        break

                    a, b = interval

                continue

            if action == 2:
                interval = readInterval(func)

                if interval is None:
                    break

                a, b = interval
                continue
