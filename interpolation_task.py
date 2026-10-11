import math

from segment import Segment
from methods import (
    tryEvaluate,
    buildFunction,
    polynomialInfo,
)
from interpolation import (
    NODE_TOLERANCE,
    MAX_NODES_WARN,
    buildEquidistantNodes,
    buildRandomNodes,
    sortTableByDistance,
    lagrangeValue,
    lebesgueValue,
    dividedDifferenceTable,
    newtonCoefficients,
    newtonValue,
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
    runUntilMainMenu,
)

DEFAULT_VARIANT = {"count": 20, "left": 0.2, "right": 0.7, "n": 13}

INTERPOLATION_FUNCTIONS = [
    ("f(x) = 2*sin(x)-x/2", "2*sin(x)-x/2", DEFAULT_VARIANT),
    ("f(x) = sin(x)", "sin(x)", None),
    ("f(x) = ln(1+x) - 2x", "log(1+x) - 2*x", None),
    ("f(x) = 2*x^3 - 3*x + 1", "2*x**3 - 3*x + 1", None),
    ("f(x) = x^2 - 2*x + 5", "x**2 - 2*x + 5", None),
]


def chooseInterpolationFunction():
    """name, func, <polyDegree, leadingCoefficient>, defaults"""
    while True:
        section("Choose a function to interpolate")

        for i, (name, _, _) in enumerate(INTERPOLATION_FUNCTIONS, 1):
            print(f"{i}. {name}")

        print(f"{len(INTERPOLATION_FUNCTIONS) + 1}. Custom expression")
        print(dim("0. Back to the main menu"))

        index = reader.value("Function: ", int)

        if index == 0:
            clearScreen()
            return None

        if 1 <= index <= len(INTERPOLATION_FUNCTIONS):
            name, expression, defaults = INTERPOLATION_FUNCTIONS[index - 1]
        elif index == len(INTERPOLATION_FUNCTIONS) + 1:
            expression = reader.line("f(x) = ").strip()
            name = f"f(x) = {expression}"
            defaults = None
        else:
            print(
                red(
                    f"Unknown function: {index} "
                    f"(enter 0..{len(INTERPOLATION_FUNCTIONS) + 1})"
                )
            )
            continue

        try:
            func, _, _ = buildFunction(expression)
        except ValueError as error:
            print(red(f'Cannot interpolate f(x) = "{expression}": {error}'))
            continue

        clearScreen()

        return name, func, polynomialInfo(expression), defaults


def printNodeTable(nodes, values):
    header = f"{'k':>4} {'z_k':>18} {'f(z_k)':>18}"

    section("Node table")
    print(bold(header))
    print("-" * len(header))

    for k, (node, value) in enumerate(zip(nodes, values)):
        print(f"{k:>4} {node:>18.12f} {value:>18.12e}")


def printSortedTable(nodes, values, x, n):
    header = f"{'k':>4} {'x_k':>18} {'f(x_k)':>18} {'|x_k - x|':>18}"

    section(f"Sorted table (by distance to x = {x:.12g})")
    print(bold(header))
    print("-" * len(header))

    for k, (node, value) in enumerate(zip(nodes, values)):
        mark = green("   <- node of Pn") if k <= n else ""
        print(
            f"{k:>4} {node:>18.12f} {value:>18.12e} "
            f"{abs(node - x):>18.12f}{mark}"
        )


def printDividedDifferenceTable(nodes, table):
    width = 15
    order = len(nodes) - 1
    header = f"{'k':>3}{'x_k':>{width}}{'f(x_k)':>{width}}"
    header += "".join(f"{'PP ' + str(j):>{width}}" for j in range(1, order + 1))

    print(bold(header))

    for i in range(len(nodes)):
        row = f"{i:>3}{nodes[i]:>{width}.6f}{table[0][i]:>{width}.6e}"

        for j in range(1, order + 1):
            if j <= i:
                row += f"{table[j][i - j]:>{width}.6e}"
            else:
                row += " " * width

        print(row)


def buildInterpolationTable(func, defaults):
    """<nodes, values>: m+1 pairwise distinct nodes and the values of f"""
    while True:
        if defaults is not None:
            count = reader.optionalValue(
                "m+1 (number of nodes, >= 2)", int, defaults["count"]
            )
            left = reader.optionalValue("A", float, defaults["left"])
            right = reader.optionalValue("B", float, defaults["right"])
        else:
            count = reader.value("m+1 (number of nodes, >= 2): ", int)
            left = reader.value("A: ", float)
            right = reader.value("B: ", float)

        if count < 2:
            print(red(f"m+1 must be >= 2, got {count}"))
            continue

        if count > MAX_NODES_WARN:
            print(
                yellow(
                    f"Warning: m+1 = {count} is large, "
                    "the divided difference table is O(m^2) and may be slow"
                )
            )

        if not math.isfinite(left) or not math.isfinite(right):
            print(red(f"A and B must be finite numbers: got A = {left}, B = {right}"))
            continue

        if right <= left:
            print(red(f"B must be greater than A: got A = {left}, B = {right}"))
            continue

        segment = Segment(left, right)

        print("Node kind: 1 - equidistant, 2 - random, 3 - manual")
        kind = reader.optionalValue("Node kind", int, 1)

        if kind == 3:
            nodes = []
            print(f"Enter {count} pairwise distinct nodes:")

            while len(nodes) < count:
                node = reader.value(f"node {len(nodes)}: ", float)

                if not math.isfinite(node):
                    print(
                        red(
                            f"Node {len(nodes)} must be a finite number, "
                            f"got {node}"
                        )
                    )
                    continue

                clash = next(
                    (other for other in nodes if abs(node - other) <= NODE_TOLERANCE),
                    None,
                )

                if clash is not None:
                    print(
                        red(
                            f"Node {len(nodes)} = {node} is not distinct: "
                            f"it is within {NODE_TOLERANCE} of the earlier "
                            f"node {clash} (|diff| = {abs(node - clash)})"
                        )
                    )
                    continue

                if not segment.left <= node <= segment.right:
                    print(
                        red(
                            f"Node {len(nodes)} = {node} must lie in "
                            f"[{left}, {right}] (the chosen interval)"
                        )
                    )
                    continue

                nodes.append(node)
        elif kind == 1 or kind == 2:
            try:
                if kind == 1:
                    nodes = buildEquidistantNodes(segment, count)
                else:
                    nodes = buildRandomNodes(segment, count)
            except ValueError as error:
                print(
                    red(
                        f"{error}. Try another interval or a smaller m+1 "
                        f"(current: [{left}, {right}], m+1 = {count})"
                    )
                )
                continue
        else:
            print(
                red(f"Unknown node kind: {kind} (enter 1 - equidistant, 2 - random, 3 - manual)")
            )
            continue

        values = [tryEvaluate(func, node) for node in nodes]

        badNodes = [
            node for node, value in zip(nodes, values) if not math.isfinite(value)
        ]

        if badNodes:
            print(
                red(
                    "f(x) cannot be evaluated at these nodes "
                    "(f is not finite there): "
                    + ", ".join(f"{node}" for node in badNodes)
                )
            )
            continue

        printNodeTable(nodes, values)

        return nodes, values


def runInterpolationSolve(func, nodes, values, polyInfo, defaultN):
    m = len(nodes) - 1

    while True:
        x = reader.value("x: ", float)

        if not math.isfinite(x):
            print(red(f"x must be a finite number, got {x}"))
            continue

        fx = tryEvaluate(func, x)

        if not math.isfinite(fx):
            print(
                red(
                    f"f(x) is not finite at x = {x} "
                    "(log/sqrt/division by zero or similar), choose another point"
                )
            )
            continue

        break

    default = defaultN if defaultN is not None and defaultN <= m else None

    while True:
        if default is not None:
            n = reader.optionalValue(f"n (0..{m})", int, default)
        else:
            n = reader.value(f"n (0..{m}): ", int)

        if 0 <= n <= m:
            break

        print(
            red(
                f"Invalid value of n: got {n}, enter n between 0 and {m} "
                f"(the degree of Pn; {m} + 1 nodes are available)"
            )
        )

    sortedNodes, sortedValues = sortTableByDistance(nodes, values, x)
    printSortedTable(sortedNodes, sortedValues, x, n)

    xs = sortedNodes[: n + 1]
    ys = sortedValues[: n + 1]

    print()
    # print(
    #     bold(f"P{n}(x) is built on the nearest nodes: ")
    #     + ", ".join(f"x{k} = {node:.6f}" for k, node in enumerate(xs))
    # )

    pnL, coeffs = lagrangeValue(xs, ys, x)
    coeffSum = sum(coeffs)
    lebesgue = lebesgueValue(coeffs)
    errL = abs(fx - pnL)

    section("Lagrange form")
    print(f"  sum of Lagrange coefficients = {coeffSum:.15f} {dim('(must be ~ 1)')}")
    print(
        f"  Lebesgue function value      = {lebesgue:.15f} "
        f"{dim('(= 1 at a node)')}"
    )
    print(f"  f(x)   = {fx:.16e}")
    print(f"  PnL(x) = {pnL:.16e}")
    print(f"  |f(x) - PnL(x)| = {bold(f'{errL:.6e}')}")

    table = dividedDifferenceTable(xs, ys)
    newtonCoeffs = newtonCoefficients(table)
    pnN = newtonValue(xs, newtonCoeffs, x)
    errN = abs(fx - pnN)

    section("Newton form")
    print(bold(f"Divided difference table up to order n = {n}:"))
    printDividedDifferenceTable(xs, table)
    print()

    for j, coeff in enumerate(newtonCoeffs):
        print(f"  A{j} = {coeff:.10e}")

    print(f"  PnN(x) = {pnN:.16e}")
    print(f"  |f(x) - PnN(x)| = {bold(f'{errN:.6e}')}")

    degree, leading = polyInfo

    if degree is not None and n >= degree:
        section("PP control")
        print(f"  f is a polynomial of degree q = {degree}")
        print(
            f"  f[x0,..,x{degree}] = {table[degree][0]:.10e} "
            f"{dim(f'(the leading coefficient is {leading:.10e})')}"
        )

        if n > degree:
            higher = max(
                abs(value) for j in range(degree + 1, n + 1) for value in table[j]
            )
            print(
                f"  max |PP of order >= {degree + 1}| = {higher:.3e} "
                f"{dim('(must be ~ 0)')}"
            )
        else:
            print(dim("  n = q: the table contains no PP of higher orders"))

    print()
    print(bold(f"|PnL(x) - PnN(x)| = {abs(pnL - pnN):.6e}"))


def runInterpolationTask():
    while True:
        chosen = chooseInterpolationFunction()

        if chosen is None:
            return

        name, func, polyInfo, defaults = chosen
        defaultN = defaults["n"] if defaults is not None else None

        print(bold(f"Chosen function: {name}"))
        nodes, values = buildInterpolationTable(func, defaults)

        runUntilMainMenu(
            lambda: runInterpolationSolve(func, nodes, values, polyInfo, defaultN),
            'Enter "r" to solve for another x and n, '
            "or press Enter to choose another function: ",
        )
