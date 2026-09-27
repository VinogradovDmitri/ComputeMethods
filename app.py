import math

from segment import Segment
from methods import (
    tryEvaluate,
    buildFunction,
    separateRoots,
    approximateByBisection,
    approximateByNewton,
    approximateBySecant,
    keepsConstantSign,
)
from reader import Reader

A = -15.0
B = 5.0
EPS = 1e-7

MATERIALS = [
    ("Cork", 0.25),
    ("Bamboo", 0.4),
    ("Pine (white)", 0.5),
    ("Cedar", 0.55),
    ("Oak", 0.7),
    ("Beech", 0.75),
    ("Mahogany", 0.8),
    ("Teak wood", 0.85),
    ("Paraffin", 0.9),
    ("Ice/Polyethylene", 0.92),
    ("Beeswax", 0.95),
]

reader = Reader()


def runSeparation(func, segment):
    N = None

    while True:
        if N is None:
            N = reader.value("N (число разбиений): ", int)

        if N < 2:
            print("N must be >= 2")
            N = None
            continue

        h = segment.len() / N
        signChangeSegments = separateRoots(func, segment, N)
        print(
            f"Found {len(signChangeSegments)} sign change intervals with step h={h:.12f}"
        )

        for i, s in enumerate(signChangeSegments, 1):
            print(f"{i}. {s}")

        answer = reader.line(
            "Enter a new N to repeat the separation, or press Enter to continue: "
        ).strip()

        if answer == "":
            return signChangeSegments

        N = reader.parse(answer, int)


def printRefinementTable(infos, func):
    header = (
        f"{'Method':<27} {'Steps':>6} {'Root':>21} "
        f"{'|f(root)|':>13} {'Error bound':>19}"
    )

    print(header)
    print("-" * len(header))

    for info in infos:
        if info.error is not None:
            print(f"{info.name:<27} {info.stepsCount:>6}   failed: {info.error}")
            continue

        row = (
            f"{info.name:<27} {info.stepsCount:>6} "
            f"{info.approximation:>21.16f} "
            f"{abs(tryEvaluate(func, info.approximation)):>13.3e}"
        )

        if info.errorEstimate is not None:
            row += f" {info.errorEstimate:>19.3e}"
        else:
            row += f" {'-':>19}"

        print(row)


def readEps():
    while True:
        eps = reader.value("EPS: ", float)

        if math.isfinite(eps) and eps > 0:
            return eps

        print("EPS must be positive")


def runUntilMainMenu(body, prompt, onRepeat=None):
    while True:
        body()

        answer = reader.line(prompt).strip().lower()

        if answer != "r":
            return

        if onRepeat is not None:
            onRepeat()


def runRefinement(func, derivative, secondDerivative, signChangeSegments, eps):
    def body():
        while True:
            index = reader.value(
                f"Interval number (1..{len(signChangeSegments)}): ", int
            )

            if 1 <= index <= len(signChangeSegments):
                break

            print(
                f"The interval number must be between 1 and {len(signChangeSegments)}"
            )

        segment = signChangeSegments[index - 1]
        newtonAvailable = (
            derivative is not None
            and keepsConstantSign(derivative, segment)
            and keepsConstantSign(secondDerivative, segment)
        )
        infos = [approximateByBisection(func, segment, eps)]

        if newtonAvailable:
            infos.append(approximateByNewton(func, derivative, segment, eps))
            infos.append(
                approximateByNewton(func, derivative, segment, eps, fixedSlope=True)
            )

        infos.append(approximateBySecant(func, derivative, segment, eps))

        print()
        print(f"Interval {segment}")

        if derivative is not None and not newtonAvailable:
            print(
                "f'(x) and f''(x) must keep a constant sign: "
                "Newton's Method and Modified Newton's Method are not available"
            )

        printRefinementTable(infos, func)

    def updateEps():
        nonlocal eps
        eps = readEps()

    runUntilMainMenu(
        body,
        'Enter "r" to refine another interval, or press Enter for the main menu: ',
        updateEps,
    )


def runTask(func, derivative, secondDerivative, segment, eps):
    signChangeSegments = runSeparation(func, segment)

    if not signChangeSegments:
        print("No sign change intervals found")
        return

    if eps is None:
        eps = readEps()

    runRefinement(func, derivative, secondDerivative, signChangeSegments, eps)


def runTestTask():
    print("f(x) = 4*cos(x) + 0.3*x")
    print("[A ; B] = [-15 ; 5]")
    print(f"EPS = {EPS:.12f}")
    print()

    func, derivative, secondDerivative = buildFunction("4*cos(x) + 0.3*x")
    runTask(func, derivative, secondDerivative, Segment(A, B), EPS)


def runBallTask():
    def body():
        while True:
            r = reader.value("r: ", float)

            if math.isfinite(r) and r > 0:
                break

            print("r must be positive")

        print("f(d) = (pi/3)*(d^3 - 3*d^2*r + 4*r^3*rho)")
        print(f"[0 ; 2r] = [0.000000000000 ; {2 * r:.12f}]")
        print(f"EPS = {EPS:.12f}")
        print("Method: Newton's Method")
        print()

        header = (
            f"{'Substance':<16} {'p g/ml':>6} {'Depth d':>21} "
            f"{'Steps':>6} {'|f(d)|':>13}"
        )
        print(header)
        print("-" * len(header))

        segment = Segment(0, 2 * r)
        sankAny = False

        for substance, density in MATERIALS:
            try:
                func, derivative, _ = buildFunction(
                    f"(pi/3)*(x**3 - 3*x**2*{r} + 4*{r}**3*{density})"
                )
            except ValueError as error:
                print(f"{substance:<16} {density:>6.2f}   failed: {error}")
                continue

            valueAtStart = tryEvaluate(func, segment.left)
            valueAtEnd = tryEvaluate(func, segment.right)
            hasRoot = (
                math.isfinite(valueAtStart)
                and math.isfinite(valueAtEnd)
                and valueAtEnd <= 1e-15 * abs(valueAtStart)
            )

            if not hasRoot:
                print(f"{substance:<16} {density:>6.2f} {'-':>21} {'-':>6} {'-':>13}")
                sankAny = True
                continue

            info = approximateByNewton(func, derivative, segment, EPS)

            if info.error is not None:
                print(f"{substance:<16} {density:>6.2f}   failed: {info.error}")
                continue

            residual = abs(tryEvaluate(func, info.approximation))
            print(
                f"{substance:<16} {density:>6.2f} {info.approximation:>21.3f} "
                f"{info.stepsCount:>6} {residual:>13.3e}"
            )

        if sankAny:
            print("( - : the ball sinks, no equilibrium root in [0; 2r])")

    runUntilMainMenu(
        body,
        'Enter "r" to refine another radius, or press Enter for the main menu: ',
    )


def runCustomTask():
    func, derivative, secondDerivative = reader.function()
    runTask(func, derivative, secondDerivative, reader.interval(), None)


def main():
    while True:
        print()
        print("Numerical methods for solving nonlinear equations")
        print("1. Test task")
        print("2. Custom task")
        print("3. Ball task")
        print("0. Exit")

        task = reader.value("Task: ", int)

        if task == 1:
            runTestTask()
        elif task == 2:
            runCustomTask()
        elif task == 3:
            runBallTask()
        elif task == 0:
            return
        else:
            print("Unknown task")


if __name__ == "__main__":
    main()
