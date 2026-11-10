from reader import reader
from style import clearScreen, bold, dim, red
from equations import runTestTask, runCustomTask, runBallTask
from interpolation_task import runInterpolationTask
from differentiation_task import runDifferentiationTask


def main():
    while True:
        print()
        print(bold("Numerical methods for solving nonlinear equations"))
        print(dim("-" * 50))
        print("1. Test task")
        print("2. Custom task")
        print("3. Ball task")
        print("4. Interpolation task")
        print("5. Differentiation task")
        print(dim("0. Exit"))

        task = reader.value("Task: ", int)

        if task == 1:
            clearScreen()
            runTestTask()
        elif task == 2:
            clearScreen()
            runCustomTask()
        elif task == 3:
            clearScreen()
            runBallTask()
        elif task == 4:
            clearScreen()
            runInterpolationTask()
        elif task == 5:
            clearScreen()
            runDifferentiationTask()
        elif task == 0:
            return
        else:
            print(red(f"Unknown task: {task} (enter 0..5)"))


if __name__ == "__main__":
    main()
