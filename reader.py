from segment import Segment
from methods import buildFunction


class Reader(object):
    ERRORS = {int: "Please enter an integer", float: "Please enter a number"}

    def line(self, prompt):
        try:
            return input(prompt)
        except EOFError:
            raise SystemExit(1)

    def parse(self, text, converter):
        try:
            return converter(text)
        except ValueError as error:
            message = self.ERRORS.get(converter, str(error))
            print(f'{message}: got "{text}"')

    def value(self, prompt, converter):
        while True:
            result = self.parse(self.line(prompt).strip(), converter)

            if result is not None:
                return result

    def optionalValue(self, prompt, converter, default):
        while True:
            text = self.line(f"{prompt} [{default}]: ").strip()

            if text == "":
                return default

            result = self.parse(text, converter)

            if result is not None:
                return result

    def function(self):
        func, derivative, secondDerivative = self.value("f(x) = ", buildFunction)

        if derivative is None:
            print(
                "f'(x) does not exist or cannot be computed for this expression "
                "(differentiation failed or f'(x) is not finite anywhere): "
                "Newton's Method and Modified Newton's Method are not available"
            )

        return func, derivative, secondDerivative

    def interval(self):
        while True:
            left = self.value("A: ", float)
            right = self.value("B: ", float)

            if right > left:
                return Segment(left, right)

            print(f"B must be greater than A: got A = {left}, B = {right}")


reader = Reader()
