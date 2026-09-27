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
            print(self.ERRORS.get(converter, error))

    def value(self, prompt, converter):
        while True:
            result = self.parse(self.line(prompt).strip(), converter)

            if result is not None:
                return result

    def function(self):
        func, derivative, secondDerivative = self.value("f(x) = ", buildFunction)

        if derivative is None:
            print(
                "Cannot find f'(x): Newton's Method and Modified Newton's Method are not available"
            )

        return func, derivative, secondDerivative

    def interval(self):
        while True:
            left = self.value("A: ", float)
            right = self.value("B: ", float)

            if right > left:
                return Segment(left, right)

            print("B must be greater than A")
