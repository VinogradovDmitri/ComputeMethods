import sys

from reader import reader


def clearScreen():
    print("\033[2J\033[H", end="")


COLORS = sys.stdout.isatty()


def paint(text, code):
    return f"\033[{code}m{text}\033[0m" if COLORS else text


def bold(text):
    return paint(text, "1")


def dim(text):
    return paint(text, "2")


def red(text):
    return paint(text, "31")


def green(text):
    return paint(text, "32")


def yellow(text):
    return paint(text, "33")


def cyan(text):
    return paint(text, "36")


def section(title):
    print()
    print(bold(cyan(f"=== {title} " + "=" * max(0, 54 - len(title)))))


def runUntilMainMenu(body, prompt, onRepeat=None):
    while True:
        body()

        answer = reader.line(prompt).strip().lower()

        if answer != "r":
            clearScreen()
            return

        if onRepeat is not None:
            onRepeat()
