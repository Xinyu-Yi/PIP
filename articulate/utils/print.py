r"""
    Print with color.
    Colored text and background from https://www.geeksforgeeks.org/print-colors-python-terminal/.
    Modified from https://github.com/luigifreda/pyslam/blob/master/utils_sys.py.
"""


__all__ = ['Color', 'print_red', 'print_error', 'print_green', 'print_orange', 'print_purple',
           'print_yellow', 'print_blue', 'print_cyan']


import sys


class Color:
    r"""
    Colors used in print().
    """
    reset = '\033[0m'
    bold = '\033[01m'
    disable = '\033[02m'
    underline = '\033[04m'
    reverse = '\033[07m'
    strikethrough = '\033[09m'
    invisible = '\033[08m'

    class fg:
        black = '\033[30m'
        red = '\033[31m'
        green = '\033[32m'
        orange = '\033[33m'
        blue = '\033[34m'
        purple = '\033[35m'
        cyan = '\033[36m'
        lightgrey = '\033[37m'
        darkgrey = '\033[90m'
        lightred = '\033[91m'
        lightgreen = '\033[92m'
        yellow = '\033[93m'
        lightblue = '\033[94m'
        pink = '\033[95m'
        lightcyan = '\033[96m'

    class bg:
        black = '\033[40m'
        red = '\033[41m'
        green = '\033[42m'
        orange = '\033[43m'
        blue = '\033[44m'
        purple = '\033[45m'
        cyan = '\033[46m'
        lightgrey = '\033[47m'


def print_red(*args, **kwargs):
    print(Color.fg.red, end="")
    print(*args, **kwargs)
    print(Color.reset, end="", flush=True)


def print_green(*args, **kwargs):
    print(Color.fg.green, end="")
    print(*args, **kwargs)
    print(Color.reset, end="", flush=True)


def print_blue(*args, **kwargs):
    print(Color.fg.blue, end="")
    print(*args, **kwargs)
    print(Color.reset, end="", flush=True)


def print_cyan(*args, **kwargs):
    print(Color.fg.cyan, end="")
    print(*args, **kwargs)
    print(Color.reset, end="", flush=True)


def print_orange(*args, **kwargs):
    print(Color.fg.orange, end="")
    print(*args, **kwargs)
    print(Color.reset, end="", flush=True)


def print_purple(*args, **kwargs):
    print(Color.fg.purple, end="")
    print(*args, **kwargs)
    print(Color.reset, end="", flush=True)


def print_yellow(*args, **kwargs):
    print(Color.fg.yellow, end="")
    print(*args, **kwargs)
    print(Color.reset, end="", flush=True)


def print_error(*args, **kwargs):
    print(Color.fg.red, end="")
    print(*args, **kwargs, file=sys.stderr)
    print(Color.reset, end="", flush=True)


print_warning = print_orange
