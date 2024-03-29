import sys

from termcolor import colored

from cythonpkg.harmonic_mean import harmonic_mean


def main(args=sys.argv[1:]) -> None:
    nums = _parse_nums(args)
    result = _calculate_results(nums)
    print(_format_output(result))


def _parse_nums(inputs: list[str]) -> list[float]:
    try:
        nums = [float(num) for num in inputs]
    except ValueError:
        nums = []

    return nums


def _calculate_results(nums: list[float]) -> float:
    result = 0.0
    if len(nums) == 0:
        pass
    else:
        result = harmonic_mean(nums)

    return result


def _format_output(result: float) -> str:
    return colored(str(result), "red", "on_black", attrs=["bold"])


if __name__ == "__main__":
    args = [0.65, 0.7]
    main(args)
