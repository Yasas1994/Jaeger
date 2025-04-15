import logging

logger = logging.getLogger("Jaeger")


def safe_divide(numerator, denominator):
    try:
        result = round(numerator / denominator, 2)
    except ZeroDivisionError:
        logger.debug("Error: Division by zero!")
        result = 0
    return result


def signal_fl(it):
    """get a signal at the begining and the end of a iterator"""
    iterable = iter(it)
    yield True, next(iterable)
    ret_var = next(iterable)
    for val in iterable:
        yield 0, ret_var
        ret_var = val
    yield 1, ret_var


def signal_l(it):
    """get a signal at the end of the iterator"""
    iterable = iter(it)
    ret_var = next(iterable)
    for val in iterable:
        yield 0, ret_var
        ret_var = val
    yield 1, ret_var


def format_seconds(seconds):
    minutes = seconds // 60
    remaining_seconds = seconds % 60
    return f"{minutes} minutes and {remaining_seconds} seconds"



