import logging
from decimal import Decimal

logger = logging.getLogger("Jaeger")

from rich.progress import ProgressColumn
from rich.text import Text
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from time import sleep

class MsPerStepColumn(ProgressColumn):
    """Custom column to display milliseconds per step."""
    def render(self, task):
        if task.speed and task.speed > 0:
            ms_per_step = 1000 / task.speed
            return Text(f"{ms_per_step:.2f} ms/step")
        return Text("– ms/step")



def track_ms(iterable, description="Working..."):
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[cyan]{task.description}"),
        BarColumn(pulse_style="cyan") if not hasattr(iterable, '__len__') else BarColumn(),
        TaskProgressColumn(),
        MsPerStepColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )
    with progress:
        task = progress.add_task(description, total=None)
        for item in iterable:
            yield item
            progress.update(task, advance=1)


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



# from https://github.com/davidsa03/numerize
def round_num(n, decimal=2):
    n=Decimal(n)
    return n.to_integral() if n == n.to_integral() else round(n.normalize(), decimal)

def numerize(n, decimal=2):
    #60 sufixes
    sufixes = [ "", "K", "M", "B", "T", "Qa", "Qu", "S", "Oc", "No", 
    "D", "Ud", "Dd", "Td", "Qt", "Qi", "Se", "Od", "Nd","V", 
    "Uv", "Dv", "Tv", "Qv", "Qx", "Sx", "Ox", "Nx", "Tn", "Qa",
    "Qu", "S", "Oc", "No", "D", "Ud", "Dd", "Td", "Qt", "Qi",
    "Se", "Od", "Nd", "V", "Uv", "Dv", "Tv", "Qv", "Qx", "Sx",
    "Ox", "Nx", "Tn", "x", "xx", "xxx", "X", "XX", "XXX", "END"] 

    sci_expr = [1e0, 1e3, 1e6, 1e9, 1e12, 1e15, 1e18, 1e21, 1e24, 1e27, 
    1e30, 1e33, 1e36, 1e39, 1e42, 1e45, 1e48, 1e51, 1e54, 1e57, 
    1e60, 1e63, 1e66, 1e69, 1e72, 1e75, 1e78, 1e81, 1e84, 1e87, 
    1e90, 1e93, 1e96, 1e99, 1e102, 1e105, 1e108, 1e111, 1e114, 1e117, 
    1e120, 1e123, 1e126, 1e129, 1e132, 1e135, 1e138, 1e141, 1e144, 1e147, 
    1e150, 1e153, 1e156, 1e159, 1e162, 1e165, 1e168, 1e171, 1e174, 1e177]
    minus_buff = n
    n=abs(n)
    for x in range(len(sci_expr)):
        try:
            if n >= sci_expr[x] and n < sci_expr[x+1]:
                sufix = sufixes[x]
                if n >= 1e3:
                    num = str(round_num(n/sci_expr[x], decimal))
                else:
                    num = str(n)
                return num + sufix if minus_buff > 0 else "-" + num + sufix
        except IndexError:
            print("You've reached the end")