import logging
from decimal import Decimal
import jinja2
from pathlib import Path
from typing import Dict
import yaml
import json
import os
from collections import defaultdict
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
            ms_per_step = 1000/ task.speed
            return Text(f"{ms_per_step:.0f} ms/step")
        return Text("– ms/step")



def track_ms(iterable, description="Working...", disable=False):
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[cyan]{task.description}"),
        BarColumn(pulse_style="cyan") if not hasattr(iterable, '__len__') else BarColumn(),
        TaskProgressColumn(),
        MsPerStepColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        disable=disable
    )
    with progress:
        task = progress.add_task(description, total=None)
        for item in iterable:
            yield item
            progress.update(task, advance=1)

def load_model_config(path: Path) -> Dict:
    '''
    loads the configuration file from the template
    '''
    
    with open(path) as fp:
        _data = yaml.safe_load(fp)

    env = jinja2.Environment(loader=jinja2.FileSystemLoader(searchpath=path.parent))
    template = env.get_template(path.name)

    data = yaml.safe_load(template.render(_data))

    return data

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

def json_to_dict(path: str):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {} 

def add_data_to_json(path: str, new_data: dict, list_key: str = None):
    """
    Add new_data into the JSON at `path`.  
    - If the top‐level JSON is a dict, this will merge new_data’s keys.  
    - If list_key is provided, it will append new_data into the list at that key.
    """
    data = json_to_dict(path)

    if list_key:
        # ensure it’s a list
        data.setdefault(list_key, [])
        if not isinstance(data[list_key], list):
            raise ValueError(f"Expected {list_key} to be a list")
        data[list_key].append(new_data)
    else:
        if not isinstance(data, dict):
            raise ValueError("Top‐level JSON is not an object")
        data.update(new_data)

    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)



class AvailableModels:
    """
    get all available models from the model path
    """
    def __init__(self, path):
        if isinstance(path, str) or isinstance(path, Path):
            self.paths = [Path(path)]
        elif isinstance(path, list):
            self.paths = [Path(i) for i in path]
        self.info = self._scan_for_models()
        

    def _scan_for_models(self) -> defaultdict:
        _tmp = defaultdict(dict)
        for path in self.paths:
            for model_graph in path.rglob("*_graph"):
                if model_graph.is_dir():
                    _tmp[model_graph.name.rstrip("_graph")]['graph'] = model_graph
                for _match in ("classes", "project", "weights"):
                    for _cfg in model_graph.parent.rglob(f"*_{_match}.yaml"):
                        if _cfg.is_file():
                            _tmp[model_graph.name.rstrip("_graph")][_match] = _cfg
        
        return _tmp    
    
def get_model_id(model:str):
    return model.split("_", 1)[1].rsplit("_", 1)[0]