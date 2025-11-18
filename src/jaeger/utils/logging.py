from __future__ import annotations
import logging
import sys
from datetime import datetime
from pathlib import Path

from rich.logging import RichHandler


def description(version):
    return rf"""
                  .
               ,'/ \`.
              |\/___\/|
              \'\   /`/          ██╗ █████╗ ███████╗ ██████╗ ███████╗██████╗
               `.\ /,'           ██║██╔══██╗██╔════╝██╔════╝ ██╔════╝██╔══██╗
                  |              ██║███████║█████╗  ██║  ███╗█████╗  ██████╔╝
                  |         ██   ██║██╔══██║██╔══╝  ██║   ██║██╔══╝  ██╔══██╗
                 |=|        ╚█████╔╝██║  ██║███████╗╚██████╔╝███████╗██║  ██║
            /\  ,|=|.  /\    ╚════╝ ╚═╝  ╚═╝╚══════╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝
        ,'`.  \/ |=| \/  ,'`. 
      ,'    `.|\ `-' /|,'    `. 
    ,'   .-._ \ `---' / _,-.   `.
       ,'    `-`-._,-'-'   `.
      '
        ## Jaeger {version} (yet AnothEr phaGe idEntifier) Deep-learning based
        bacteriophage discovery https://github.com/Yasas1994/Jaeger.git
    """





def get_logger(log_path: Path, log_file: str, level: int) -> logging.Logger:
    # Create a custom logger
    current_datetime = datetime.now()
    current_date_time = current_datetime.strftime("%m%d%Y_%H%M%S")
    logger = logging.getLogger("jaeger")
    levels = {1: logging.INFO, 2: logging.DEBUG}
    loglevel = levels.get(level, logging.INFO)
    logger.setLevel(loglevel)

    # Prevent duplicate handlers if this is called multiple times
    logger.propagate = False
    for h in list(logger.handlers):
        logger.removeHandler(h)

    # Formatter (used for file; Rich renders console nicely on its own)
    formatter = logging.Formatter(
        "[%(name)s] %(filename)8s:%(lineno)3d| %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler (Rich)
    console_handler = RichHandler(
        show_time=True,
        omit_repeated_times=False,
        show_level=True,
        show_path=False,
        enable_link_path=False,
    )
    # Keep console simple; RichHandler formats its own output
    console_handler.setFormatter(formatter)
    console_handler.setLevel(loglevel)
    logger.addHandler(console_handler)

    # Optional file handler — only add if log_file is truthy (non-empty string)
    if log_file:
        # Ensure directory exists
        if log_path is not None:
            log_path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path / Path(f"{current_date_time}_{log_file}"))
        file_handler.setLevel(logging.DEBUG)  # file logs DEBUG and above
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
