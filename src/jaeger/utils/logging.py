import sys
import logging
from pathlib import Path
from datetime import datetime
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
    loglevel = levels[level]
    logger.setLevel(
        loglevel
    )  # Set minimum level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    # Create handlers
    console_handler = logging.StreamHandler(sys.stderr)  # Logs to console
    file_handler = logging.FileHandler(
        log_path / Path(f"{current_date_time}_{log_file}")
    )  # Logs to a file

    console_handler.setLevel(loglevel)  # Console shows INFO and above
    file_handler.setLevel(logging.DEBUG)  # File logs DEBUG and above

    formatter = logging.Formatter(
        "[%(name)s] %(asctime)s | %(levelname)7s | %(filename)8s:%(lineno)3d| %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler = RichHandler(
        show_time=False,
        omit_repeated_times=False,
        show_level=False,
        show_path=False,
        enable_link_path=False,
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
