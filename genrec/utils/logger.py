import logging
from tqdm.auto import tqdm
from colorama import Fore, Style


class ColorFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\033[94m',  # Blue
        'INFO': '\033[92m',   # Green
        'WARNING': '\033[93m', # Yellow
        'ERROR': '\033[91m',   # Red
        'CRITICAL': '\033[41m' # Red background
    }
    RESET = '\033[0m'

    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.RESET)
        formatted_message = super().format(record)
        return f"{log_color}{formatted_message}{self.RESET}"

def setup_logger(name=__name__):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    formatter = ColorFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)

    logger.addHandler(ch)
    return logger

def bar_iter(iterable, epoch: int, num_epochs: int, prefix: str = ""):
    return tqdm(
        iterable,
        desc = f"{Fore.CYAN}{prefix}{Style.RESET_ALL} "+ f"{Fore.CYAN}Epoch {epoch+1}/{num_epochs}{Style.RESET_ALL}",
        total=len(iterable),
        dynamic_ncols=True,
        leave=False,
        bar_format=(
            "{l_bar}"
            "{bar:30}"
            "| {n_fmt}/{total_fmt} "
            "[{elapsed}<{remaining}, {rate_fmt}]"
        ),
    )
