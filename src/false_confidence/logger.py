import logging
import sys

SUCCESS_LEVEL = 25
logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")


class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    WHITE = "\033[97m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    GRAY = "\033[90m"
    CYAN = "\033[96m"


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors"""

    LEVEL_COLORS = {
        logging.DEBUG: Colors.GRAY,
        logging.INFO: Colors.WHITE,
        SUCCESS_LEVEL: Colors.GREEN,
        logging.WARNING: Colors.YELLOW,
        logging.ERROR: Colors.RED,
    }

    def format(self, record):
        level_color = self.LEVEL_COLORS.get(record.levelno, Colors.RESET)
        record.levelname = f"{level_color}{record.levelname}{Colors.RESET}"

        record.name = f"{Colors.CYAN}{record.name}{Colors.RESET}"

        if record.levelno == logging.ERROR:
            record.msg = f"{Colors.RED}{record.msg}{Colors.RESET}"
        elif record.levelno == logging.WARNING:
            record.msg = f"{Colors.YELLOW}{record.msg}{Colors.RESET}"
        elif record.levelno == SUCCESS_LEVEL:
            record.msg = f"{Colors.GREEN} {record.msg}{Colors.RESET}"

        return super().format(record)


class AppLogger:
    """Simple logger with colors and context"""

    def __init__(self, name: str = "app"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(
                ColoredFormatter(fmt="[%(levelname)s] [%(name)s] %(message)s", datefmt="%H:%M:%S")
            )
            self.logger.addHandler(handler)
            self.logger.propagate = False

    def _format_extras(self, **kwargs) -> str:
        """Format extra context fields"""
        if not kwargs:
            return ""
        parts = [f"{Colors.GRAY}{k}={v}{Colors.RESET}" for k, v in kwargs.items()]
        return " " + " ".join(parts)

    def info(self, message: str, **kwargs):
        """Log info message with optional context"""
        extras = self._format_extras(**kwargs)
        self.logger.info(f"{message}{extras}")

    def warning(self, message: str, **kwargs):
        """Log warning message with optional context"""
        extras = self._format_extras(**kwargs)
        self.logger.warning(f"{message}{extras}")

    def error(self, message: str, **kwargs):
        """Log error message with optional context"""
        extras = self._format_extras(**kwargs)
        self.logger.error(f"{message}{extras}")

    def success(self, message: str, **kwargs):
        """Log success message with optional context"""
        extras = self._format_extras(**kwargs)
        self.logger.log(SUCCESS_LEVEL, f"{message}{extras}")


def get_logger(name: str = "app") -> AppLogger:
    """Get a logger for a specific module"""
    return AppLogger(name)
