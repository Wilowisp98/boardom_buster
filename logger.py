import os
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime

class CustomLogger:
    """
    A custom logger class that provides logging functionality with separate files for
    regular logs and errors, using date-stamped filenames and rotation.
    """

    def __init__(self, logger_name: str = 'CustomLogger', log_level: int = logging.DEBUG, base_dir: str = None):
        """
        Initialize the custom logger with specified name, log level, and base directory.

        Args:
            logger_name (str): Name of the logger instance
            log_level (int): Logging level (e.g., logging.DEBUG, logging.INFO)
            base_dir (str): Base directory for log and error files
        """
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(log_level)
        self.base_dir = base_dir if base_dir else os.path.dirname(os.path.abspath(__file__))
        self.error_handler = None

        # Avoid adding handlers if they already exist
        if not self.logger.hasHandlers():
            self._setup_logging()

    def _setup_logging(self):
        """
        Set up logging configuration with regular logs handler and delayed error handler.
        Creates log directory if it doesn't exist.
        """
        # Create log directory inside the specified base directory
        log_dir = os.path.join(self.base_dir, 'log')
        os.makedirs(log_dir, exist_ok=True)

        # Get current date for log files
        current_date = datetime.now().strftime('%Y%m%d')

        # Create formatters with custom datetime format
        log_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Regular log file handler
        log_file = os.path.join(log_dir, f'log_{current_date}.log')
        file_handler = RotatingFileHandler(
            log_file,
            mode='w',
            maxBytes=10485760,
            backupCount=5
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(log_formatter)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(log_formatter)

        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        # Create custom error handler
        self._create_error_handler()

    def _create_error_handler(self):
        """
        Creates a custom error handler that only creates the error file when an error occurs.
        """
        class ErrorHandler(RotatingFileHandler):
            def __init__(self, filename, mode='a', maxBytes=0, backupCount=0, encoding=None, delay=True):
                # Use delay=True to prevent file creation until first use
                super().__init__(filename, mode, maxBytes, backupCount, encoding, delay=True)
                self.setFormatter(logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s\n'
                    'Stack Trace:\n%(exc_info)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                ))
                self.setLevel(logging.ERROR)

            def emit(self, record):
                if not record.levelno >= logging.ERROR:
                    return
                if not os.path.exists(os.path.dirname(self.baseFilename)):
                    os.makedirs(os.path.dirname(self.baseFilename), exist_ok=True)
                super().emit(record)

        # Create error directory
        err_dir = os.path.join(self.base_dir, 'err')
        current_date = datetime.now().strftime('%Y%m%d')
        error_file = os.path.join(err_dir, f'error_{current_date}.err')

        # Create and add the error handler
        self.error_handler = ErrorHandler(
            error_file,
            mode="w",
            maxBytes=10485760,
            backupCount=5
        )
        self.logger.addHandler(self.error_handler)

    def get_logger(self) -> logging.Logger:
        """
        Get the configured logger instance.

        Returns:
            logging.Logger: Configured logger instance
        """
        return self.logger

# Convenience function to get a logger instance
def get_logger(name: str = 'CustomLogger', level: int = logging.DEBUG, base_dir: str = None) -> logging.Logger:
    """
    Get a configured logger instance.

    Args:
        name (str): Name for the logger
        level (int): Logging level
        base_dir (str): Base directory for log and error files

    Returns:
        logging.Logger: Configured logger instance
    """
    return CustomLogger(name, level, base_dir).get_logger()