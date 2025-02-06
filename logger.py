import os
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime

class CustomLogger:
    """
    A custom logger class that provides logging functionality with separate files for
    regular logs and errors, using date-stamped filenames and rotation.
    """

    def __init__(self, logger_name: str = 'CustomLogger', log_level: int = logging.DEBUG):
        """
        Initialize the custom logger with specified name and log level.

        Args:
            logger_name (str): Name of the logger instance
            log_level (int): Logging level (e.g., logging.DEBUG, logging.INFO)
        """
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(log_level)

        # Avoid adding handlers if they already exist
        if not self.logger.hasHandlers():
            self._setup_logging()

    def _setup_logging(self):
        """
        Set up logging configuration with separate handlers for regular logs and errors.
        Creates log and error directories if they don't exist.
        """
        # Create log directories
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        log_dir = os.path.join(base_dir, 'log')
        err_dir = os.path.join(base_dir, 'err')
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(err_dir, exist_ok=True)

        # Get current date for log files
        current_date = datetime.now().strftime('%Y%m%d')

        # Create formatters
        log_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        error_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s\n'
            'Stack Trace:\n%(exc_info)s'
        )

        # Regular log file handler
        log_file = os.path.join(log_dir, f'log_{current_date}.log')
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10485760,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(log_formatter)

        # Error log file handler
        error_file = os.path.join(err_dir, f'error_{current_date}.err')
        error_handler = RotatingFileHandler(
            error_file,
            maxBytes=10485760,  # 10MB
            backupCount=5
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(error_formatter)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(log_formatter)

        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(error_handler)
        self.logger.addHandler(console_handler)

    def get_logger(self) -> logging.Logger:
        """
        Get the configured logger instance.

        Returns:
            logging.Logger: Configured logger instance
        """
        return self.logger

# Convenience function to get a logger instance
def get_logger(name: str = 'CustomLogger', level: int = logging.DEBUG) -> logging.Logger:
    """
    Get a configured logger instance.

    Args:
        name (str): Name for the logger
        level (int): Logging level

    Returns:
        logging.Logger: Configured logger instance
    """
    return CustomLogger(name, level).get_logger()
