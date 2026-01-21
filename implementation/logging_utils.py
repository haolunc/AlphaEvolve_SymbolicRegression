import logging
import os

def setup_logger(name: str, log_dir: str = "./logger"):
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        return logger  # Prevent adding handlers multiple times

    logger.setLevel(logging.DEBUG)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Add handlers
    # console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(log_dir+"/"+name+".log")

    # Set handler levels
    # console_handler.setLevel(logging.WARNING)
    file_handler.setLevel(logging.DEBUG)

    # Create and add formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    # console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add handlers to logger
    # logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger 


def setup_console_logger(name: str):
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        return logger  # Prevent adding handlers multiple times

    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    return logger 
