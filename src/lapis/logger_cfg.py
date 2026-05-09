import logging

# Create and configure the logger
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("my_logger")
logger.setLevel(logging.DEBUG)
logger.propagate = False  # Prevent messages from bubbling up to the root logger

# Define log format
formatter = logging.Formatter(
    "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"
)

# Create a console handler (INFO+)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Create a file handler (DEBUG+)
file_handler = logging.FileHandler("app.log", mode="a")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Test logging
logger.debug("This will be in the file but not the console")
logger.info("This will be in both the console and the file")
logger.warning("Warnings and above also go to both")

