import logging

# Configure logging
logger = logging.getLogger('gym_cutting_stock')
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)  # Default level

def set_log_level(level):
    """
    Set the logging level for the gym_cutting_stock package.
    
    Args:
        level: Can be logging.DEBUG, logging.INFO, logging.WARNING, 
              logging.ERROR, logging.CRITICAL
              or equivalent string ('DEBUG', 'INFO', etc.)
    """
    if isinstance(level, str):
        level = level.upper()
        level = getattr(logging, level)
    logger.setLevel(level)
