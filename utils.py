# utils.py

import logging

def initialize_logging():
    logging.basicConfig(
        filename='logs/system.log',
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
    )
