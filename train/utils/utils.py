import os
import sys
import time
import logging

def get_logger(exp_dir):
    logger = logging.getLogger('log')
    logger.setLevel(logging.DEBUG)
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(exp_dir, '{}.log'.format(timestamp))
    file_handler = logging.FileHandler(log_file)
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger