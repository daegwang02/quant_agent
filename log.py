# log.py

import logging

def get_module_logger(name):
    logger = logging.getLogger(name)
    if not logger.handlers:
        # 기본 로그 핸들러 설정
        formatter = logging.Formatter(
            "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"
        )
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)  # 로그 레벨 설정
    return logger
