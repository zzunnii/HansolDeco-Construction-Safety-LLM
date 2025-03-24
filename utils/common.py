import os
import re
import random
import numpy as np
import logging
from config import LOG_FILE


def setup_logging():
    """로깅 설정"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(LOG_FILE,encoding='utf-8')]
    )
    return logging.getLogger(__name__)


logger = setup_logging()

fixed_seed=42

def set_seed(seed=fixed_seed):
    """랜덤 시드 고정"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    logger.info(f"랜덤 시드 고정: {seed}")


def parse_response(response):
    """모델 응답에서 최종 대책 추출"""
    boxed_match = re.search(r'\\boxed\{(.*?)\}', response, re.DOTALL)
    if not boxed_match:
        # 다른 형식 시도
        boxed_match = re.search(r'boxed\{(.*?)\}', response, re.DOTALL)

    prevention_plan = boxed_match.group(1).strip() if boxed_match else "기본 안전관리 강화"
    return prevention_plan