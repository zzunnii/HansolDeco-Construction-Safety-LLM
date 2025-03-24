import re
from openai import OpenAI
from utils.common import logger, fixed_seed
from config import API_BASE_URL, SYSTEM_PROMPT


def initialize_client():
    """vLLM 클라이언트 초기화"""
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key="token",  # vLLM은 API 키 필요 없음
        timeout=600  # 10분 타임아웃
    )
    logger.info("OpenAI 클라이언트 초기화 완료")
    return client


def call_model(client, prompt, system=SYSTEM_PROMPT, history=None):
    """모델 호출 및 응답 처리"""
    logger.info(f"모델 입력 프롬프트:\n{'-' * 50}\n{prompt}\n{'-' * 50}")
    messages = [{"role": "system", "content": system}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": prompt})

    try:
        response = client.chat.completions.create(
            model="Qwen/QwQ-32B",
            messages=messages,
            temperature=0.6,
            top_p=0.95,
            seed=fixed_seed,
            max_tokens=6000
        )
        full_response = response.choices[0].message.content.strip()

        # <think> 태그 파싱
        think_match = re.search(r'<think>(.*?)</think>', full_response, re.DOTALL)
        think_content = think_match.group(1).strip() if think_match else None
        assistant_content = re.sub(r'<think>.*?</think>', '', full_response, flags=re.DOTALL).strip()

        logger.info(f"모델 출력 응답:\n{'-' * 50}\n{full_response}\n{'-' * 50}")
        return {"response": assistant_content, "thought": think_content}
    except Exception as e:
        logger.error(f"모델 호출 오류: {e}")
        return {"response": "모델 호출 실패", "thought": None}