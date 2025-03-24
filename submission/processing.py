import os
import pandas as pd
from tqdm import tqdm
from utils.common import logger
from workflow.workflow import run_workflow
from config import TEST_FILE, TEXT_OUTPUT_DIR, OUTPUT_DIR
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import re

# SBERT 모델 로드 (jhgan/ko-sbert-sts)
tokenizer = AutoTokenizer.from_pretrained("jhgan/ko-sbert-sts")
model = AutoModel.from_pretrained("jhgan/ko-sbert-sts")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# 데이터 로드 및 전처리
def load_and_preprocess_data(file_path):
    logger.info(f"데이터 로드 시작: {file_path}")
    df = pd.read_csv(file_path, encoding='utf-8').fillna("정보 없음")
    try:
        df['공사종류(대분류)'] = df['공사종류'].str.split(' > ').str[0]
        df['공사종류(중분류)'] = df['공사종류'].str.split(' > ').str[1]
        df['공종(대분류)'] = df['공종'].str.split(' > ').str[0]
        df['공종(중분류)'] = df['공종'].str.split(' > ').str[1]
        df['사고객체(대분류)'] = df['사고객체'].str.split(' > ').str[0]
        df['사고객체(중분류)'] = df['사고객체'].str.split(' > ').str[1]
    except Exception as e:
        logger.error(f"데이터 전처리 중 오류 발생: {e}")
    logger.info(f"데이터 로드 및 전처리 완료: {len(df)} 행")
    return df


# 텍스트를 768차원 벡터로 변환
def get_sbert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    # [CLS] 토큰의 임베딩을 사용 (768차원)
    embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    return embeddings.flatten()  # 1D 배열로 반환 (768차원)


# 응답에서 텍스트 추출하는 함수
def parse_response(response):
    # 1. boxed 패턴 확인 (LaTeX 형식)
    boxed_match = re.search(r'\\boxed\{(.*?)\}', response, re.DOTALL)
    if boxed_match:
        return boxed_match.group(1).strip()

    # 2. boxed 패턴 확인 (LaTeX 형식 - 백슬래시 없는 버전)
    boxed_match = re.search(r'boxed\{(.*?)\}', response, re.DOTALL)
    if boxed_match:
        return boxed_match.group(1).strip()

    # 3. JSON 형식 응답과 같은 패턴 제거
    cleaned_response = re.sub(r'{"answer":\s*"[A-Z]"}\s*', '', response)

    # 4. "<think>" 태그가 있는 경우, 이를 처리
    if "<think>" in response and "</think>" in response:
        # <think> 태그 내용만 추출
        think_content = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
        if think_content:
            think_text = think_content.group(1).strip()

            # "d) 최종 대책:" 이후의 텍스트
            match = re.search(r'd\)\s*최종\s*대책:\s*(.*?)(?:\n\n|$)', think_text, re.DOTALL)
            if match:
                extracted_text = match.group(1).strip()
                if extracted_text.endswith("</think>"):
                    extracted_text = extracted_text[:-9].strip()
                return extracted_text

            # "최종 대책:" 이후의 텍스트
            match = re.search(r'최종\s*대책:\s*(.*?)(?:\n\n|$)', think_text, re.DOTALL)
            if match:
                extracted_text = match.group(1).strip()
                if extracted_text.endswith("</think>"):
                    extracted_text = extracted_text[:-9].strip()
                return extracted_text

            # 그냥 마지막 줄을 반환 (최후의 수단)
            lines = think_text.split('\n')
            non_empty_lines = [line.strip() for line in lines if line.strip()]
            if non_empty_lines:
                extracted_text = non_empty_lines[-1]
                if extracted_text.endswith("</think>"):
                    extracted_text = extracted_text[:-9].strip()
                return extracted_text

    # 5. "d) 최종 대책:" 이후의 텍스트
    match = re.search(r'd\)\s*최종\s*대책:\s*(.*?)(?:\n\n|$)', cleaned_response, re.DOTALL)
    if match:
        extracted_text = match.group(1).strip()
        if extracted_text.endswith("</think>"):
            extracted_text = extracted_text[:-9].strip()
        return extracted_text

    # 6. "최종 대책:" 이후의 텍스트
    match = re.search(r'최종\s*대책:\s*(.*?)(?:\n\n|$)', cleaned_response, re.DOTALL)
    if match:
        extracted_text = match.group(1).strip()
        if extracted_text.endswith("</think>"):
            extracted_text = extracted_text[:-9].strip()
        return extracted_text

    # 7. 가능한 응답이 없는 경우, 전체 응답에서 JSON 형식 및 불필요한 부분을 제거하고 반환
    extracted_text = cleaned_response.strip()
    if extracted_text.endswith("</think>"):
        extracted_text = extracted_text[:-9].strip()
    return extracted_text


# 대책 생성
def generate_prevention_plan(client, accident_info):
    logger.info(f"사고 ID: {accident_info.get('ID', 'N/A')} 처리 시작")
    try:
        result = run_workflow(client, {"accident_info": accident_info})
        response = result["response"]
        prevention_plan = parse_response(response)

        # 결과 파일 저장
        if accident_info.get('ID'):
            text_file_path = os.path.join(TEXT_OUTPUT_DIR, f"{accident_info['ID']}.txt")
            with open(text_file_path, 'w', encoding='utf-8') as f:
                f.write(f"사고 정보:\n")
                for k, v in accident_info.items():
                    f.write(f"{k}: {v}\n")
                f.write("\n전체 응답:\n")
                f.write(response)
                f.write("\n\n최종 추출된 대책:\n")
                f.write(prevention_plan)

        logger.info(f"사고 ID: {accident_info.get('ID', 'N/A')} 처리 완료")
        return prevention_plan
    except Exception as e:
        logger.error(f"대책 생성 오류: {e}")
        return "안전관리 강화 및 위험요소 제거를 통한 사고예방 시스템 구축"


# 텍스트 리스트를 768차원 벡터로 변환 (배치 처리)
def get_sbert_embeddings_batch(text_list):
    """
    텍스트 리스트를 한 번에 처리하여 임베딩 생성

    Args:
        text_list (list): 텍스트 문자열 리스트

    Returns:
        numpy.ndarray: 각 텍스트에 대한 768차원 임베딩 배열 (n_samples, 768)
    """
    logger.info(f"총 {len(text_list)}개 텍스트의 임베딩 생성 시작")

    # 배치 크기 설정 (GPU 메모리에 맞게 조정)
    batch_size = 16
    embeddings_list = []

    # 배치 단위로 처리
    for i in range(0, len(text_list), batch_size):
        batch_texts = text_list[i:i + batch_size]

        # 토크나이징
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {key: val.to(device) for key, val in inputs.items()}

        # 임베딩 계산
        with torch.no_grad():
            outputs = model(**inputs)

        # [CLS] 토큰의 임베딩 사용
        batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings_list.append(batch_embeddings)

    # 모든 배치의 임베딩 합치기
    all_embeddings = np.vstack(embeddings_list)
    logger.info(f"임베딩 생성 완료: {all_embeddings.shape}")

    return all_embeddings


# 테스트 데이터 처리 및 768차원 임베딩 생성
def process_test_data(client):
    logger.info("=" * 50)
    logger.info("건설 안전사고 대응 AI 모델 제출 파일 생성 시작")
    logger.info("=" * 50)
    print("건설 안전사고 대응 AI 모델 제출 파일 생성 시작...")

    # 1. 데이터 로드
    test_df = load_and_preprocess_data(TEST_FILE)

    # 2. 기본 결과 리스트 준비
    results = []
    prevention_plans = []
    ids = []

    # 3. 모든 사고 정보에 대해 대책 생성 (임베딩 생성 없이)
    for idx, (_, row) in enumerate(tqdm(test_df.iterrows(), total=len(test_df),
                                        desc="사고 정보 처리 중",
                                        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')):
        if idx % 10 == 0:
            print(f"\r처리 중: {idx}/{len(test_df)} 완료 ({idx / len(test_df) * 100:.1f}%)", end="")

        accident_info = {
            'ID': row['ID'],
            '인적사고': row['인적사고'],
            '물적사고': row['물적사고'],
            '공사종류': row['공사종류'],
            '공종': row['공종'],
            '사고객체': row['사고객체'],
            '작업프로세스': row['작업프로세스'],
            '사고원인': row['사고원인']
        }

        # 대책 생성
        prevention_plan = generate_prevention_plan(client, accident_info)

        # 결과 저장
        prevention_plans.append(prevention_plan)
        ids.append(row['ID'])

        # 기본 결과 딕셔너리 생성 (임베딩 없이)
        results.append({
            'ID': row['ID'],
            '재발방지대책 및 향후조치계획': prevention_plan
        })

    # 4. 모든 대책에 대해 한 번에 임베딩 생성 (배치 처리)
    print("\n모든 대책에 대한 임베딩 생성 중...")
    embeddings = get_sbert_embeddings_batch(prevention_plans)

    # 5. 임베딩을 결과에 추가
    for i, result in enumerate(results):
        for j in range(768):
            result[f'vec_{j}'] = embeddings[i, j]

    # 6. DataFrame 생성 및 저장
    results_df = pd.DataFrame(results)
    submission_path = os.path.join(OUTPUT_DIR, "submission_with_embeddings.csv")
    results_df.to_csv(submission_path, index=False)

    print(f"\n제출 파일 생성 완료: {submission_path}")
    print(f"총 처리된 사고 정보 수: {len(results_df)}")
    print(f"개별 텍스트 파일은 {TEXT_OUTPUT_DIR} 디렉토리에 저장되었습니다.")
    logger.info(f"제출 파일 생성 완료: {submission_path}")
    logger.info(f"총 처리된 사고 정보 수: {len(results_df)}")

    return results_df