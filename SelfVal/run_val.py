import os
import pandas as pd
from tqdm import tqdm
from utils.common import logger, parse_response
from workflow.workflow import run_workflow
from config import VAL_SET, OUTPUT_DIR
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from workflow.models import initialize_client

client = initialize_client()

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
    return embeddings


# 자카드 유사도 계산
def jaccard_similarity(text1, text2):
    set1 = set(text1.split())
    set2 = set(text2.split())
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0


# 대책 생성
def generate_prevention_plan(client, accident_info):
    logger.info(f"사고 ID: {accident_info.get('ID', 'N/A')} 처리 시작")
    try:
        result = run_workflow(client, {"accident_info": accident_info})
        response = result["response"]
        prevention_plan = parse_response(response)
        logger.info(f"사고 ID: {accident_info.get('ID', 'N/A')} 처리 완료")
        return prevention_plan
    except Exception as e:
        logger.error(f"대책 생성 오류: {e}")
        return "안전관리 강화 및 위험요소 제거를 통한 사고예방 시스템 구축"


# 유사도 기반 스코어 계산
def calculate_score(pred_text, gt_text):
    # 768차원 벡터로 변환
    pred_embedding = get_sbert_embedding(pred_text)
    gt_embedding = get_sbert_embedding(gt_text)

    # 코사인 유사도 계산
    cos_sim = cosine_similarity(pred_embedding, gt_embedding)[0][0]

    # 자카드 유사도 계산
    jac_sim = jaccard_similarity(pred_text, gt_text)

    # 최종 스코어: 0.7 * 코사인 유사도 + 0.3 * 자카드 유사도
    final_score = 0.7 * cos_sim + 0.3 * jac_sim
    return final_score, cos_sim, jac_sim


# 테스트 데이터 처리 및 평가
def process_and_evaluate(client):
    logger.info("=" * 50)
    logger.info("건설 안전사고 대응 AI 모델 평가 시작")
    logger.info("=" * 50)
    print("건설 안전사고 대응 AI 모델 평가 시작...")

    test_df = load_and_preprocess_data(VAL_SET)
    results = []
    scores = []

    for idx, (_, row) in enumerate(tqdm(test_df.iterrows(), total=len(test_df),
                                        desc="사고 정보 처리 및 평가 중",
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

        # 모델이 생성한 대책
        pred_plan = generate_prevention_plan(client, accident_info)

        # Ground Truth 대책 (val_set에서 가져옴)
        gt_plan = row['재발방지대책 및 향후조치계획']

        # 유사도 스코어 계산
        final_score, cos_sim, jac_sim = calculate_score(pred_plan, gt_plan)

        results.append({
            'ID': row['ID'],
            'Predicted': pred_plan,
            'Ground_Truth': gt_plan,
            'Final_Score': final_score,
            'Cosine_Similarity': cos_sim,
            'Jaccard_Similarity': jac_sim
        })
        scores.append(final_score)

        logger.info(
            f"ID: {row['ID']} | Final Score: {final_score:.4f} | Cosine: {cos_sim:.4f} | Jaccard: {jac_sim:.4f}")

    # 결과 DataFrame 생성
    results_df = pd.DataFrame(results)

    # 평균 스코어 계산
    avg_score = np.mean(scores)
    print(f"\n평가 완료!")
    print(f"총 처리된 사고 정보 수: {len(results_df)}")
    print(f"평균 스코어: {avg_score:.4f}")
    logger.info(f"평가 완료: 평균 스코어 {avg_score:.4f}")

    # 결과 일부 출력
    print("\n== 평가 결과 첫 5개 샘플 ==")
    for idx, row in results_df.head().iterrows():
        print(f"ID: {row['ID']}")
        print(f"Predicted: {row['Predicted'][:100]}..." if len(row['Predicted']) > 100 else row['Predicted'])
        print(
            f"Ground Truth: {row['Ground_Truth'][:100]}..." if len(row['Ground_Truth']) > 100 else row['Ground_Truth'])
        print(
            f"Final Score: {row['Final_Score']:.4f} (Cosine: {row['Cosine_Similarity']:.4f}, Jaccard: {row['Jaccard_Similarity']:.4f})")
        print("-" * 50)

    return results_df


if __name__ == "__main__":
    results_df = process_and_evaluate(client)