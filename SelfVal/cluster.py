import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModel

import umap
import hdbscan

# -------------------------------------------
# 1) 데이터 불러오기
# -------------------------------------------
file_path = r"C:\Users\tjdwn\GitHub\HansolDeco-Construction-Safety-LLM\train.csv"
df = pd.read_csv(file_path)

# 사용할 텍스트 컬럼
text_cols = ["공사종류", "인적사고", "공종", "사고객체", "작업프로세스", "장소", "재발방지대책 및 향후조치계획"]
df[text_cols] = df[text_cols].fillna("")

def combine_text(row):
    return " ".join(str(row[col]) for col in text_cols)

df["combined_text"] = df.apply(combine_text, axis=1)
text_data = df["combined_text"].tolist()

# -------------------------------------------
# 2) 임베딩 모델 로드
# -------------------------------------------
model_name = "dragonkue/BGE-m3-ko"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# -------------------------------------------
# 3) 배치 임베딩 함수 (tqdm 적용)
# -------------------------------------------
def get_embeddings_in_batch(texts, tokenizer, model, batch_size=16, max_length=128):
    """
    texts: 텍스트 리스트
    tokenizer, model: Transformer 모델
    batch_size: 한 번에 처리할 문장 개수
    """
    all_embeddings = []
    total = len(texts)

    # 배치 단위로 잘라서 추론
    for i in tqdm(range(0, total, batch_size), desc="Embedding Batches"):
        batch_texts = texts[i : i + batch_size]

        # 토크나이징
        inputs = tokenizer(
            batch_texts,
            return_tensors='pt',
            truncation=True,
            max_length=max_length,
            padding='max_length'
        ).to(device)

        # 모델 추론
        with torch.no_grad():
            outputs = model(**inputs)
            # BERT 계열은 일반적으로 [CLS] 벡터 혹은 토큰 임베딩 평균을 사용
            last_hidden_state = outputs.last_hidden_state  # [batch, seq_len, hidden_dim]
            cls_embeddings = last_hidden_state[:, 0, :]    # [CLS]만 뽑아서 사용
            # 필요에 따라 평균 pooling 등 사용 가능
            cls_embeddings = cls_embeddings.cpu().numpy()

        all_embeddings.append(cls_embeddings)

    # np.array로 이어붙이기
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    return all_embeddings

# 실제로 임베딩 계산 (배치추론)
embeddings = get_embeddings_in_batch(
    text_data,
    tokenizer,
    model,
    batch_size=4,    # GPU 메모리에 맞춰 조정
    max_length=128    # 텍스트 길이에 따라 조정
)

# -------------------------------------------
# 4) UMAP으로 차원 축소 (선택적)
# -------------------------------------------
# - 차원 축소로 성능이나 시각화가 더 좋아질 수 있음
# - n_components=5~20 정도로 조정 가능
reducer = umap.UMAP(n_components=5, random_state=42)
embeddings_2d = reducer.fit_transform(embeddings)

# -------------------------------------------
# 5) HDBSCAN 클러스터링
# -------------------------------------------
# - min_cluster_size: 최소 클러스터 크기
# - min_samples: 노이즈 판단에 영향
# - 파라미터를 조정해가며 적절한 결과 확인 가능
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=5,
    min_samples=5,
    metric='euclidean',
    cluster_selection_method='leaf'
)

# 차원 축소된 임베딩으로 클러스터링 (또는 원본 embeddings 사용 가능)
labels = clusterer.fit_predict(embeddings_2d)

# -------------------------------------------
# 6) 결과 저장
# -------------------------------------------
# labels == -1 은 노이즈(어느 클러스터에도 속하지 않음)
df["cluster_label"] = labels
df.to_csv("train_with_hdbscan_cluster.csv", index=False)
print("클러스터링 완료. 결과를 'train_with_hdbscan_cluster.csv'에 저장했습니다.")

# -------------------------------------------
# 7) 클러스터링 요약 확인
# -------------------------------------------
unique_labels, counts = np.unique(labels, return_counts=True)
print("클러스터 레이블 분포:")
for lbl, cnt in zip(unique_labels, counts):
    print(f"  Cluster {lbl}: {cnt}개 샘플")
