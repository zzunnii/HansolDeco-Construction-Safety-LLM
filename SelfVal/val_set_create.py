import pandas as pd

# 1) 이전에 만든 클러스터링 결과 파일 불러오기
df = pd.read_csv("train_with_hdbscan_cluster.csv")

# 2) 클러스터별 샘플 수 계산
cluster_counts = df.groupby("cluster_label")["ID"].count()

# 3) 20개 이상인 클러스터만 추려내기
valid_clusters = cluster_counts[cluster_counts >= 20].index.tolist()

# 4) 각 클러스터에서 대표 1건씩 추출
selected_rows = []
for clus in valid_clusters:
    # 해당 클러스터만 추출
    cluster_df = df[df["cluster_label"] == clus]
    # 대표 1건 (첫 번째 샘플)
    row = cluster_df.iloc[0]
    selected_rows.append(row)

# 5) DataFrame으로 다시 만들고 cluster_label 기준 정렬
selected_df = pd.DataFrame(selected_rows)
selected_df.sort_values(by="cluster_label", inplace=True)

desired_cols = [
    "cluster_label",
    "ID",
    "인적사고",
    "물적사고",
    "공사종류",
    "공종",
    "사고객체",
    "작업프로세스",
    "사고원인",
    "재발방지대책 및 향후조치계획",
]

# 실제 df에 존재하지 않는 컬럼이 있을 수 있으므로 교차 처리
final_cols = [c for c in desired_cols if c in selected_df.columns]
selected_df = selected_df[final_cols]

# 7) val_set.csv로 저장 (UTF-8 인코딩)
selected_df.to_csv("val_set.csv", index=False, encoding="utf-8-sig")

print("'val_set.csv'로 저장!")
