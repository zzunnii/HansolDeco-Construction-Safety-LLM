import re
import os
from tqdm import tqdm

# Markdown 파일이 저장된 폴더
markdown_dir = "./markdown"
os.makedirs(markdown_dir, exist_ok=True)
# 특정 단어만 제거할 패턴 목록
remove_only_pattern = [
    r"KOSHA Guide",  # "KOSHA Guide" 부분만 제거
    r"KOSHA GUIDE",  # 대문자 버전 포함
]

# 특정 패턴이 단독으로 있는 경우 해당 줄 전체 삭제
remove_full_line_patterns = [
    r"[A-Z] *- *\d{1,3} *- *\d{4}",  # "C - 56 - 2017" 같은 형식이 단독으로 있으면 삭제
]

# `##`, `###` 헤더에 특정 문자열 포함 시 헤더 전체 삭제
header_remove_pattern = re.compile(r"^(#|##|###) (.*KOSHA Guide.*|.*KOSHA GUIDE.*)")

# `##`, `###`만 단독으로 있는 경우 삭제
empty_header_pattern = re.compile(r"^(#|##|###)\s*$")

# 마크다운 폴더 내 모든 `.md` 파일 가져오기
md_files = [f for f in os.listdir(markdown_dir) if f.endswith('.md')]

# tqdm을 사용하여 진행 상황 표시
for md_file in tqdm(md_files, desc="Markdown 후처리", unit="파일"):
    md_path = os.path.join(markdown_dir, md_file)

    # 파일 읽기
    with open(md_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    processed_lines = []
    for line in lines:
        original_line = line.strip()  # 공백 제거

        # 특정 패턴이 포함된 헤더(`##`, `###`)는 전체 삭제
        if header_remove_pattern.match(original_line):
            continue  # 헤더 삭제

        # 특정 패턴이 단독으로 있는 경우 해당 줄 삭제
        if any(re.fullmatch(pattern, original_line) for pattern in remove_full_line_patterns):
            continue  # 해당 줄 제거

        # `##`, `###`만 단독으로 있는 경우 삭제
        if empty_header_pattern.fullmatch(original_line):
            continue  # 빈 헤더 삭제

        # 특정 패턴이 포함된 문자열은 해당 패턴만 삭제하고 나머지는 유지
        for pattern in remove_only_pattern:
            original_line = re.sub(pattern, "", original_line).strip()

        # 빈 줄이 되면 추가하지 않음
        if original_line:
            processed_lines.append(original_line + "\n")

    # 파일 덮어쓰기
    with open(md_path, "w", encoding="utf-8") as f:
        f.writelines(processed_lines)

print("\nMarkdown 후처리 완료!")
