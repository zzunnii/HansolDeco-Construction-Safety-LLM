import os
import re
import pandas as pd
from tqdm import tqdm
from langchain.schema import Document
from langchain.text_splitter import MarkdownHeaderTextSplitter
from utils.common import logger
from config import MARKDOWN_DIR, TRAIN_FILE


def extract_safety_guidelines():
    """마크다운 파일에서 안전지침 추출"""
    safety_docs = []
    headers_to_split_on = [("#", "header_1"), ("##", "header_2"), ("###", "header_3")]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

    md_files = [f for f in os.listdir(MARKDOWN_DIR) if f.endswith('.md')]
    for md_file in tqdm(md_files, desc="안전지침 추출 중"):
        file_path = os.path.join(MARKDOWN_DIR, md_file)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"파일 읽기 오류 ({md_file}): {e}")
            continue

        split_docs = markdown_splitter.split_text(content)
        for doc in split_docs:
            metadata = {"source": md_file, "title": md_file.replace('.md', ''), "type": "safety_guideline"}
            metadata.update({k: v for k, v in doc.metadata.items() if k in ["header_1", "header_2", "header_3"]})
            if not any(term_pattern in "".join(metadata.get(h, "") for h in ["header_1", "header_2", "header_3"])
                       for term_pattern in ["용어", "정의"]):
                safety_docs.append(Document(page_content=doc.page_content, metadata=metadata))

    logger.info(f"총 {len(safety_docs)}개의 안전지침 청크 추출")
    return safety_docs


def extract_term_definitions():
    """마크다운 파일에서 용어 정의 추출"""
    term_docs = []
    section_patterns = [
        r'#\s*3\.\s*용어의\s*정의.*?(?=\n#|\Z)', r'^3\.\s*용어의\s*정의.*?(?=\n#|\Z)',
        r'#\s*3\.\s*정의.*?(?=\n#|\Z)', r'^3\.\s*정의.*?(?=\n#|\Z)', r'##\s*3\s*용어의\s*정의.*?(?=\n#|\Z)'
    ]
    term_entry_pattern = r'-\s*\(([가-힣]|[0-9])\)\s*[\'\"]*([^\'\"]+)[\'\"]*(?:이|란|은|는)[^\n.]*\.'

    md_files = [f for f in os.listdir(MARKDOWN_DIR) if f.endswith('.md')]
    for md_file in tqdm(md_files, desc="용어 정의 추출 중"):
        # 용어 정의 추출 로직...
        file_path = os.path.join(MARKDOWN_DIR, md_file)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"파일 읽기 오류 ({md_file}): {e}")
            continue

        term_section = next(
            (match for pattern in section_patterns if (match := re.search(pattern, content, re.DOTALL | re.MULTILINE))),
            None)
        if term_section:
            term_section = term_section.group(0)
            term_entries = re.findall(term_entry_pattern, term_section, re.DOTALL)
            full_entries = re.split(r'-\s*\((?:[가-힣]|[0-9])\)', term_section)[1:]
            for i, (item_num, term) in enumerate(term_entries):
                if i < len(full_entries):
                    definition_text = re.match(r'.*?\.', full_entries[i].strip()) or full_entries[i].strip()
                    term_docs.append(Document(
                        page_content=f"용어: {term.strip()}\n정의: {definition_text.group(0).strip() if hasattr(definition_text, 'group') else definition_text}",
                        metadata={"source": md_file, "term": term.strip(), "item_number": item_num,
                                  "type": "term_definition"}
                    ))

    logger.info(f"총 {len(term_docs)}개의 용어 정의 추출")
    return term_docs


def extract_similar_cases():
    """train.csv에서 유사 사례 추출"""
    if not os.path.exists(TRAIN_FILE):
        logger.warning(f"경고: {TRAIN_FILE} 존재하지 않음")
        return []

    try:
        train_df = pd.read_csv(TRAIN_FILE, encoding="utf-8").fillna("정보 없음")
    except Exception as e:
        logger.error(f"CSV 로드 오류: {e}")
        return []

    # 데이터 전처리 및 사례 추출 로직...
    train_df['공사종류(대분류)'] = train_df['공사종류'].str.split(' / ').str[0]
    train_df['공사종류(중분류)'] = train_df['공사종류'].str.split(' / ').str[1]
    train_df['공종(대분류)'] = train_df['공종'].str.split(' > ').str[0]
    train_df['공종(중분류)'] = train_df['공종'].str.split(' > ').str[1]
    train_df['사고객체(대분류)'] = train_df['사고객체'].str.split(' > ').str[0]
    train_df['사고객체(중분류)'] = train_df['사고객체'].str.split(' > ').str[1]

    case_docs = []
    for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="유사 사례 추출 중"):
        accident_info = (
            f"인적사고: {row['인적사고']}\n물적사고: {row['물적사고']}\n"
            f"공사종류: {row['공사종류(대분류)']} > {row['공사종류(중분류)']}\n"
            f"공종: {row['공종(대분류)']} > {row['공종(중분류)']}\n"
            f"사고객체: {row['사고객체(대분류)']} ({row['사고객체(중분류)']})\n"
            f"작업 프로세스: {row['작업프로세스']}\n사고 원인: {row['사고원인']}\n"
            f"재발방지대책 및 향후조치계획: {row.get('재발방지대책 및 향후조치계획', '정보 없음')}"
        )
        case_docs.append(Document(
            page_content=accident_info,
            metadata={
                "ID": row["ID"], "type": "accident", "인적사고": row["인적사고"], "물적사고": row["물적사고"],
                "공사종류": row["공사종류"], "공종": row["공종"], "사고객체": row["사고객체"], "작업프로세스": row["작업프로세스"]
            }
        ))

    logger.info(f"총 {len(case_docs)}개의 유사 사례 추출")
    return case_docs