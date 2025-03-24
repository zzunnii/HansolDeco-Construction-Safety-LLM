import os
from tqdm import tqdm
import pymupdf
import fitz
import traceback
import re
import unicodedata
import glob
from collections import defaultdict
from pathlib import Path

from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from typing import Annotated, List, TypedDict, Literal
from langgraph.graph.message import add_messages
from langchain_core.documents import Document

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder


# PDF가 있는 폴더와 변환된 Markdown을 저장할 폴더
pdf_dir = "./architecturePDF"  # 건설안전지침 데이터 경로
markdown_dir = "./markdown"

# Markdown 폴더 생성
os.makedirs(markdown_dir, exist_ok=True)

# GPU 아키텍처 설정
os.environ['TORCH_CUDA_ARCH_LIST'] = "8.6"  # 사용 GPU 아키텍처에 맞게 수정

# PDF 디렉토리 내 모든 PDF 파일 가져오기
pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]

# 오류 로그를 저장할 파일
error_log_path = os.path.join(".", "pdf_conversion_errors.log")


def improve_headers(markdown_text):
    """마크다운 텍스트에서 헤더 구조를 개선하는 함수"""
    # 1. 줄 단위로 분할
    lines = markdown_text.split('\n')
    improved_lines = []

    # 숫자 형식 패턴 - 헤더인지 확인하는 데 사용
    chapter_pattern = re.compile(r'^##\s+(\d+\.)\s+(.+)$')  # 1. 제목
    section_pattern = re.compile(r'^##\s+(\d+\.\d+\.)\s+(.+)$')  # 1.1. 제목
    subsection_pattern = re.compile(r'^##\s+(\d+\.\d+\.\d+\.)\s+(.+)$')  # 1.1.1. 제목

    for line in lines:
        # 헤더 수준 개선
        if line.startswith('## '):
            # 챕터 (1. 제목)
            chapter_match = chapter_pattern.match(line)
            if chapter_match:
                improved_lines.append(f"# {chapter_match.group(1)} {chapter_match.group(2)}")
                continue

            # 섹션 (1.1. 제목)
            section_match = section_pattern.match(line)
            if section_match:
                improved_lines.append(f"## {section_match.group(1)} {section_match.group(2)}")
                continue

            # 서브섹션 (1.1.1. 제목)
            subsection_match = subsection_pattern.match(line)
            if subsection_match:
                improved_lines.append(f"### {subsection_match.group(1)} {subsection_match.group(2)}")
                continue

            # 특수 헤더 식별 (모두 대문자이거나 특수 키워드 포함)
            text = line[3:].strip()
            if text.isupper() or any(keyword in text for keyword in ['목적', '적용범위', '용어정의', '참고문헌', '부록']):
                improved_lines.append(f"# {text}")
            elif len(text) < 50 and not text.endswith('.'):  # 짧은 문장이고 마침표로 끝나지 않으면 보통 제목
                improved_lines.append(f"## {text}")
            else:
                improved_lines.append(line)  # 기존 헤더 유지
        else:
            improved_lines.append(line)

    return '\n'.join(improved_lines)


# tqdm을 사용하여 진행 상황 표시
for pdf_file in tqdm(pdf_files, desc="PDF transform", unit="file"):
    pdf_path = os.path.join(pdf_dir, pdf_file)

    try:
        # LangChain DoclingLoader로 PDF -> Markdown 변환
        loader = DoclingLoader(
            file_path=pdf_path,
            export_type=ExportType.MARKDOWN
        )
        docs = loader.load()

        # 마크다운 내용 수집 및 개선
        improved_content = ""
        for doc in docs:
            # 각 문서 내용 개선
            improved_content += improve_headers(doc.page_content) + "\n\n"

        # 파일 이름에서 확장자를 제외한 부분을 제목으로 사용
        title = os.path.splitext(pdf_file)[0]
        final_content = f"# {title}\n\n{improved_content}"

        # 변환된 텍스트 저장
        markdown_path = os.path.join(markdown_dir, pdf_file.replace('.pdf', '.md'))
        with open(markdown_path, "w", encoding="utf-8") as f:
            f.write(final_content)

    except Exception as e:
        # 오류 발생 시 로그 파일에 기록
        with open(error_log_path, "a", encoding="utf-8") as error_file:
            error_file.write(f"Error converting {pdf_file}:\n")
            error_file.write(traceback.format_exc())
            error_file.write("\n" + "=" * 50 + "\n")
        print(f"Error converting {pdf_file}. See log for details.")

        # 오류가 발생해도 빈 마크다운 파일 생성
        markdown_path = os.path.join(markdown_dir, pdf_file.replace('.pdf', '.md'))
        with open(markdown_path, "w", encoding="utf-8") as f:
            f.write(f"# Conversion Error\n\nFailed to convert {pdf_file}. See error log for details.\n")

print("Conversion process completed")

# PDF가 있는 폴더와 변환된 Markdown을 저장할 폴더
pdf_dir = Path("./arc")  # 상대 경로로 변경
markdown_dir = Path("./markdown")
markdown_dir.mkdir(exist_ok=True)  # Markdown 폴더 생성

# 오류 로그 파일
error_log_path = Path("./pdf_conversion_errors.log")

# 문제가 있는 파일 목록 (정규화 적용)
problematic_files = [
    "타워크레인 설치, 조립, 해체 작업계획서 작성지침.pdf",
    "강관비계 안전작업지침.pdf",
    "파이프 서포트 동바리 안전작업 지침.pdf",
    "초고층 건축물공사(화재예방) 안전보건작업지침.pdf",
    "시스템폼(RCS폼,ACS폼 중심) 안전작업 지침.pdf",
    "F.C.M 교량공사 안전보건작업 지침.pdf",
    "시스템 동바리 안전작업 지침.pdf",
    "금속 커튼월(Curtain wall) 안전작업 지침.pdf",
    "타기, 항발기 사용 작업계획서 작성지침.pdf",
    "탑다운(Top down) 공법 안전작업 지침.pdf",
    '블록식 보강토 옹벽 공사 안전보건작업 지침.pdf'
]


# 한글 파일명 정규화 (NFC/NFD 문제 해결)
def normalize_filename(filename, form="NFC"):
    return unicodedata.normalize(form, filename)


# PDF를 구조적인 마크다운으로 변환하는 함수 (기존과 동일)
def convert_pdf_to_structured_markdown(pdf_path: Path, markdown_path: Path):
    doc = fitz.open(str(pdf_path))
    markdown_content = [f"# {pdf_path.stem}\n"]
    font_stats = defaultdict(int)
    font_sizes = []

    # 첫 번째 패스: 폰트 통계 수집
    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        font_size = span["size"]
                        font_name = span["font"]
                        is_bold = "bold" in font_name.lower() or span["flags"] & 2 > 0
                        font_key = (font_size, is_bold)
                        font_stats[font_key] += 1
                        font_sizes.append(font_size)

    avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 11
    max_font_size = max(font_sizes) if font_sizes else 12

    # 두 번째 패스: 마크다운 변환
    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        current_block_text = []

        for block in blocks:
            if "lines" not in block:
                continue
            block_text = ""
            previous_y = None
            for line in block["lines"]:
                line_text = ""
                is_header = False
                font_size = 0
                is_bold = False

                for span in line["spans"]:
                    line_text += span["text"]
                    font_size = max(font_size, span["size"])
                    is_bold = is_bold or "bold" in span["font"].lower() or span["flags"] & 2 > 0

                if previous_y is not None and line["bbox"][1] - previous_y > 1.5 * avg_font_size:
                    block_text += "\n\n"
                elif previous_y is not None:
                    block_text += " "

                is_header = (
                        font_size > avg_font_size * 1.2 or
                        is_bold and font_size > avg_font_size * 1.1 or
                        font_size > max_font_size * 0.9 or
                        re.match(r"^\s*\d+(\.\d+)*\s+[A-Za-z가-힣]", line_text) or
                        re.match(r"^\s*[제장항절]\s*\d+", line_text)
                )

                if is_header:
                    header_level = 2
                    if font_size > max_font_size * 0.95:
                        header_level = 2
                    elif font_size > avg_font_size * 1.5:
                        header_level = 3
                    elif font_size > avg_font_size * 1.2 or is_bold:
                        header_level = 4
                    else:
                        header_level = 5

                    if re.match(r"^\s*\d+\.\d+\.\d+", line_text):
                        header_level = 4
                    elif re.match(r"^\s*\d+\.\d+", line_text):
                        header_level = 3
                    elif re.match(r"^\s*\d+\s+", line_text) or re.match(r"^\s*[제장]\s*\d+", line_text):
                        header_level = 2

                    block_text += f"\n\n{'#' * header_level} {line_text.strip()}\n\n"
                else:
                    block_text += line_text

                previous_y = line["bbox"][3]

            if block_text.strip():
                current_block_text.append(block_text)

        if current_block_text:
            page_text = "\n".join(current_block_text)
            page_text = re.sub(r'\n{3,}', '\n\n', page_text)
            markdown_content.append(page_text)

    markdown_path.write_text("\n".join(markdown_content), encoding="utf-8")
    return True


# GLOB 패턴을 사용하여 problematic_files에 있는 파일만 처리
processed_files = []
for problem_file in tqdm(problematic_files, desc="Finding problematic files", unit="file"):
    file_base = problem_file.replace('.pdf', '')

    # NFC와 NFD 모두 시도
    for norm_form in ["NFC", "NFD"]:
        normalized_file = normalize_filename(file_base, norm_form)
        pattern = f"{pdf_dir}/*{normalized_file}*.pdf"
        matching_files = glob.glob(pattern)

        if matching_files:
            break  # 매칭되면 더 이상 NFD/NFC 시도 안 함

    if matching_files:
        for pdf_path in matching_files:
            pdf_filename = os.path.basename(pdf_path)
            markdown_path = markdown_dir / pdf_filename.replace('.pdf', '.md')

            print(f"🔍 Found file: {pdf_filename} (Normalization: {norm_form})")

            if pdf_filename in processed_files:
                print(f"⏩ Already processed: {pdf_filename}")
                continue

            processed_files.append(pdf_filename)

            try:
                print(f"Processing: {pdf_filename}")
                success = convert_pdf_to_structured_markdown(Path(pdf_path), markdown_path)
                if success:
                    print(f"converted: {pdf_filename}")
                else:
                    print(f"Failed to convert: {pdf_filename}")
            except Exception as e:
                with error_log_path.open("a", encoding="utf-8") as error_file:
                    error_file.write(f"Error converting {pdf_filename}: {str(e)}\n")
                    error_file.write(traceback.format_exc())
                    error_file.write("\n" + "=" * 50 + "\n")
                print(f"Error converting {pdf_filename}. See log for details.")
    else:
        print(f"No matching file found for: {problem_file}")
        print(f"  - Tried patterns: {pdf_dir}/*{normalize_filename(file_base, 'NFC')}*.pdf")
        print(f"                  : {pdf_dir}/*{normalize_filename(file_base, 'NFD')}*.pdf")
        # 디렉토리 내 파일 목록 확인
        all_files = glob.glob(f"{pdf_dir}/*.pdf")
        print(f"  - Files in directory: {[os.path.basename(f) for f in all_files]}")

print("모든 PDF 파일 변환 완료")
