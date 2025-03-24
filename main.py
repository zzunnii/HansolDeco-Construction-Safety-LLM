import os
import argparse
import json
from utils.common import set_seed, logger, fixed_seed
from workflow.models import initialize_client
from vectordb.vectordb import create_or_load_vectorstores
from submission.processing import process_test_data
from workflow.workflow import run_workflow


def test_single_case(client, args):
    """단일 사고 분석 테스트 실행"""
    accident_info = {}

    # JSON 파일에서 사고 정보 로드
    if args.input:
        if not os.path.exists(args.input):
            print(f"오류: 입력 파일 '{args.input}'이 존재하지 않습니다.")
            return

        try:
            with open(args.input, 'r', encoding='utf-8') as f:
                accident_info = json.load(f)
            print(f"'{args.input}' 파일에서 사고 정보를 로드했습니다.")
        except Exception as e:
            print(f"JSON 파일 로드 오류: {e}")
            return

    # 대화형 모드로 사고 정보 입력
    elif args.interactive:
        print("\n=== 사고 정보 입력 모드 ===")
        print("각 항목을 순서대로 입력해주세요. 입력을 건너뛰려면 Enter를 누르세요.")

        fields = [
            ("인적사고", "인적사고 유형 (예: 떨어짐, 부딪힘, 넘어짐 등)"),
            ("물적사고", "물적사고 유형 (예: 붕괴, 전도, 없음 등)"),
            ("공사종류", "공사종류 (예: 건축 / 건축물 / 공동주택)"),
            ("공종", "공종 (예: 건축 > 철근콘크리트공사)"),
            ("사고객체", "사고객체 (예: 건설기계 > 콘크리트펌프)"),
            ("작업프로세스", "작업프로세스 (예: 타설작업, 이동, 해체작업 등)"),
            ("사고원인", "사고원인 (상세히 작성)")
        ]

        for field, description in fields:
            value = input(f"{description}: ")
            if value.strip():
                accident_info[field] = value.strip()
            else:
                accident_info[field] = "정보 없음"

    # 입력 방식 선택되지 않음
    else:
        print("오류: --input 파일 경로나 --interactive 중 하나를 선택해야 합니다.")
        return

    # 사고 정보 확인
    if not accident_info or "사고원인" not in accident_info:
        print("오류: 사고 정보가 충분하지 않습니다. 최소한 '사고원인'은 입력해야 합니다.")
        return

    # 필요한 필드가 없을 경우 기본값 설정
    required_fields = ["인적사고", "물적사고", "공사종류", "공종", "사고객체", "작업프로세스", "사고원인"]
    for field in required_fields:
        if field not in accident_info:
            accident_info[field] = "정보 없음"

    # 사고 분석 실행
    print("\n=== 사고 정보 ===")
    for field, value in accident_info.items():
        print(f"{field}: {value}")

    print("\n사고 분석 시작...")
    result = run_workflow(client, {"accident_info": accident_info})

    # 분석 결과 표시
    response = result["response"]
    parsed = result["parsed_result"]

    print("\n=== 분석 결과 ===")

    # 용어 확인 결과
    term_check = "모두 이해 완료" if "모두 이해 완료" in response else "모르는 용어 발견"
    print(f"\n=== 사고 정보 분석 ===\n{term_check}")

    # 심각도 및 원인 추출
    severity_match = re.search(r'a\).*?\{"answer": "(\w)"\}', response)
    cause_match = re.search(r'b\).*?\{"answer": "(\w)"\}', response)

    severity = severity_match.group(1) if severity_match else "C"
    cause = cause_match.group(1) if cause_match else "A"

    severity_text = {"A": "치명적", "B": "중대", "C": "경미"}.get(severity, "경미")
    cause_text = {
        "A": "구체적/직접적",
        "B": "인적",
        "C": "구조적/기술적",
        "D": "환경적"
    }.get(cause, "구체적/직접적")

    # 심각도 및 원인 설명 추출
    severity_reason = re.search(r'a\).*?\{\"answer\": \"[A-C]\"\}(.*?)b\)', response, re.DOTALL)
    cause_reason = re.search(r'b\).*?\{\"answer\": \"[A-D]\"\}(.*?)c\)', response, re.DOTALL)

    severity_reason_text = severity_reason.group(1).strip() if severity_reason else ""
    cause_reason_text = cause_reason.group(1).strip() if cause_reason else ""

    # 결과 출력
    print(f"\n=== 사고 심각도 판단 ===\n심각도: {severity} ({severity_text})")
    if severity_reason_text:
        print(f"이유: {severity_reason_text.split('.')[0]}")

    print(f"\n=== 사고 원인 분류 ===\n원인 유형: {cause} ({cause_text})")
    if cause_reason_text:
        print(f"이유: {cause_reason_text.split('.')[0]}")

    # 맞춤 대책 섹션 추출
    measures_section = re.search(r'c\)(.*?)d\)', response, re.DOTALL)
    if measures_section:
        measures_text = measures_section.group(1).strip()
        print(f"\n=== 맞춤 대책 ===\n{measures_text}")

    # 최종 대책 출력
    final_plan = parsed.get("final_plan", "")
    print(f"\n=== 최종 대책 ===\n{final_plan}")

    # 로그 저장
    log_dir = "./logs/single_test"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/test_{timestamp}.txt"

    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("=== 사고 정보 ===\n")
        for field, value in accident_info.items():
            f.write(f"{field}: {value}\n")

        f.write("\n=== 전체 응답 ===\n")
        f.write(response)

        f.write("\n\n=== 최종 대책 ===\n")
        f.write(final_plan)

    print(f"\n분석 결과가 '{log_file}'에 저장되었습니다.")


def main():
    """
    건설 안전사고 대응 AI 시스템의 메인 실행 함수
    """
    # 명령행 인자 파싱
    parser = argparse.ArgumentParser(description='건설 안전사고 대응 AI 시스템')
    parser.add_argument('--build_vectordb', action='store_true', help='벡터 DB 구축 실행')
    parser.add_argument('--generate_submission', action='store_true', help='제출 파일 생성 실행')
    parser.add_argument('--all', action='store_true', help='모든 단계 실행')

    # 단일 사고 테스트 모드 관련 인자
    parser.add_argument('--test_single', action='store_true', help='단일 사고 테스트 모드')
    parser.add_argument('--input', type=str, help='사고 정보 JSON 파일 경로')
    parser.add_argument('--interactive', action='store_true', help='대화형 사고 정보 입력 모드')

    args = parser.parse_args()

    set_seed(fixed_seed)

    # 벡터 데이터베이스 구축 (필요시)
    if args.build_vectordb or args.all:
        logger.info("벡터 데이터베이스 구축 시작")
        guidelines_db, terms_db, cases_db = create_or_load_vectorstores()
        logger.info("벡터 데이터베이스 구축 완료")

    # vLLM 클라이언트 초기화
    client = initialize_client()

    # 단일 사고 테스트 모드
    if args.test_single:
        test_single_case(client, args)
        return

    # 테스트 데이터 처리 및 제출 파일 생성 (필요시)
    if args.generate_submission or args.all:
        results_df = process_test_data(client)
        print(f"생성된 결과 크기: {results_df.shape}")
        print(f"개별 텍스트 파일은 TEXT_OUTPUT_DIR 디렉토리에 저장되었습니다.")
        print("\n== 처리 완료된 첫 5개 결과 ==")
        for idx, row in results_df[['ID', '재발방지대책 및 향후조치계획']].head().iterrows():
            print(f"ID: {row['ID']}")
            print(f"대책: {row['재발방지대책 및 향후조치계획'][:100]}..." if len(row['재발방지대책 및 향후조치계획']) > 100 else row[
                '재발방지대책 및 향후조치계획'])
            print("-" * 50)

    logger.info("프로세스 완료")
    return 0


if __name__ == "__main__":
    import re
    from datetime import datetime

    main()