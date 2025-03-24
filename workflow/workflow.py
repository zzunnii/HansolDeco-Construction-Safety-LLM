import re
from typing import Dict, TypedDict
from langgraph.graph import StateGraph, START, END
from workflow.models import call_model
from vectordb.retriever import GuidelineRetriever, TermRetriever, CaseRetriever
from utils.common import logger, parse_response


# 랭그래프 상태 정의
class GraphState(TypedDict):
    accident_info: Dict[str, str]
    final_response: str
    parsed_result: Dict[str, str]
    conversation_history: list


# 사고 분석 및 응답 생성 노드
def process_and_generate_response(state: GraphState, client):
    logger.info("사고 분석 및 대책 생성 시작")
    accident_info = state.get("accident_info", {})
    accident_info_str = "\n".join([f"{k}: {v}" for k, v in accident_info.items()])
    history = state.get("conversation_history", [])

    # Step 1: 용어 확인
    prompt_term_check = f"""**사고 정보 분석**
아래 사고 정보를 검토하고, 모르는 건설 용어가 있으면 [해당 단어] 형식으로 표시해.
모든 건설 용어를 이해하면 "모두 이해 완료"라고만 답변해. 분석이나 대책은 제시하지 마.
---
{accident_info_str}
---"""
    term_result = call_model(client, prompt_term_check, history=history)
    history.append({"role": "user", "content": prompt_term_check})
    history.append({"role": "assistant", "content": term_result["response"], "thought": term_result["thought"]})

    unknown_terms = [] if "모두 이해 완료" in term_result["response"] else re.findall(r'\[(.*?)\]', term_result["response"])
    logger.info(f"식별된 모르는 용어: {unknown_terms}")

    # 용어 정의 검색
    term_retriever = TermRetriever()
    definitions = {term: term_retriever.get_definition(term) for term in unknown_terms}
    definitions_text = "\n".join([f"- {term}: {definitions[term]}" for term in definitions]) if definitions else "없음"

    # Step 2: 분석 및 대책 생성
    guideline_retriever = GuidelineRetriever()
    case_retriever = CaseRetriever()
    safety_guidelines = guideline_retriever.search(accident_info, k=2)
    similar_cases = case_retriever.search(accident_info_str)

    prompt_analysis = f"""**최종 분석 및 대책 수립**
### 사고 정보
{accident_info_str}
### 용어 정의
{definitions_text}
### 관련 안전 지침
{safety_guidelines}
### 유사 사례
{similar_cases}
### 지시사항
<think>
Please reason step by step through the following sub-steps, ensuring logical consistency and practical outcomes based solely on the provided data.
</think>
a) 사고의 심각도 판단: A: 치명적, B: 중대, C: 경미 중 선택하고 이유 설명. {{"answer": "[선택한 문자]"}}
b) 사고 원인 분류: A: 구체적/직접적, B: 인적, C: 구조적/기술적, D: 환경적 중 선택. {{"answer": "[선택한 문자]"}}
c) 심각도에 따른 맞춤 대책 제시:
   * 치명적 사고: 매우 구체적이고 상세한 대책 3가지 이상
   * 중대 사고: 구체적인 실행 방법이 포함된 대책 2가지
   * 경미한 사고: 포괄적이지만 실행 가능한 대책 1가지
d) 최종 대책:
   \\boxed{{[200자로 요약된 명사형으로 끝나는 재발 방지 대책 및 향후조치계획]}}
"""
    analysis_result = call_model(client, prompt_analysis, history=history)
    history.append({"role": "user", "content": prompt_analysis})
    history.append({"role": "assistant", "content": analysis_result["response"], "thought": analysis_result["thought"]})

    final_plan = parse_response(analysis_result["response"])

    return {
        **state,
        "final_response": analysis_result["response"],
        "parsed_result": {"final_plan": final_plan},
        "conversation_history": history
    }


# 워크플로우 생성
def create_workflow(client):
    logger.info("워크플로우 구성 시작")
    workflow = StateGraph(GraphState)
    workflow.add_node("process_and_generate_response", lambda state: process_and_generate_response(state, client))
    workflow.add_edge(START, "process_and_generate_response")
    workflow.add_edge("process_and_generate_response", END)
    logger.info("워크플로우 컴파일 완료")
    return workflow.compile()


# 워크플로우 실행 (각 사고마다 초기화)
def run_workflow(client, input_data):
    logger.info(f"워크플로우 실행 시작: {input_data}")
    try:
        # 각 사고마다 새로운 상태로 초기화
        initial_state = {
            "accident_info": input_data["accident_info"],
            "conversation_history": []  # 히스토리 초기화
        }
        graph = create_workflow(client)
        result = graph.invoke(initial_state, config={"recursion_limit": 50})
        logger.info("워크플로우 실행 완료")
        return {
            "response": result.get("final_response", "응답을 생성할 수 없습니다."),
            "parsed_result": result.get("parsed_result", {"final_plan": "기본 안전관리 강화"}),
            "conversation_history": result.get("conversation_history", [])  # 추가
        }

    except Exception as e:
        logger.error(f"워크플로우 실행 오류: {e}")
        return {
            "response": "워크플로우 실행 실패",
            "parsed_result": {"final_plan": "기본 안전관리 강화"}
        }