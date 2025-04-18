# 건설 안전사고 대응 및 재발방지 대책 AI 시스템

## 목차 (빠른 이동)
- [프로젝트 개요](#overview)
- [시스템 아키텍처](#architecture)
- [주요 기능](#features)
- [프롬프트 엔지니어링](#prompts)
- [기술 스택](#tech-stack)
- [프로젝트 구조](#structure)
- [리더보드 재현](#leaderboard)
- [단일사고 & 실제 사용](#single-case)
- [출력 예시](#models-think)
---

## 데이터셋
**Dacon: 건설공사 사고 예방 및 대응책 생성 :**[한솔데코 시즌3 AI 경진대회](https://dacon.io/competitions/official/236455/data)

## 프로젝트 개요

이 프로젝트는 건설 현장에서 발생하는 안전사고에 대한 분석과 재발방지 대책을 자동으로 생성하는 AI 시스템을 개발한 것입니다. RAG(Retrieval-Augmented Generation) 기술과 LangGraph를 활용한 워크플로우 기반 접근법으로, 건설 현장의 안전성을 강화하고 효과적인 사고 대응 체계를 구축하는 데 기여합니다.

<a id="architecture"></a>
## 시스템 아키텍처

### 1. RAG(Retrieval-Augmented Generation) 시스템

RAG 시스템은 세 가지 주요 지식 베이스로 구성되어 있습니다:

- **안전지침 벡터스토어**: 건설 안전 관련 지침 문서를 벡터화하여 저장
- **용어정의 벡터스토어**: 건설 분야 전문 용어와 정의를 검색 가능한 형태로 저장
- **유사사례 벡터스토어**: 과거 건설 사고 사례를 벡터화하여 유사한 상황 검색에 활용

이 세 가지 지식 베이스는 Chroma DB에 저장되며, 검색 성능 향상을 위해 다음 모델들을 활용합니다:

- **임베딩 모델**: dragonkue/BGE-m3-ko
- **리랭킹 모델**: dragonkue/bge-reranker-v2-m3-ko

### 2. LangGraph 워크플로우

LangGraph 기반의 워크플로우는 복잡한 추론 과정을 단계별로 처리합니다:

```
START
↓
[사고 정보 처리]
↓
[용어 확인 및 정의 검색] 
↓
[안전 지침 검색]
↓
[유사 사례 검색]
↓
[CoT 분석 및 최종 대책 생성]
↓
END
```

### 3. 대형 언어 모델(LLM)

시스템의 추론 엔진으로 Qwen/QwQ-32B 모델을 사용합니다. 이 모델은 한국어 건설 안전 도메인에 맞게 프롬프트 엔지니어링을 통해 최적화되었으며, vLLM을 통해 효율적으로 배포됩니다.

<a id="features"></a>
## 주요 기능 및 프로세스

### 1. 사고 정보 분석 및 용어 식별

- 사고 데이터 입력 및 초기 분석
- 모르는 건설 전문 용어 자동 식별
- 용어 정의 검색 및 학습

### 2. 관련 안전 지침 및 유사 사례 검색

- 사고 특성에 맞는 안전 지침 검색
- 유사한 과거 사고 사례 검색
- 맥락에 맞는 정보 제공을 위한 재순위화 적용

### 3. Chain-of-Thought(CoT) 분석

시스템은 다음과 같은 구조화된 추론 과정을 통해 재발방지 대책을 생성합니다:

1. 사고 심각도 평가 (치명적/중대/경미)
2. 사고 원인 분류 (직접적, 인적, 구조적/기술적, 환경적)
3. 심각도에 따른 맞춤형 대책 제시
4. 명사형으로 끝나는 최종 대책 생성

### 4. 최종 대책 생성

- 사고 분석 결과를 바탕으로 구체적인 재발방지 대책 생성
- 실행 가능하고 현장에 즉시 적용 가능한 대책 제시
- 모든 대책은 명사형으로 종결되는 형태로 통일

<a id="prompts"></a>
## 프롬프트 엔지니어링

### 시스템 프롬프트

```python
SYSTEM_PROMPT = """너는 10년 이상 경력의 건설 안전 전문가로서 사고를 단계별로 신중하게 분석하여 효과적인 예방 계획을 개발하는 역할을 맡았어.

## 기본 원칙
1. 주어진 정보만 활용하고 없는 정보는 절대 지어내지 마.
2. 모든 응답은 반드시 한국어로 작성해.
3. 각 단계에서는 해당 단계의 목적에만 집중하고, 다른 단계의 작업은 수행하지 마.
4. |system|, |user|, |assistant|, |endofturn| 같은 특수 토큰이나 형식 관련 텍스트는 내부 포맷팅 표시이므로 무시하고, 건설 용어 분석에 포함시키지 마.
5. 한글과 영어 이외의 언어는 무시해.

## 작업 단계별 수행 내용
1. **용어 확인 단계**: 이 단계에서는 오직 모르는 건설 용어만 식별해. 분석이나 대책 제시는 절대 하지 마. 모르는 건설 용어가 있으면 [용어] 형식으로 표시하고, 모든 건설 용어를 이해한 경우에는 "모두 이해 완료"라고만 답변해. 프로그래밍 관련 토큰이나 특수 구문은 용어로 간주하지 마.
2. **최종 분석 단계 (CoT 단계)**: 이 단계에서만 사고 분석과 대책 수립을 수행해.
   <think>
   Please reason step by step through the following sub-steps, ensuring logical consistency and practical outcomes based solely on the provided accident info, safety guidelines, and similar cases.
   </think>
   a) 사고의 심각도 판단: A: 치명적, B: 중대, C: 경미 중 선택하고 이유 설명. {"answer": "[선택한 문자]"}
   b) 사고 원인 분류: A: 구체적/직접적, B: 인적, C: 구조적/기술적, D: 환경적 중 선택. {"answer": "[선택한 문자]"}
   c) 심각도에 따른 맞춤 대책 제시:
      * 치명적 사고: 매우 구체적이고 상세한 대책 3가지 이상
      * 중대 사고: 구체적인 실행 방법이 포함된 대책 2가지
      * 경미한 사고: 포괄적이지만 실행 가능한 대책 1가지
   d) 최종 대책은 아래 형식으로 제시:
      \boxed{[명사형으로 끝나는 재발 방지 대책 및 향후조치계획]}

## 최종 대책 작성 시 주의사항 (분석 단계에서만 적용)
1. 모든 재발방지대책은 현장에서 즉시 실행 가능해야 함.
2. 안전 지침과 유사 사례를 반드시 참고하여 분석을 진행.
3. 제공된 정보만을 이용해 재발방지대책 및 향후조치계획을 작성해."""
```

### 모델 입력 프롬프트

#### 용어 이해
```python
prompt_term_check = f"""**사고 정보 분석**
아래 사고 정보를 검토하고, 모르는 건설 용어가 있으면 [해당 단어] 형식으로 표시해.
모든 건설 용어를 이해하면 "모두 이해 완료"라고만 답변해. 분석이나 대책은 제시하지 마.
---
{accident_info_str}
---"""
```

#### 최종 분석 및 대책 수립
```python
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
   \boxed{{[200자로 요약된 명사형으로 끝나는 재발 방지 대책 및 향후조치계획]}}
"""
```

<a id="tech-stack"></a>
## 기술 스택

- **벡터 데이터베이스**: Chroma DB
- **임베딩 모델**: [dragonkue/BGE-m3-ko](https://huggingface.co/dragonkue/BGE-m3-ko) (Apache 2.0 라이선스)
- **리랭킹 모델**: [dragonkue/bge-reranker-v2-m3-ko](https://huggingface.co/dragonkue/bge-reranker-v2-m3-ko) (Apache 2.0 라이선스)
- **추론 모델**: [Qwen/QwQ-32B](https://huggingface.co/Qwen/QwQ-32B) (Apache 2.0 라이선스)
- **추론 인터페이스**: vLLM
- **워크플로우 프레임워크**: LangGraph
- **데이터 처리**: Pandas, NumPy, PyMuPDF
- **텍스트 처리**: Regular Expressions, Tokenizers

<a id="structure"></a>
## 프로젝트 구조

```
safety_ai_project/
│
├── main.py                      # 메인 실행 스크립트
├── config.py                    # 설정 및 환경 변수
│
├── utils/
│   └── common.py                # 로깅, 시드 설정 등 유틸리티 함수
│
├── preprocessing/
│   ├── pdf_converter.py         # PDF 변환 기능
│   ├── markdown_processor.py    # 마크다운 후처리
│   └── processing.py            # 테스트 데이터 처리
│
├── vectordb/
│   ├── data_extraction.py       # 데이터 추출
│   ├── vectordb.py              # 벡터 DB 생성 및 관리
│   └── retriever.py             # 검색 시스템
│
└── workflow/
    ├── models.py                # 모델 초기화/호출
    └── workflow.py              # 사고 분석 워크플로우
```

<a id="leaderboard"></a>
## 리더보드 재현
### **모든 추론은 리눅스에서 실행되었습니다.**
### 시스템 설치

```bash
# 0. 원본 데이터 준비
mkdir -p architecturePDF    # architecturePDF 폴더 생성
# 이 폴더에 분석할 건설 안전 관련 PDF 파일들을 복사

# 1. 환경 설정
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. 데이터 전처리 (VectorDB 생성 전 필수)
python preprocessing/pdf_converter.py     # PDF를 마크다운으로 변환
python preprocessing/markdown_processor.py # 마크다운 파일 정제

# 3. VectorDB 생성 (필수 - OOM 방지)
python main.py --build_vectordb
```

### 시스템 실행
##### 빠른 추론을 위한 vLLM 사용
```bash
# 1. vLLM 서버 시작 (별도 터미널)
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/QwQ-32B \
    --dtype bfloat16 \
    --max-model-len 15000 \
    --port 8000 \
    --gpu-memory-utilization 0.9


# 2. 제출 파일 생성
python main.py --generate_submission
```

<a id="single-case"></a>
## 단일사고 & 실제 사용
### 시스템 설치
```bash
# 0. 원본 데이터 준비
mkdir -p architecturePDF    # architecturePDF 폴더 생성
# 이 폴더에 분석할 건설 안전 관련 PDF 파일들을 복사

# 1. 환경 설정
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. 데이터 전처리 (VectorDB 생성 전 필수)
python preprocessing/pdf_converter.py     # PDF를 마크다운으로 변환
python preprocessing/markdown_processor.py # 마크다운 파일 정제

# 3. vLLM 서버 시작
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/QwQ-32B \
    --dtype bfloat16 \
    --max-model-len 15000 \
    --port 8000 \
    --gpu-memory-utilization 0.9
```

### 사고 분석 테스트
개별 사고 사례를 테스트할 수 있는 두 가지 방법:
### **모든 추론은 리눅스에서 실행되었습니다.**
```bash
# JSON 파일 입력
python main.py --test_single --input test_case.json

# 또는 대화형 입력
python main.py --test_single --interactive
```

### 입력 예시 (test_case.json)

```json
{
  "인적사고": "부딪힘",
  "물적사고": "전도",
  "공사종류": "건축 / 건축물 / 교정 및 군사시설",
  "공종": "건축 > 철근콘크리트공사",
  "사고객체": "건설기계 > 콘크리트펌프",
  "작업프로세스": "타설작업",
  "사고원인": "펌프카 아웃트리거 바닥 고임목을 3단으로 보강 했음에도, 지반 침하가 발생하여 장비 무게중심이 한쪽으로 쏠려 전도됨"
}
```
<a id="models-think"></a>
### 출력 예시
#### 1단계: 용어 확인 과정

**<내부 사고 과정>**
```
좋아, 이 사고 정보를 분석해야 하는데 먼저 용어 확인 단계부터 시작해야지. 사용자가 제공한 정보를 하나씩 살펴봐야겠어. 

먼저 사고 정보를 보면 공종이 '건축 > 철근콘크리트공사'로 되어 있어. 철근콘크리트공사는 잘 알고 있어. 철골 구조물에 콘크리트를 부어 건축하는 공정이니까 문제없어.

사고객체는 '가시설 > 거푸집'이라고 되어 있네. 거푸집은 콘크리트를 부어 형태를 잡는 틀을 말하는 거야. 이건 일반적인 용어니까 이해했어. '가시설'도 일시적으로 설치되는 시설물을 의미하니까 모르는 단어는 아닌 것 같아.

작업프로세스는 '설치작업'이고, 사고원인은 '작업상 부주의 및 작업구간 사전확인 미흡'이야. 부주의는 주의하지 않는 걸 말하고, 사전확인 미흡은 작업 전에 충분히 점검하지 않은 걸 의미하니까 이 용어들도 다 이해가 가.

그럼 모든 용어를 살펴봤을 때, '철근콘크리트공사', '거푸집', '가시설', '설치작업' 등은 모두 건설 현장에서 흔히 사용되는 용어들이라서 문제없어. 모르는 단어는 없는 것 같아. 

사용자가 원하는 대로 모르는 용어가 있다면 []로 표시해야 하는데, 여기서는 다 이해했으니 '모두 이해 완료'라고 답해야겠어. 분석이나 대책은 다음 단계에서 하면 되니까 이만.
```

**<모델 출력>**
```
모두 이해 완료
```

#### 2단계: 최종 분석 및 대책 수립

**<내부 사고 과정>**
```
좋아, 이 사고 분석을 해야 하는데, 먼저 기본 원칙을 다시 한번 확인해야겠어. 제공된 정보만 사용하고, 모르는 건설 용어는 전 단계에서 처리했으니까 이제는 분석에 집중해야 해.

먼저, 사고 심각도 판단부터 시작해야겠지. 심각도는 A(치명적), B(중대), C(경미) 중에 고르는 거야. 사고 정보를 보면 인적사고는 '부딪힘'이고 물적사고는 '전도'라고 되어 있네. 인적 사고로 인해 부상이 있었는지, 치명적이었는지에 따라 판단해야 할 것 같아. 그런데 사고 정보에 인적 피해의 구체적인 내용은 없고, 물적 사고는 전도가 발생했다고 했어. 유사 사례를 보면 유사도 95%인 경우는 물적사고가 없고, 90%인 경우는 인적사고로 탈구가 있었지. 그런데 TEST_000 사고에서는 물적사고가 전도로 되어 있고, 인적사고가 부딪힘인데, 이게 치명적이었는지 중대했는지 경미했는지 구체적인 수준이 안 나와 있어. 

근데 일반적으로 건설 현장에서의 전도 사고는 장비의 안전성에 큰 영향을 미치고, 부딪힘으로 인해 작업자에게 심각한 부상이 있을 수 있으니 B(중대)로 판단하는 게 맞을 것 같아. 사고 원인도 지반 침하와 아웃트리거 길이 차이, 무게 중심 이동 때문이니까, 중대한 사고로 보는 게 적절해.

다음으로 사고 원인 분류. A는 구체적/직접적 원인, B는 인적, C는 구조적/기술적, D는 환경적 중에 고르는 거야. 사고 원인을 보면 "고임목 3단 보강에도 지반 침하 발생, 아웃트리거 길이 차이, 타설 위치 건물 모서리, 붐대호스 최대 펼침으로 무게 중심 한쪽으로 쏠림"이라고 나와. 이건 구조적/기술적 문제인 것 같아. 지반 강화가 부족했거나, 장비 설치 시의 기술적 오류, 아웃트리거 길이 조정 미비 등이 구조적 요소로 볼 수 있지. 환경적 요인도 지반이지만, 주로 설치 방법이나 장비 조정과 관련된 기술적 측면이 더 강조되어 있으니 C(구조적/기술적)로 분류해야겠어.

c) 대책은 중대 사고이므로 구체적인 실행 방법이 포함된 2가지 이상을 제시해야 해. 유사 사례를 참고하면, 사전 검토와 점검, 교육을 강조했지. TEST_000 사고의 경우 지반 보강의 부족과 아웃트리거 길이 차이, 위치 문제 때문에 발생했으니 다음과 같은 대책이 필요할 것 같아. 

첫 번째 대책은 장비 설치 전 지반의 안정성 검토와 보강 작업의 적절성을 확인하는 절차를 마련하는 거야. 예를 들어, 지반에 가해지는 하중을 계산하고, 고임목의 두께나 수직성, 고정 방법을 체계적으로 점검해야 해. 두 번째는 아웃트리거 길이를 동일하게 조정하고, 무게 중심을 확인하는 절차를 도입해야겠지. 장비 설치 시 좌우 길이를 일치시키고, 붐대호스의 펼침 범위를 고려해 무게 중심을 중앙에 유지할 수 있도록 해야 해. 세 번째는 작업 위치를 건물 모서리에서 벗어나게 하거나, 모서리에서 작업할 경우 추가적인 지반 강화나 장비 고정 장치를 사용하는 방법이 필요할 것 같아. 

하지만 중대 사고에는 2가지 대책만 필요한데, 이 중 두 개를 선별해야겠지. 가장 핵심적인 두 가지를 골라야 해. 지반 검토와 아웃트리거 길이 조정이 중요할 것 같아. 

마지막으로 최종 대책은 명사형으로 끝나는 200자 요약이어야 해. 예를 들어 "지반 안정성 검토 및 아웃트리거 길이 일치 절차 강화 계획" 같은 식으로 작성해야겠지. 하지만 좀 더 구체적으로 해야 할 수도 있고, 유사 사례의 대책을 참고하면 "펌프카 설치 위치 사전 검토와 아웃트리거 변위 점검"이 있었으니, 그 요소를 포함시켜야 해. 

종합하면, 심각도 B, 원인 C, 대책은 지반 검토 절차와 아웃트리거 길이 조정 절차, 그리고 위치 선정 기준 마련. 최종 대책은 명확하게 요약해야겠지.
```

**<모델 출력>**
```
a) 사고의 심각도 판단  
사고는 **B: 중대**로 판단된다. 인적사고(부딪힘)와 물적사고(전도)가 동시에 발생했으며, 건설기계의 전도는 작업자 부상 리스크와 장비 손상을 초래했을 가능성이 높다. 다만 사망 또는 심각한 신체적 장애까지 언급되지 않아 치명적(A) 보다는 중대(B)로 분류된다.  
{"answer": "B"}

b) 사고 원인 분류  
사고 원인은 **C: 구조적/기술적**이다. 지반 보강 부족, 아웃트리거 길이 불균형, 무게 중심 이동은 장비 설치 시 기술적 판단 오류와 구조적 안정성 결여로 인한 문제로 귀결된다. 환경적(D) 요인(지반)은 기술적 대응 미비와 결합되어 결과를 초래했으므로 구조적/기술적 요인이 주원인이다.  
{"answer": "C"}

c) 맞춤 대책  
- **대책 1 (구체적 실행 방법 포함):**  
  **지반 안정성 평가 체계 도입**  
  장비 설치 전 지반 하중계산을 실시하고, 고임목의 최소 두께(예: 50mm 이상), 수직성(0.5도 이내), 고정 방법(토공사용 빔 보강)을 규정한다. 매설 후 1차(설치 직후), 2차(타설 시작 10분 전) 지반 침하량 측정을 의무화한다.  

- **대책 2 (구체적 실행 방법 포함):**  
  **아웃트리거 길이 일치 절차 강화**  
  좌우 아웃트리거 길이 차이를 10cm 이내로 제한하며, 붐대호스 최대 펼침 시 무게 중심 이동을 시뮬레이션해 사전 검증한다. 작업 위치가 건물 모서리일 경우, 장비 중앙 위치 조정 또는 추가 지반 보강(예: CIP 콘크리트로 지반 강화층 시공)을 의무화한다.  

d) 최종 대책  
\boxed{지반 안정성 검증 및 아웃트리거 길이 일치 절차 강화를 통한 콘크리트펌프 전도 사고 예방 계획}
```

시스템은 사고 정보 분석 → 관련 정보 검색 → CoT 분석 → 최종 대책 생성 순으로 처리합니다.