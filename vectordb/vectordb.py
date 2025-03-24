from langchain_community.vectorstores import Chroma
from utils.common import logger
from vectordb.data_extraction import extract_safety_guidelines, extract_term_definitions, extract_similar_cases
from vectordb.retriever import embeddings
from config import OUTPUT_DIR
import os

def create_or_load_vectorstores():
    logger.info("벡터스토어 생성/로드 시작...")
    guidelines_path = os.path.join(OUTPUT_DIR, "chroma_guidelines")
    terms_path = os.path.join(OUTPUT_DIR, "chroma_terms")
    cases_path = os.path.join(OUTPUT_DIR, "chroma_cases")

    if os.path.exists(guidelines_path):
        guidelines_db = Chroma(persist_directory=guidelines_path, embedding_function=embeddings,
                               collection_name="safety_guidelines")
        logger.info("기존 안전지침 벡터스토어 로드 완료")
    else:
        guidelines_docs = extract_safety_guidelines()
        guidelines_db = Chroma.from_documents(documents=guidelines_docs, embedding=embeddings,
                                              persist_directory=guidelines_path, collection_name="safety_guidelines")

    if os.path.exists(terms_path):
        terms_db = Chroma(persist_directory=terms_path, embedding_function=embeddings,
                          collection_name="term_definitions")
        logger.info("기존 용어정의 벡터스토어 로드 완료")
    else:
        terms_docs = extract_term_definitions()
        terms_db = Chroma.from_documents(documents=terms_docs, embedding=embeddings, persist_directory=terms_path,
                                         collection_name="term_definitions")

    if os.path.exists(cases_path):
        cases_db = Chroma(persist_directory=cases_path, embedding_function=embeddings, collection_name="similar_cases")
        logger.info("기존 유사사례 벡터스토어 로드 완료")
    else:
        cases_docs = extract_similar_cases()
        cases_db = Chroma.from_documents(documents=cases_docs, embedding=embeddings, persist_directory=cases_path,
                                         collection_name="similar_cases")

    logger.info("벡터스토어 작업 완료!")
    return guidelines_db, terms_db, cases_db