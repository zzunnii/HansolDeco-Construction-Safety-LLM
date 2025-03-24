import os
import re
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from config import OUTPUT_DIR, EMBEDDING_MODEL, RERANKER_MODEL

# 임베딩 모델 초기화
device = "cuda" if os.path.exists("/dev/nvidia0") else "cpu"
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={'device': device}
)


class GuidelineRetriever:
    def __init__(self, guidelines_db=None):
        self.db = guidelines_db or Chroma(
            persist_directory=os.path.join(OUTPUT_DIR, "chroma_guidelines"),
            embedding_function=embeddings,
            collection_name="safety_guidelines"
        )
        self.reranker = CrossEncoderReranker(
            model=HuggingFaceCrossEncoder(model_name=RERANKER_MODEL),
            top_n=3
        )
        self.retriever = self.db.as_retriever(search_kwargs={"k": 10})
        self.compression_retriever = ContextualCompressionRetriever(
            base_compressor=self.reranker,
            base_retriever=self.retriever
        )

    def search(self, query, k=3):
        if isinstance(query, dict):
            weighted_query = " ".join([
                query.get("사고 원인", "") * 3,
                query.get("사고객체", "") * 2,
                query.get("인적사고", ""),
                query.get("작업 프로세스", "")
            ])
        else:
            weighted_query = query
        results = self.compression_retriever.get_relevant_documents(weighted_query)
        return "\n\n".join([f"{i + 1}. {doc.page_content}" for i, doc in enumerate(results[:k])])


class TermRetriever:
    def __init__(self, terms_db=None):
        self.db = terms_db or Chroma(
            persist_directory=os.path.join(OUTPUT_DIR, "chroma_terms"),
            embedding_function=embeddings,
            collection_name="term_definitions"
        )

    def get_definition(self, term):
        results = self.db.similarity_search_with_score(term, k=3, filter={"type": "term_definition"})
        if results and results[0][1] < 0.3:
            content = results[0][0].page_content
            definition = re.search(r'정의: (.+)', content)
            return definition.group(1).strip() if definition else f"{term}은(는) 건설 현장에서 사용되는 용어야."
        return f"{term}은(는) 건설 현장에서 사용되는 용어야."


class CaseRetriever:
    def __init__(self, cases_db=None):
        self.db = cases_db or Chroma(
            persist_directory=os.path.join(OUTPUT_DIR, "chroma_cases"),
            embedding_function=embeddings,
            collection_name="similar_cases"
        )
        self.reranker = CrossEncoderReranker(
            model=HuggingFaceCrossEncoder(model_name=RERANKER_MODEL),
            top_n=2
        )
        self.retriever = self.db.as_retriever(search_kwargs={"k": 10})
        self.compression_retriever = ContextualCompressionRetriever(
            base_compressor=self.reranker,
            base_retriever=self.retriever
        )

    def search(self, query, min_score=0.6):
        results = self.compression_retriever.get_relevant_documents(query)
        return "\n\n".join([
            f"유사 사례 {i + 1} (유사도: {max(0.85, 0.95 - i * 0.05) * 100:.1f}%):\n{doc.page_content}"
            for i, doc in enumerate(results[:2])
        ])