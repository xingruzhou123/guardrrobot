import asyncio
from guardrails.retrieval.knowledge_base import KnowledgeBase
from typing import List, Tuple


class RAGRetriever:
    """
    A wrapper class for the KnowledgeBase that provides the 'search' method
    expected by the SimpleRuntime.
    """

    def __init__(self, kb_path: str, top_k: int, confidence_threshold: float, **kwargs):
        """
        Initializes the retriever and the underlying KnowledgeBase.
        """
        self.kb_path = kb_path
        self.top_k = top_k
        self.confidence_threshold = confidence_threshold

        print(f"[RAGRetriever] ✅ Initialized (kb_path={kb_path}, top_k={top_k})")
        print(f"[RAGRetriever] 📂 Initializing KnowledgeBase from: {kb_path}")

        # RAGRetriever 拥有一个 KnowledgeBase 实例
        self.kb = KnowledgeBase(kb_dir=kb_path)

    # --- [ 关键修复 ] ---
    # 添加缺失的 'search' 方法
    # 它必须是 'async' 才能调用 KnowledgeBase.search

    async def search(self, query: str) -> List[Tuple[str, float, str]]:
        """
        Asynchronous search method that delegates the call to the
        KnowledgeBase instance.
        """
        # 调用 KnowledgeBase 的 search 方法
        return await self.kb.search(query, top_k=self.top_k)
