# src/guardrails/rules/rag_faithfulness_rule.py

import numpy as np
from sentence_transformers import SentenceTransformer, util
from guardrails.rules.base import BaseOutputRule, OutputRuleResult


class RAGFaithfulnessRule(BaseOutputRule):
    """
    Verify that the LLM's generated answer is grounded in the retrieved RAG evidence.
    """

    def __init__(
        self,
        name="rag_faithfulness_check",
        block_threshold=0.5,
        warn_threshold=0.75,
        **kwargs,
    ):
        self.name = name
        self.block_threshold = block_threshold
        self.warn_threshold = warn_threshold

        try:
            self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
            print(f"[RAGFaithfulnessRule] ✅ Initialized with encoder all-MiniLM-L6-v2")
        except Exception as e:
            print(f"[RAGFaithfulnessRule] ⚠️ Failed to load encoder: {e}")
            self.encoder = None

    def apply(self, text: str, context: dict) -> OutputRuleResult:
        if not self.encoder:
            return OutputRuleResult(action="allow", text=text)

        # 从 RAG 流程中获取已检索证据
        evidence_list = context.get("retrieved_docs", [])
        if not evidence_list:
            return OutputRuleResult(action="allow", text=text)

        # --- 1. 将回答与证据编码 ---
        answer_emb = self.encoder.encode(text, convert_to_tensor=True)
        evid_embs = self.encoder.encode(evidence_list, convert_to_tensor=True)

        # --- 2. 计算语义相似度 ---
        sims = util.cos_sim(answer_emb, evid_embs).flatten().cpu().numpy()
        max_sim = float(np.max(sims))
        avg_sim = float(np.mean(sims))

        print(
            f"[RAGFaithfulnessRule] sims={np.round(sims, 3).tolist()}  max={max_sim:.3f}  avg={avg_sim:.3f}"
        )

        # --- 3. 应用阈值 ---
        if max_sim < self.block_threshold:
            reason = f"Answer not grounded in RAG evidence (max_sim={max_sim:.3f} < {self.block_threshold})"
            return OutputRuleResult(action="warn", text=text, reason=reason)

        if max_sim < self.warn_threshold:
            print(f"[RAGFaithfulnessRule] ⚠️ Weak grounding (max_sim={max_sim:.3f})")
            context["rag_grounding_warning"] = True
            return OutputRuleResult(action="allow", text=text)

        print(f"[RAGFaithfulnessRule] ✅ Grounded (max_sim={max_sim:.3f})")
        return OutputRuleResult(action="allow", text=text)
