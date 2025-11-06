import json
import re
import asyncio
import numpy as np
from typing import Any, Dict, List, Tuple
from sentence_transformers import SentenceTransformer, util  # <-- [新增] 导入

from .base import BaseOutputRule, OutputRuleResult
from ..llms.base import BaseLLM


class SelfConsistencyRule(BaseOutputRule):
    """
    执行“自我验证”（Self-Verification）。
    它接收一个答案，然后生成N个对该答案的“事实核查”回复。
    它比较原始答案和事实核查回复之间的语义相似度。
    低相似度 → 可能是幻觉或自相矛盾。
    """

    def __init__(
        self,
        name: str,
        shared_llm: BaseLLM,  # 这是用于 *验证* 的LLM
        num_alternates: int = 2,
        mode: str = "warn",
        block_threshold: float = 0.5,
        warn_threshold: float = 0.75,
        **kwargs,
    ):
        self.name = name
        self.verification_llm = shared_llm
        self.num_alternates = int(num_alternates)
        self.mode = mode
        self.block_threshold = float(block_threshold)
        self.warn_threshold = float(warn_threshold)

        # [修复] 初始化 Sentence Transformer
        try:
            self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as e:
            print(
                f"[SelfConsistencyRule] ⚠️ 错误: 无法加载 SentenceTransformer. 规则将被跳过. 错误: {e}"
            )
            self.encoder = None
            self.verification_llm = None  # 禁用规则

        print(
            f"[SelfConsistencyRule] Initialized with mode={mode}, alternates={num_alternates}, thresholds=({block_threshold}, {warn_threshold})"
        )

    async def _get_verification(self, text_to_check: str) -> str:
        """
        调用LLM来生成一个对文本的验证（事实核查）。
        """
        # [修复] 这是一个“自我验证”提示词
        prompt = [
            {
                "role": "system",
                "content": "You are a meticulous fact-checker. Evaluate the following statement for factual accuracy. If it is accurate, repeat the statement. If it is inaccurate, provide the correction.",
            },
            {"role": "user", "content": text_to_check},
        ]

        try:
            return await self.verification_llm.acomplete(
                prompt, temperature=0.5, max_tokens=256
            )
        except Exception as e:
            print(f"[SelfConsistencyRule] Error during verification: {e}")
            return ""

    async def apply(self, text: str, context: Dict[str, Any]) -> OutputRuleResult:
        """
        应用自我验证规则。
        """
        print(f"[SelfConsistencyRule] 🚀 called with text[:50]={text[:50]!r}")
        print(
            f"[SelfConsistencyRule] 🔧 has encoder: {self.encoder is not None}, verification_llm: {self.verification_llm is not None}"
        )

        # [修复] 检查我们是否可以运行
        if not self.verification_llm or not self.encoder or not text or len(text) < 20:
            return OutputRuleResult(action="allow", text=text)  # 跳过

        main_response = text

        # --- 1. 创建 N 个验证任务 ---
        verification_tasks = [
            self._get_verification(main_response) for _ in range(self.num_alternates)
        ]

        # --- 2. 异步执行所有任务 ---
        verifications = await asyncio.gather(*verification_tasks)
        print(f"[SelfConsistencyRule] 🧩 verification outputs = {verifications}")
        # --- 3. 嵌入所有文本 (原始 + 验证) ---
        all_texts = [main_response] + [v for v in verifications if v]
        if len(all_texts) < 2:
            return OutputRuleResult(action="allow", text=text)  # 验证失败，跳过

        embeddings = self.encoder.encode(all_texts, convert_to_tensor=True)

        # --- 4. 计算相似度 ---
        # 计算所有验证结果与[0] (原始文本)的余弦相似度
        similarities = util.cos_sim(embeddings[0], embeddings[1:]).flatten()

        if len(similarities) == 0:
            return OutputRuleResult(action="allow", text=text)

        # --- 5. 获取平均相似度 ---
        avg_similarity = np.mean([s.item() for s in similarities])

        # 打印你正在寻找的关键日志！
        print(
            f"[SelfConsistencyRule] Avg. Similarity: {avg_similarity:.3f} (Block < {self.block_threshold}, Warn < {self.warn_threshold})"
        )

        # --- 6. 应用阈值 ---
        if avg_similarity < self.block_threshold:
            reason = f"Verification failed (Avg. Similarity: {avg_similarity:.3f} < {self.block_threshold})"
            if self.mode == "block":
                return OutputRuleResult(action="block", text=text, reason=reason)
            else:
                return OutputRuleResult(action="warn", reason=reason, text=text)

        if avg_similarity < self.warn_threshold:
            print(
                f"[SelfConsistencyRule] ⚠️ WARNING: Low consistency detected. (Avg. Similarity: {avg_similarity:.3f})"
            )
            context["self_consistency_warning"] = True
            return OutputRuleResult(action="allow", text=text)  # 允许，但发出警告

        return OutputRuleResult(
            action="allow",
            text=text,
            reason=f"Self-consistency check passed (score: {avg_similarity:.3f})",
        )
