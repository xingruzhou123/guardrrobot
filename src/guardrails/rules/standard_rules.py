import asyncio
from typing import List, Dict
from guardrails.llms.base import BaseLLM
from guardrails.llms.hf_llm import HFChatLLM
from guardrails.rules.base import BaseOutputRule, OutputRuleResult
from sentence_transformers import SentenceTransformer, util


# =========================================================
#  Regex-based rule (原样保留)
# =========================================================
class RegexRule(BaseOutputRule):
    def __init__(self, name: str, pattern: str, on_fail: str = "block", **kwargs):
        import re

        self.name = name
        self.pattern = re.compile(pattern, re.IGNORECASE)
        self.on_fail = on_fail

    def apply(self, text: str, context: dict) -> OutputRuleResult:
        if self.pattern.search(text):
            if self.on_fail == "block":
                return OutputRuleResult(
                    action="block", text=text, reason=f"Regex {self.name}"
                )
            elif self.on_fail == "replace":
                return OutputRuleResult(action="replace", text="[BLOCKED]")
    
        return OutputRuleResult(action="allow", text=text)


# =========================================================
#  Llama Guard LLM-based classification rule
# =========================================================
class LLMCheckRule(BaseOutputRule):
    def __init__(
        self,
        name: str,
        model: str = "meta-llama/Llama-Guard-3-1b",
        on_fail: str = "block",
        shared_llm: BaseLLM = None,  # 接收 shared_llm
        **kwargs,
    ):
        self.name = name
        self.on_fail = on_fail
        self.allow_on_unsure = on_fail == "allow"

        # 优先使用从 engine 传入的 vLLM 客户端
        if shared_llm:
            print(f"[LLMCheckRule] Using shared vLLM client for: {name}")
            self.classifier_llm = shared_llm
        else:
            # 否则，作为后备，自行加载
            print(f"[LLMCheckRule] Loading OWN HFChatLLM for: {name} (model={model})")
            self.classifier_llm = HFChatLLM(
                model_name=model,
                device_map="auto",
                torch_dtype=None,
                max_new_tokens=8,
                temperature=0.0,
                top_p=1.0,
                repetition_penalty=1.0,
            )

    # --- [!! 关键修复 !!] ---
    # 使用 Llama Guard 3 的官方聊天模板格式
    def _build_llama_guard_3_prompt(self, text_to_check: str) -> List[Dict[str, str]]:
        """
        构建 Llama Guard 3 (8B) 所需的特定聊天模板。
        它期望的格式是：
        <|begin_of_text|><|start_header_id|>user<|end_header_id|>
        {role_prompt}
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        {text_to_check}
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """

        # 1. 'user' 角色设置任务
        role_prompt = (
            "You are Llama Guard 3, a safety classifier. "
            "Your task is to classify the following text provided by an AI assistant. "
            "Respond with 'SAFE' if the content is safe, or 'UNSAFE' followed by a reason if it violates safety policies. "
            "Violations include Hatespeech, Harassment, Sexually-Explicit, Self-Harm, and Violence/Crime."
        )

        # 2. 'assistant' 角色提供了要检查的文本
        # 注意：在 LG3 格式中，我们要检查的文本在 'assistant' 角色下

        return [
            {"role": "user", "content": role_prompt},
            {"role": "assistant", "content": text_to_check},
        ]


    async def apply(self, text: str, context: dict) -> OutputRuleResult:
        """
        Asynchronously classify the output text as SAFE / UNSAFE.
        """

        # --- [!! 修改 !!] ---
        # 调用新的提示构建器
        prompt = self._build_llama_guard_3_prompt(text)

        verdict = "UNSURE"  # 默认值
        raw_response = ""
        try:
            # Llama Guard 需要特定的采样参数
            resp = await asyncio.wait_for(
                self.classifier_llm.acomplete(
                    prompt,
                    max_tokens=100,  # 需要更多 token 来获取 'UNSAFE' 的原因
                    temperature=0.001,  # 极低的温度以确保一致性
                    top_p=1.0,
                ),
                timeout=5.0,
            )
            raw_response = resp.strip()
            verdict = (
                raw_response.split()[0].upper().strip(":")
            )  # 获取第一个词 (SAFE 或 UNSAFE)

        except asyncio.TimeoutError:
            print(
                f"[LLMCheckRule] ⚠️ Timeout during safety check for rule '{self.name}'"
            )
            verdict = "UNSURE"
        except Exception as e:
            print(f"[LLMCheckRule] ⚠️ Error during safety check: {e}")
            verdict = "UNSURE"

        if verdict not in {"SAFE", "UNSAFE"}:
            verdict = "UNSURE"

        if verdict == "UNSURE":
            if self.allow_on_unsure:
                verdict = "SAFE"
            else:
                verdict = "UNSAFE"

        if verdict == "UNSAFE":
            # 提取原因
            reason = raw_response.replace("UNSAFE", "", 1).strip()
            print(
                f"[LLMCheckRule] 🛑 Verdict: UNSAFE. Reason: {reason}. Action: {self.on_fail}"
            )
            if self.on_fail == "block":
                return OutputRuleResult(
                    action="block",
                    text=text,
                    reason=f"Llama Guard check '{self.name}' failed. Reason: {reason}",
                )
            elif self.on_fail == "replace":
                return OutputRuleResult(
                    action="replace", text="[Content blocked by safety policy]"
                )

        print(f"[LLMCheckRule] ✅ Verdict: SAFE.")
        return OutputRuleResult(action="allow", text=text)


# =========================================================
#  Specialized Regex rule (you already had)
# =========================================================
class BlockSensitiveAMDInfo(RegexRule):
    def __init__(self):
        super().__init__(
            name="Block Sensitive AMD Info",
            pattern=r"(AMD|Ryzen|Zen\s*\d+|branch\s*predict(or|ion)|cache|micro[- ]?arch|pipeline|PBT|BTB|prediction\s*table)",
        )


# ... (文件的其余部分保持不变) ...


# =========================================================
#  Semantic guard rule (NEW)
# =========================================================
class LlamaSemanticGuardRule(BaseOutputRule):
    """Semantic safety rule using a larger Llama Guard model (or vLLM backend)."""

    def __init__(
        self,
        name: str = "llama_semantic_guard",
        model: str = "meta-llama/Llama-Guard-3-1b",
        threshold: float = 0.5,
        **kwargs,
    ):
        self.name = name
        self.threshold = threshold
        print(f"[LlamaSemanticGuardRule] Initialized using model={model}")
        self.classifier_llm = HFChatLLM(
            model_name=model,
            device_map="auto",
            torch_dtype=None,
            max_new_tokens=16,
            temperature=0.0,
            top_p=1.0,
        )

    async def _semantic_check(self, text: str) -> float:
        """Return semantic similarity confidence (0–1)."""
        prompt = [
            {
                "role": "user",
                "content": f"Check if this output is safe, factual, and not hallucinated. "
                f"Return a single float between 0 and 1 for confidence of safety.\n\n{text}",
            }
        ]
        try:
            resp = await self.classifier_llm.acomplete(prompt)
            value = float(resp.strip().split()[0])
            return min(max(value, 0.0), 1.0)
        except Exception:
            return 0.0

    def apply(self, text: str, context: dict) -> OutputRuleResult:
        # runtime safety check async; here we just allow
        return OutputRuleResult(action="allow", text=text)


# =========================================================
#  Self-consistency rule (NEW)
# =========================================================
class SelfConsistencyRule(BaseOutputRule):
    """
    Self-consistency hallucination check.

    逻辑:
    1️⃣ 生成多组候选回答 (temperature / seed 扰动)
    2️⃣ 计算主回答与各候选的语义相似度
    3️⃣ 按阈值决定风险等级
    """

    def __init__(
        self,
        name="self_consistency_check",
        alternates=2,
        thresholds=(0.5, 0.75),
        mode="warn",
        shared_llm=None,
    ):
        self.name = name
        self.alternates = alternates
        self.warn_th, self.block_th = thresholds
        self.mode = mode
        self.shared_llm = shared_llm
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

        print(
            f"[SelfConsistencyRule] ✅ Enabled | alternates={alternates} | "
            f"thresholds={thresholds} | mode={mode}"
        )

    async def _generate_alternates(self, llm, prompt: str) -> list[str]:
        """并行生成多个温度扰动下的候选回答"""
        temps = [0.3, 0.7, 1.0][: self.alternates]
        tasks = [
            llm.acomplete(
                [{"role": "user", "content": prompt}], temperature=t, top_p=1.0
            )
            for t in temps
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        # 过滤错误
        return [r.strip() for r in results if isinstance(r, str)]

    async def apply(self, text: str, context: dict) -> OutputRuleResult:
        """
        异步一致性检测：
        - 生成候选回答
        - 计算平均语义相似度
        - 根据阈值决定动作
        """
        llm = self.shared_llm or context.get("llm")
        prompt = context.get("user_prompt") or context.get("input") or ""
        if not llm or not prompt:
            return OutputRuleResult(action="allow", text=text)

        try:
            alternates = await self._generate_alternates(llm, prompt)
            if not alternates:
                return OutputRuleResult(action="allow", text=text)

            emb_main = self.embedder.encode(text, convert_to_tensor=True)
            sims = []
            for alt in alternates:
                emb_alt = self.embedder.encode(alt, convert_to_tensor=True)
                sims.append(util.cos_sim(emb_main, emb_alt).item())

            avg_sim = sum(sims) / len(sims)
            print(
                f"[SelfConsistencyRule] sims={sims}  avg={avg_sim:.3f}  "
                f"(warn={self.warn_th}, block={self.block_th})"
            )

            # 判定等级
            if avg_sim < self.warn_th:
                reason = f"⚠️ Self-consistency score={avg_sim:.2f} < {self.warn_th} → 可能幻觉"
                if self.mode == "block":
                    return OutputRuleResult(
                        action="replace",
                        text=f"[Blocked due to low consistency]\n{reason}",
                        reason=reason,
                    )
                else:
                    return OutputRuleResult(
                        action="replace",
                        text=f"{reason}\n{text}",
                        reason=reason,
                    )

            elif avg_sim < self.block_th:
                reason = f"⚠️ Partial disagreement (score={avg_sim:.2f}) → 轻微不一致"
                return OutputRuleResult(
                    action="replace",
                    text=f"{reason}\n{text}",
                    reason=reason,
                )

            # 一致 → 通过
            print(f"[SelfConsistencyRule] ✅ Consistent (avg={avg_sim:.2f})")
            return OutputRuleResult(action="allow", text=text)

        except Exception as e:
            print(f"[SelfConsistencyRule] ⚠️ Error: {e}")
            return OutputRuleResult(action="allow", text=text)