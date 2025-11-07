import inspect
import asyncio
import copy
import yaml
from typing import Any, Coroutine, Dict, List, AsyncIterator

from .config_types import RailsConfig
from guardrails.rules.base import BaseOutputRule, OutputRuleResult
from ..llms.base import BaseLLM

from guardrails.actions import action_handler
from guardrails.rules.consistency_rule import SelfConsistencyRule
from guardrails.core.rag_retriever import RAGRetriever


class SimpleRuntime:
    """
    A simple runtime that processes a user request through a set of rails.
    It orchestrates the flow of input rules, action dispatching,
    and output rules. Now includes RAG fallback with confidence threshold.
    """

    def __init__(
        self,
        config: RailsConfig,
        llm: BaseLLM,
        input_rules: List[BaseOutputRule] = None,
        output_rules: List[BaseOutputRule] = None,
    ):
        """Initializes the runtime."""
        self.config = config
        self.llm = llm
        self.input_rules = input_rules or []
        self.output_rules = output_rules or []

        # This dictionary links intent names from YAML to Python functions.
        self.action_registry = {
            "query_product_price": action_handler.handle_price_query,
            "control_robot_arm": action_handler.handle_robot_arm,
        }
        self.action_registry = {
            "query_product_price": action_handler.handle_price_query,
            "control_robot_arm": action_handler.handle_robot_arm,
            "check_flight_price": action_handler.handle_flight_query,  # <-- [!! 必须添加这一行 !!]
        }
        # --- Load retrieval and fallback configurations ---
        self.conf_threshold = 0.4
        self.fallback_message = ""
        try:
            with open("config/rails.yml", "r") as f:
                yml = yaml.safe_load(f) or {}
                # retrieval section
                retrievals = yml.get("retrieval", [])
                if retrievals:
                    conf = retrievals[0].get("config", {}) or {}
                    self.conf_threshold = float(conf.get("confidence_threshold", 0.4))

                # fallback section
                dialogs = yml.get("dialog", [])
                for d in dialogs:
                    if d.get("type") == "fallback":
                        fb = d.get("config", {}) or {}
                        if "threshold" in fb:
                            self.conf_threshold = float(fb["threshold"])
                        self.fallback_message = fb.get("fallback_message", "")
                        break
        except Exception as e:
            print(f"[runtime] Could not read config/rails.yml: {e}")

        # --- Initialize RAG retriever ---
        self.retriever = RAGRetriever(
            kb_path="/guardrrobot/userrag",
            top_k=3,
            confidence_threshold=self.conf_threshold,
        )

    async def _rag_or_general(
        self, user_query: str, messages: List[Dict[str, Any]], context: Dict[str, Any]
    ) -> str:
        """Unified routing for 'none' intent or unregistered intents."""

        # --- [ 修复 ] ---
        # 直接 await 异步的 .search() 方法
        results = await self.retriever.search(user_query)

        # 'results' 现在是 (text, score, source) 的元组列表
        docs = [{"content": t, "score": s, "source": n} for t, s, n in results]
        max_sim = max([r["score"] for r in docs], default=0.0)

        print(
            f"[RAG] max_similarity={max_sim:.3f} (threshold={self.conf_threshold:.3f})"
        )

        rag_mode = (max_sim >= self.conf_threshold) and bool(docs)
        if rag_mode:
            # 仅使用 content 进行 RAG
            joined = "\n\n---\n".join([d["content"] for d in docs])
            augmented = (
                "You MUST answer based strictly on the evidence below. "
                "If the evidence is insufficient, say so explicitly.\n\n"
                f"[EVIDENCE]\n{joined}\n\n[QUESTION]\n{user_query}"
            )
            aug_msgs = [dict(m) for m in messages]
            aug_msgs[-1] = {"role": "user", "content": augmented}
            context["retrieved_docs"] = [d["content"] for d in docs]

            print("[RAG] Using retrieved context to answer.")
            return await self.llm.acomplete(messages=aug_msgs)

        print("[RAG] Low confidence → falling back to general answering.")
        if self.fallback_message:
            gen_msgs = [*messages]
            gen_msgs.insert(0, {"role": "system", "content": self.fallback_message})
            return await self.llm.acomplete(messages=gen_msgs)
        return await self.llm.acomplete(messages=messages)

    async def run_once(
        self, messages: List[Dict], context: Dict[str, Any] = None
    ) -> str:
        """Processes a single turn of a conversation."""
        if context is None:
            context = {}

        latest_user_message = messages[-1]["content"]
        dispatched = False

        # --- [修改 1] ---
        # bot_response 用于存储"原始"回复，以便后续检查
        bot_response = ""

        # --- 1. Run input rails (intent classifier etc.) ---
        for rule in self.input_rules:
            if inspect.iscoroutinefunction(rule.apply):
                result: OutputRuleResult = await rule.apply(
                    latest_user_message, context
                )
            else:
                result: OutputRuleResult = rule.apply(latest_user_message, context)

            if result.action == "block":
                return f"Input blocked: {result.reason}"

            if result.action == "dispatch":
                extra_info = result.extra_info or {}
                intent = extra_info.get("intent", "none")
                entities = extra_info.get("entities", {})

                print(
                    f"[debug] LLM Classifier Result: Intent='{intent}', Entities={entities}"
                )

                action_func = self.action_registry.get(intent)

                if action_func:
                    dispatched = True
                    # --- [修改 2] ---
                    # 不要立即返回，而是存储回复
                    bot_response = await action_func(entities, llm=self.llm)
                    break  # 退出输入规则循环
                else:
                    print(
                        f"[debug] Intent '{intent}' not registered. Using RAG-or-General routing."
                    )
                    dispatched = True  # 标记为"已处理"
                    # --- [修改 3] ---
                    # 存储回复
                    bot_response = await self._rag_or_general(
                        latest_user_message, messages, context
                    )
                    break  # 退出输入规则循环

        # --- 2. No rule triggered dispatch ---
        if not dispatched:
            # --- [修改 4] ---
            # 存储回复
            bot_response = await self._rag_or_general(
                latest_user_message, messages, context
            )

        # --- 3. [!! 关键修复 !!] 运行 OUTPUT rails ---
        # 这是你缺失的逻辑

        print(f"[debug] Raw Bot Response: {bot_response[:80]}...")

        for rule in self.output_rules:
            if inspect.iscoroutinefunction(rule.apply):
                result: OutputRuleResult = await rule.apply(bot_response, context)
            else:
                result: OutputRuleResult = rule.apply(bot_response, context)

            if result.action == "block":
                print(
                    f"[OutputRail] ⚠️ BLOCKED by rule '{rule.name}'. Reason: {result.reason}"
                )
                # 你可以返回一个通用消息
                return f"Output blocked by safety policy."

            if result.action == "replace":
                print(f"[OutputRail] 🔄 REPLACED by rule '{rule.name}'.")
                bot_response = result.text  # 更新回复以供下一个规则检查

        # --- 4. 返回最终经过验证的回复 ---
        return bot_response

    async def run_stream(
        self, messages: List[Dict], context: Dict[str, Any] = None
    ) -> AsyncIterator[str]:
        """Streaming mode wrapper."""
        response = await self.run_once(messages, context)
        yield response
