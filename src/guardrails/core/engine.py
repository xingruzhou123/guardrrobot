# src/guardrails/core/engine.py

import yaml
import time
import asyncio
from typing import Any, AsyncIterator, Dict, List, Optional

from guardrails.core.runtime import SimpleRuntime
from guardrails.core.config_types import RailsConfig
from guardrails.rules.base import BaseOutputRule
from guardrails.rules.standard_rules import RegexRule, LLMCheckRule
from guardrails.rules.custom_rules import LLMClassifierRule
from guardrails.rules.rag_faithfulness_rule import RAGFaithfulnessRule
from guardrails.rules.consistency_rule import SelfConsistencyRule
from guardrails.llms.vllm import VLLMOpenAIClient
from guardrails.actions import action_handler


class LLMRails:
    def __init__(self, config: RailsConfig):
        self.config = config

        print("Initializing vLLM Clients...")
        # [修复 1] 明确区分所有 vLLM 客户端

        # 客户端 1: 用于生成主要回复的 LLM (Qwen @ 8000)
        self.llm = VLLMOpenAIClient(
            base_url="http://localhost:8000/v1",
            model_name="Qwen/Qwen3-4B-Instruct-2507",
        )

        # 客户端 2: 用于意图分类的 LLM (Qwen @ 8000)
        print("Initializing Classifier LLM (Qwen)...")
        self.classifier_llm = VLLMOpenAIClient(
            base_url="http://localhost:8000/v1",  # 使用 8000 端口
            model_name="Qwen/Qwen3-4B-Instruct-2507",
        )

        # 客户端 3: 用于语义安全检查的 LLM (Llama Guard @ 8002)
        print("Initializing Semantic Guard LLM (Llama Guard @ 8002)...")
        self.semantic_guard_llm = VLLMOpenAIClient(
            base_url="http://localhost:8002/v1",  # <-- [关键] 指向 8002 端口
            model_name="meta-llama/Llama-Guard-3-1B",  # <-- [关键] 指定 Llama Guard
        )

        # 客户端 4: 用于一致性验证的 LLM (Qwen @ 8000)
        print("Initializing Verification LLM (Qwen)...")
        self.verification_llm = VLLMOpenAIClient(
            base_url="http://localhost:8000/v1",  # 默认使用相同的服务器
            model_name="Qwen/Qwen3-4B-Instruct-2507",  # 默认使用相同的模型
        )

        print("vLLM Clients Initialized.")

        self.input_rules = self._load_rules_from_config(
            "config/rails.yml", rail_type="input_rails"
        )
        self.output_rules = self._load_rules_from_config(
            "config/rails.yml", rail_type="output_rails"
        )

        self.runtime = SimpleRuntime(
            config=self.config,
            llm=self.llm,
            input_rules=self.input_rules,
            output_rules=self.output_rules,
        )

    def _load_rules_from_config(
        self, filepath: str, rail_type: str
    ) -> List[BaseOutputRule]:
        rule_class_mapping = {
            "regex": RegexRule,
            "llm_check": LLMCheckRule,
            "llm_classifier": LLMClassifierRule,
            "self_consistency": SelfConsistencyRule,
            "rag_faithfulness": RAGFaithfulnessRule,
        }
        loaded_rules = []

        try:
            with open(filepath, "r") as f:
                yaml_config = yaml.safe_load(f)
        except FileNotFoundError:
            print(
                f"Warning: Configuration file not found at {filepath}. No rails loaded."
            )
            return []

        for rule_conf in yaml_config.get(rail_type, []):
            rule_type = rule_conf.get("type")
            RuleClass = rule_class_mapping.get(rule_type)
            if RuleClass:
                try:
                    # --- [修复 2] 将正确的 LLM 客户端传递给正确的规则 ---
                    if rule_type == "llm_classifier":
                        # 意图分类器使用 Qwen (@ 8000)
                        instance = RuleClass(
                            **rule_conf, shared_llm=self.classifier_llm
                        )
                    elif rule_type == "llm_check":
                        # Llama Guard 规则使用 Llama Guard (@ 8002)
                        instance = RuleClass(
                            **rule_conf, shared_llm=self.semantic_guard_llm
                        )
                    elif rule_type == "self_consistency":
                        # 一致性检查使用 Qwen (@ 8000)
                        instance = RuleClass(
                            **rule_conf, shared_llm=self.verification_llm
                        )
                    else:
                        # Regex 规则等不需要 LLM
                        instance = RuleClass(**rule_conf)
                    # --- 结束修改 ---

                    loaded_rules.append(instance)
                    print(
                        f"Successfully loaded rule: '{rule_conf.get('name')}' for {rail_type}"
                    )
                except Exception as e:
                    print(f"Error loading rule '{rule_conf.get('name')}': {e}")
        return loaded_rules

    async def generate_async(self, messages, context=None) -> str:
        context = context or {}
        t0 = time.time()
        out = await self.runtime.run_once(messages, context)
        if self.config.tracing_enabled:
            print(f"[trace] took={time.time() - t0:.3f}s")
        return out

    def generate(
        self, messages: List[Dict[str, str]], context: Optional[Dict[str, Any]] = None
    ) -> str:
        return asyncio.run(self.generate_async(messages, context))

    async def stream_async(
        self, messages: List[Dict[str, str]], context: Optional[Dict[str, Any]] = None
    ) -> AsyncIterator[str]:
        context = context or {}
        t0 = time.time()
        try:
            async for ch in self.runtime.run_stream(messages, context):
                yield ch
        finally:
            if self.config.tracing_enabled:
                print(f"\n[trace] took={time.time() - t0:.3f}s")
