import asyncio
import os
from openai import OpenAI, base_url
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid
from typing import List, Dict, AsyncIterator

from .base import BaseLLM


class VLLMOpenAIClient(BaseLLM):
    def __init__(self, base_url: str, model_name: str):
        api_key = "EMPTY"  # vLLM 默认值
        if "localhost" not in base_url and "127.0.0.1" not in base_url:
            # 如果不是本地 vLLM，就从环境变量中读取 OpenAI API 密钥
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                print("⚠️ [VLLMOpenAIClient] 警告：未在环境变量中找到 OPENAI_API_KEY。")
        
        self.client = OpenAI(base_url=base_url, api_key=api_key)        
        self.model = model_name

    async def acomplete(self, messages: List[Dict[str, str]], **kwargs) -> str:
        loop = asyncio.get_event_loop()

        params = {
            "model": self.model,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", 512),
            "temperature": kwargs.get("temperature", 0.1),
            "stream": False,
        }
        if "seed" in kwargs:
            params["seed"] = kwargs["seed"]
        if "top_p" in kwargs:
            params["top_p"] = kwargs["top_p"]
     

        try:
            response = await loop.run_in_executor(
                None,
                lambda: self.client.chat.completions.create(**params),
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            print(
                f"ERROR: Could not connect to vLLM server at {self.client.base_url}. Details: {e}"
            )
            return "Error: The generation model is currently unavailable."

    async def astream(
        self, messages: List[Dict[str, str]], **kwargs
    ) -> AsyncIterator[str]:
        full_response = await self.acomplete(messages, **kwargs)
        yield full_response


class VLLMInProcessLLM(BaseLLM):
    def __init__(self, model_name: str):
        engine_args = AsyncEngineArgs(
            model=model_name,
            trust_remote_code=True,
            enforce_eager=True,
            tensor_parallel_size=1,
            distributed_executor_backend="mp",
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self.model_name = model_name

    async def acomplete(self, messages: List[Dict[str, str]], **kwargs) -> str:
        # ... (full content of acomplete)
        sampling_params = SamplingParams(
            max_tokens=kwargs.get("max_tokens", 512),
            temperature=kwargs.get("temperature", 0.7),
            top_p=kwargs.get("top_p", 0.95),
            # --- ↓↓↓ 确保 InProcess 模式也能接受 seed (如果使用) ↓↓↓ ---
            seed=kwargs.get("seed", None),
        )
        tokenizer = await self.engine.get_tokenizer()
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        request_id = f"cmpl-{random_uuid()}"
        results_generator = self.engine.generate(prompt, sampling_params, request_id)
        final_output = ""
        async for request_output in results_generator:
            final_output = request_output.outputs[0].text
        if prompt in final_output:
            return final_output[len(prompt) :]
        return final_output

    async def astream(
        self, messages: List[Dict[str, str]], **kwargs
    ) -> AsyncIterator[str]:
        # ... (full content of astream)
        sampling_params = SamplingParams(
            max_tokens=kwargs.get("max_tokens", 512),
            temperature=kwargs.get("temperature", 0.7),
            top_p=kwargs.get("top_p", 0.95),
            # --- ↓↓↓ 确保 InProcess 模式也能接受 seed (如果使用) ↓↓↓ ---
            seed=kwargs.get("seed", None),
        )
        tokenizer = await self.engine.get_tokenizer()
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        request_id = f"cmpl-{random_uuid()}"
        results_generator = self.engine.generate(prompt, sampling_params, request_id)
        last_text = ""
        async for request_output in results_generator:
            current_text = request_output.outputs[0].text
            delta = current_text[len(last_text) :]
            last_text = current_text
            if delta:
                yield delta
