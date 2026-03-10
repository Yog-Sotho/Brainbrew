from vllm import LLM, SamplingParams
from typing import List
from .base import BaseEngine

class VLLMEngine(BaseEngine):
    def __init__(self, model: str):
        self.llm = LLM(model=model, gpu_memory_utilization=0.9)
        self.params = SamplingParams(temperature=0.7, max_tokens=2048)

    def generate(self, prompt: str) -> str:
        return self.llm.generate([prompt], self.params)[0].outputs[0].text.strip()

    def generate_batch(self, prompts: List[str]) -> List[str]:
        outputs = self.llm.generate(prompts, self.params)
        return [o.outputs[0].text.strip() for o in outputs]
