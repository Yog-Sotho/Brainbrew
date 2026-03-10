from typing import List
from .base import BaseEngine

class EnsembleTeacher(BaseEngine):
    def __init__(self, engines: List[BaseEngine]):
        self.engines = engines

    def generate(self, prompt: str) -> str:
        outputs = [e.generate(prompt) for e in self.engines if hasattr(e, "generate")]
        return max(outputs, key=len) if outputs else ""

    def generate_batch(self, prompts: List[str]) -> List[str]:
        return [self.generate(p) for p in prompts]
