from abc import ABC, abstractmethod
from typing import List

class BaseEngine(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str: ...
    @abstractmethod
    def generate_batch(self, prompts: List[str]) -> List[str]: ...
