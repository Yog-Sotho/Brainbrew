from openai import OpenAI
from tenacity import retry, wait_exponential, stop_after_attempt
from typing import List
from .base import BaseEngine

class TeacherEngine(BaseEngine):
    def __init__(self, model: str, api_key: str | None = None):
        self.model = model
        self.client = OpenAI(api_key=api_key)

    @retry(wait=wait_exponential(min=1, max=20), stop=stop_after_attempt(5))
    def generate(self, prompt: str) -> str:
        resp = self.client.chat.completions.create(model=self.model, messages=[{"role": "user", "content": prompt}], temperature=0.7, max_tokens=2048)
        return resp.choices[0].message.content.strip()

    def generate_batch(self, prompts: List[str]) -> List[str]:
        return [self.generate(p) for p in prompts]
