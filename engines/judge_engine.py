from .base import BaseEngine

JUDGE_PROMPT = """Score the response quality from 1 to 10.
Instruction: {instruction}
Response: {response}
Return ONLY the number."""

class JudgeEngine:
    def __init__(self, engine: BaseEngine):
        self.engine = engine

    def score(self, instruction: str, response: str) -> float:
        prompt = JUDGE_PROMPT.format(instruction=instruction, response=response)
        try:
            return float(self.engine.generate(prompt).strip())
        except:
            return 0.0
