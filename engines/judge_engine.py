JUDGE_PROMPT = """
Score the quality of the response from 1 to 10.

Instruction:
{instruction}

Response:
{response}

Return only the number.
"""

class JudgeEngine:

    def __init__(self, engine):
        self.engine = engine

    def score(self, instruction, response):

        prompt = JUDGE_PROMPT.format(
            instruction=instruction,
            response=response
        )

        result = self.engine.generate(prompt)

        try:
            return float(result.strip())
        except:
            return 0
