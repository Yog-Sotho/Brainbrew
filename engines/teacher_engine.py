from openai import OpenAI
from tenacity import retry, wait_exponential, stop_after_attempt

class TeacherEngine:

    def __init__(self, model, api_key=None, base_url=None):

        self.model = model
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )

    @retry(wait=wait_exponential(min=1, max=20), stop=stop_after_attempt(5))
    def generate(self, prompt):

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role":"user","content":prompt}],
            temperature=0.7,
            max_tokens=1024
        )

        return resp.choices[0].message.content.strip()
