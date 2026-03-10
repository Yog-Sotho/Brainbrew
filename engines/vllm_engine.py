from vllm import LLM, SamplingParams

class VLLMEngine:

    def __init__(self, model):

        self.llm = LLM(model=model)

        self.params = SamplingParams(
            temperature=0.7,
            max_tokens=1024
        )

    def generate_batch(self, prompts):

        outputs = self.llm.generate(prompts, self.params)

        results = []

        for o in outputs:
            results.append(o.outputs[0].text.strip())

        return results
