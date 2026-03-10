class EnsembleTeacher:

    def __init__(self, engines):
        self.engines = engines

    def generate(self, prompt):

        outputs = []

        for engine in self.engines:

            try:
                out = engine.generate(prompt)
                outputs.append(out)
            except:
                continue

        if not outputs:
            return ""

        # choose longest answer as proxy for completeness
        outputs.sort(key=len, reverse=True)

        return outputs[0]
