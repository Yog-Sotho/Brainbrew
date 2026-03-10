import random

TEMPLATES = [
"Rewrite this prompt to require deeper reasoning:",
"Transform this into a multi-step task:",
"Increase the complexity of the following question:",
]

def evolve(prompts, multiplier):

    evolved = []

    for p in prompts:

        evolved.append(p)

        for _ in range(multiplier-1):

            t = random.choice(TEMPLATES)

            evolved.append(f"{t}\n\n{p}")

    return evolved
