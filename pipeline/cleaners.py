import json

REFUSALS = [
"as an ai language model",
"i cannot assist",
"i'm unable to"
]

def is_refusal(text):

    t = text.lower()

    for r in REFUSALS:
        if r in t:
            return True

    return False


def clean_dataset(input_path, output_path):

    seen=set()

    with open(input_path) as fin, open(output_path,"w") as fout:

        for line in fin:

            obj=json.loads(line)

            if is_refusal(obj["output"]):
                continue

            key=obj["instruction"]

            if key in seen:
                continue

            seen.add(key)

            fout.write(json.dumps(obj)+"\n")
