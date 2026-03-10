import json

def export_alpaca(input_path: str, output_path: str) -> None:
    with open(input_path, encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            obj = json.loads(line)
            rec = {"instruction": obj["instruction"], "input": "", "output": obj.get("output", "")}
            fout.write(json.dumps(rec) + "\n")
