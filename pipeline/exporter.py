import json

def export_alpaca(input_path: str, output_path: str) -> None:
    with open(input_path, encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            # FIX: skip malformed JSON lines instead of crashing
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            # FIX: support both 'output' (post-RenameColumns) and 'generation' (raw distilabel)
            output = obj.get("output") or obj.get("generation") or ""
            # FIX: skip records with no instruction or empty output
            if not obj.get("instruction") or not output:
                continue
            rec = {"instruction": obj["instruction"], "input": "", "output": output}
            fout.write(json.dumps(rec) + "\n")
