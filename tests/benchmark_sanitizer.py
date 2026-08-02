import time
import json
from pathlib import Path
from pipeline.sanitizer import sanitize_dataset, SanitizerConfig

def run_benchmark():
    input_path = Path("tests/benchmark_input.jsonl")
    output_path = Path("tests/benchmark_output.jsonl")

    # 1. PII-heavy dataset (the original benchmark)
    records_pii = []
    for i in range(1000):
        record = {
            "instruction": f"My email is user_{i}@example.com, and my IP is 192.168.1.{i % 255}.",
            "input": f"Additional context {i} with a phone number +1-555-555-01{i % 100:02d}.",
            "output": f"This is a high quality response number {i} containing a URL https://example.com/item/{i}. " * 5
        }
        records_pii.append(record)

    with open(input_path, "w", encoding="utf-8") as f:
        for r in records_pii:
            f.write(json.dumps(r) + "\n")

    cfg = SanitizerConfig(remove_pii=True, deduplicate=True)

    # Warm up
    sanitize_dataset(input_path, output_path, cfg)

    # Time execution
    start = time.perf_counter()
    for _ in range(10):
        sanitize_dataset(input_path, output_path, cfg)
    end = time.perf_counter()

    duration_pii = (end - start) / 10.0
    print(f"Average sanitization time (PII-heavy): {duration_pii * 1000:.2f} ms")

    # 2. Clean dataset (typical case)
    records_clean = []
    for i in range(1000):
        record = {
            "instruction": f"Explain key concept number {i} clearly and in detail.",
            "input": f"Use the context of chapter {i} from the main handbook.",
            "output": f"This is a high quality response number {i} explaining the key concepts perfectly. " * 5
        }
        records_clean.append(record)

    with open(input_path, "w", encoding="utf-8") as f:
        for r in records_clean:
            f.write(json.dumps(r) + "\n")

    # Warm up
    sanitize_dataset(input_path, output_path, cfg)

    # Time execution
    start = time.perf_counter()
    for _ in range(10):
        sanitize_dataset(input_path, output_path, cfg)
    end = time.perf_counter()

    duration_clean = (end - start) / 10.0
    print(f"Average sanitization time (Clean dataset): {duration_clean * 1000:.2f} ms")

    # Clean up
    if input_path.exists():
        input_path.unlink()
    if output_path.exists():
        output_path.unlink()

if __name__ == "__main__":
    run_benchmark()
