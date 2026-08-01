import time
import json
from pathlib import Path
from pipeline.sanitizer import sanitize_dataset, SanitizerConfig

def run_benchmark():
    input_path = Path("tests/benchmark_input.jsonl")
    output_path = Path("tests/benchmark_output.jsonl")
    cfg = SanitizerConfig(remove_pii=True, deduplicate=True)

    # 1. PII-Heavy Dataset (100% records have PII)
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

    # Warm up
    sanitize_dataset(input_path, output_path, cfg)

    # Time execution
    start = time.perf_counter()
    for _ in range(10):
        sanitize_dataset(input_path, output_path, cfg)
    end = time.perf_counter()
    duration_pii = (end - start) / 10.0
    print(f"Average sanitization time (100% PII text): {duration_pii * 1000:.2f} ms")

    # 2. Clean/Normal Dataset (0% records have PII, which is normal for LLM datasets)
    records_clean = []
    for i in range(1000):
        record = {
            "instruction": f"Explain the concept of quantum computing in simple terms for beginners.",
            "input": f"Use analogies if possible.",
            "output": f"Quantum computing is a rapidly-emerging technology that harnesses the laws of quantum mechanics to solve problems too complex for classical computers. " * 5
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
    print(f"Average sanitization time (0% PII / clean text): {duration_clean * 1000:.2f} ms")

    # Clean up
    if input_path.exists():
        input_path.unlink()
    if output_path.exists():
        output_path.unlink()

if __name__ == "__main__":
    run_benchmark()
