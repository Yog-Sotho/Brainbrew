# Hardware

| Setup | GPU | Notes |
|---|---|---|
| Hosted API (OpenAI or another provider) | none | Any OS. Cost per run is shown before and recorded after. |
| Local `vllm serve` with a 4–8B model | 16–24 GB VRAM, Linux, NVIDIA driver R580+ | Free per run. `compose.yaml` sets it up. |
| Ollama / llama.cpp | optional | Works on CPU but slowly; use 1–2 parallel requests and a long timeout. |
| LoRA training | 8 GB+ VRAM for 4-bit QLoRA of a 4B model | One training at a time per machine. Tiny models train on CPU (tests do). |

The web app itself needs little: about 1 GB of RAM plus room for the documents
and run folders.
