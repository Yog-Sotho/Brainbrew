# User guide

## Generate page

**Model endpoint** (sidebar). OpenAI, a local vLLM server (`localhost:8000`),
Ollama (`localhost:11434`) or a custom URL. Local servers need no key. An
operator can preset the endpoint with `OPENAI_BASE_URL` and lock it with
`BRAINBREW_ALLOW_CUSTOM_ENDPOINTS=0`. With login on, it is locked unless the
operator sets that variable to `1`.

**Teacher model(s).** The model name as the endpoint knows it. Separate several
with commas for an ensemble: chunks are spread across them.

**Quality mode.**

| Mode | What it does |
|---|---|
| Fast | Pattern filters only (refusals, non-answers, prompt leaks). |
| Balanced | Plus an LLM judge: faithfulness, helpfulness and correctness, all at least the minimum score (default 4/5). |
| Research | Plus harder, evolved questions, kept only if still answerable from the source. |

**Output format.** Alpaca, ShareGPT, ChatML or OpenAI fine-tuning JSONL.

**Target dataset size.** Generation stops when this many pairs pass every check,
or when the documents stop producing new questions. After upload the app shows
how many pairs the documents can likely support (about 6 per chunk) and warns
when the target is higher.

**Data cleaning** (sidebar).

- *Semantic chunking*: split on paragraphs and sentences instead of fixed windows.
- *Deduplicate*: remove near-duplicates (MinHash). With an *embedding model* set
  under Generation settings, paraphrases are removed too.
- *Clean & sanitize*: redact personal data (emails, phones, Luhn-valid card
  numbers, IBANs, IP addresses, links per the *Links* policy, and names with
  Presidio when installed), strip HTML and drop low-quality pairs.
- *Remove benchmark overlap*: drop pairs that share a 13-word passage with the
  selected public test sets.

**Generation settings** (sidebar): temperature, max answer length, parallel
requests, request timeout, judge model, minimum judge score, embedding model and
paraphrase cut-off.

**Auto-train LoRA adapter.** Fine-tunes the chosen base model on the dataset
(needs the `train` extra and, for real models, an NVIDIA GPU). Only one training
runs at a time per machine.

**Publish to Hugging Face.** Uploads the dataset with a generated dataset card.
Repos are private unless *Make it public* is ticked. With training on, the
adapter can be published as `<dataset repo>-lora`.

## While a run is going

The run happens in the background. The page shows progress and a **Cancel run**
button; you can reload the page or close the tab. The address bar holds
`?run=<id>`, which brings you back to the run.

## Run history page

Lists your runs (with login on, only yours) with status, size, grade, mode,
teacher and cost. Open one to see its results, downloads (dataset, adapter,
rejected pairs, log) and details: counts, token usage, cost, seed and timings.

A run shown as *running elsewhere* belongs to another Brainbrew process, such
as the CLI. *Interrupted* means the process running it stopped.

## Run folders

```
runs/<run-id>/
  source.txt           the document text
  raw.jsonl            pairs straight from generation
  records.jsonl        pairs after decontamination and sanitizing
  rejected.jsonl       pairs the filters or the judge rejected, with the reason
  dataset.<fmt>.jsonl  the export
  run.log              this run's log (JSON lines)
  manifest.json        settings (no secrets), models, seed, usage, cost,
                       timings, counts, quality report
  adapter/, adapter.zip   the LoRA adapter, when trained
```
