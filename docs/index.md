# Brainbrew

Brainbrew turns your documents (PDF or text) into **grounded synthetic training
data** for language models: question/answer pairs written from passages of your
documents, answered from those passages only, checked by filters and an LLM
judge, deduplicated, and exported in the format your training framework needs.

It works with any **OpenAI-compatible model endpoint**: OpenAI, a local
`vllm serve`, Ollama, llama.cpp or a hosted provider. You can use it from a
browser (Streamlit), from the command line, or in Docker.

## What a run does

1. Reads your documents and splits them into passages (about 1,600 characters).
2. Asks the teacher model for several kinds of questions per passage
   (factual, conceptual, procedural, comparative, multi-hop).
3. Answers each question from its passage only.
4. Drops refusals, non-answers and questions that only make sense next to the
   source; in Balanced and Research mode an LLM judge scores every pair.
5. Removes near-duplicates (and optionally paraphrases), optionally overlap with
   public benchmarks and personal data.
6. Repeats until the target size is reached or the documents run out of new
   questions, then exports, and optionally trains a LoRA adapter and publishes
   to Hugging Face.

Every run keeps its files, log and a manifest in `runs/<run-id>/`.

## Where to go next

- [Quick start](quick-start.md): install and make a first dataset.
- [User guide](user-guide.md): every option in the app.
- [Command line](cli.md): headless runs and scripting.
- [Docker](docker.md): images and the vLLM compose setup.
- [Configuration reference](configuration.md): every setting.
- [Security model](security.md): what is protected and how.
