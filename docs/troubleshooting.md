# Troubleshooting

**"An API key is required for the OpenAI API".** Enter a key, set
`OPENAI_API_KEY`, or pick a local endpoint (local servers need no key).

**Rate limits or timeouts.** Lower *Parallel requests* or raise *Request
timeout* under Generation settings. CPU servers need 1–2 parallel requests.

**Fewer pairs than the target.** The documents ran out of new questions. The run
details (or `manifest.json` → `generation`) show how many questions were
dropped, answers filtered, pairs rejected by the judge and duplicates removed;
`rejected.jsonl` holds the rejected pairs with the reason. Use longer documents,
a lower target, a stronger teacher, or a lower minimum judge score.

**"No usable question/answer pairs were produced".** Every request failed or
every pair was rejected. Check the model name and endpoint, then the run log
(`runs/<run-id>/run.log`).

**Structured-output errors from a local server.** Brainbrew falls back from JSON
schema to JSON mode to a schema in the prompt automatically. Very small models
may still produce unusable JSON; use a larger one.

**A run shows "interrupted".** The process running it stopped (the server was
restarted, or the CLI was killed). Start a new run; interrupted runs cannot be
resumed.

**"Waiting for the GPU".** Another run is training on this machine. It continues
when that run finishes; you can cancel it meanwhile.

**CUDA out of memory.** Use a smaller model in `vllm serve`, lower
`--gpu-memory-utilization`, or train a smaller base model.

**Hugging Face publishing fails.** Check that the token has write access and the
repo name is `user/name`.
