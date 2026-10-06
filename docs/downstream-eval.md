# Downstream evaluation

The [quality benchmark](testing.md) checks the generated *data*: is it faithful,
non-repetitive and free of refusals? This page checks what the data is for:
**does a model trained on it get better?**

The experiment trains a small "student" model on a Brainbrew dataset and gives it a
closed-book exam about the source documents. The base model takes the same exam.
A control adapter is trained on a cheaper dataset (Fast mode, no LLM judge) to show
what Brainbrew's quality steps add. A general benchmark then checks that the
student did not get worse at everything else.

`bench/downstream.py` does the parts no other tool covers. It writes the exam,
wraps LoRA training, and grades every model's answers.

## Hardware

The commands below fit **one 8 GB NVIDIA GPU**, such as an RTX 3060 Ti, plus 32 GB
of RAM. Check the card's memory with:

```bash
nvidia-smi --query-gpu=name,memory.total --format=csv
```

| GPU memory | Teacher (writes the data) | Student (gets trained) |
|---|---|---|
| 8 GB | `Qwen/Qwen2.5-7B-Instruct-AWQ` | `Qwen/Qwen2.5-1.5B-Instruct` |
| 12 GB or more | the same, or a larger AWQ model | `Qwen/Qwen3-4B-Instruct-2507` |

One GPU can run either vLLM or training, not both at once: vLLM reserves most of
the card's memory. The steps below start and stop the server in the right order.
For the same reason, do not tick *Auto-train LoRA adapter* in a run whose teacher
runs on the same GPU.

The exam writer and the judge should be a stronger, independent model. The
commands use `gpt-4o-mini` through the OpenAI API, which costs little. A local
judge works too, but then it shares blind spots with the teacher.

## Setup

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh        # uv brings its own Python 3.12
git clone https://github.com/Yog-Sotho/Brainbrew && cd Brainbrew
uv sync --locked --extra vllm --extra train            # name both: syncing one extra removes the other
read -rs OPENAI_API_KEY && export OPENAI_API_KEY       # for the exam writer and the judge only
```

How API keys are used:

- `OPENAI_API_KEY` is only ever sent to OpenAI.
- A local vLLM server needs no key.
- If another endpoint needs one, put it in `ENDPOINT_API_KEY`.

## 1. Pick the documents

Use 20–50 pages the student cannot already know: your own documents, or material
published after the model was trained. Well-known text measures memory from
pre-training rather than what the training taught. The commands below call
them `docs/*.pdf`.

## 2. Generate the training sets (teacher on the GPU)

```bash
uv run vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ --host 127.0.0.1 --port 8000 \
  --max-model-len 8192 --gpu-memory-utilization 0.90
```

In a second terminal:

```bash
export T=http://127.0.0.1:8000/v1 M=Qwen/Qwen2.5-7B-Instruct-AWQ
uv run brainbrew run docs/*.pdf --base-url $T --model $M --size 500 --mode balanced   # Brainbrew
uv run brainbrew run docs/*.pdf --base-url $T --model $M --size 500 --mode fast       # control
```

Each run prints its folder, `runs/<run-id>/`. The canonical records are in
`records.jsonl` there. Below, `$RUN` is the Balanced run and `$CTRL` the Fast run.
Stop vLLM (Ctrl-C) when both are done.

## 3. Write the exam

```bash
uv run python bench/downstream.py testset docs/*.pdf --train runs/$RUN/records.jsonl \
  --model gpt-4o-mini --size 150 --out testset.jsonl
```

How the exam is built:

- Questions are closed-book and self-contained, each with a one- or two-sentence
  reference answer taken from the source passage.
- They are written in a different style from the training questions.
- A question that is a near-copy of a training question is dropped (MinHash
  similarity 0.5 or more).
- The exam can still ask about the same *facts* as the training data. That is
  the point: it measures whether the student learned them, not whether it
  memorised the wording.
- The command prints how many proposed questions each filter removed.

## 4. Train the adapters (GPU free)

```bash
uv run python bench/downstream.py train runs/$RUN/records.jsonl \
  --base-model Qwen/Qwen2.5-1.5B-Instruct --out adapters/brainbrew --epochs 3
uv run python bench/downstream.py train runs/$CTRL/records.jsonl \
  --base-model Qwen/Qwen2.5-1.5B-Instruct --out adapters/control --epochs 3
```

With CUDA, training is QLoRA (4-bit). Three epochs suit learning new facts; the
app's default of one is tuned for style.

## 5. Take the exam (student on the GPU)

vLLM serves the base model and both adapters from one process:

```bash
uv run vllm serve Qwen/Qwen2.5-1.5B-Instruct --host 127.0.0.1 --port 8000 \
  --enable-lora --lora-modules brainbrew=adapters/brainbrew control=adapters/control \
  --max-lora-rank 16 --max-model-len 4096 --gpu-memory-utilization 0.85
```

```bash
uv run python bench/downstream.py evaluate testset.jsonl \
  --student-base-url http://127.0.0.1:8000/v1 \
  --models Qwen/Qwen2.5-1.5B-Instruct brainbrew control \
  --judge-model gpt-4o-mini --out downstream-report.json
```

How the evaluation works:

- Each model answers every question at temperature 0, without the documents.
- The judge grades each answer from 1 to 5 against the reference and the source
  passage. 4 or 5 counts as correct.
- The first model listed is the baseline.
- Every other model gets its mean gain over the baseline, a paired bootstrap 95%
  confidence interval, and its wins, ties and losses per question.
- `downstream-report.json` holds every answer and grade, for reading the failures.

```text
| Model                        | Mean grade | Correct | Gain vs baseline [95% CI] | Wins / ties / losses |
| `Qwen/Qwen2.5-1.5B-Instruct` | ...        | ...     | baseline                  | –                    |
| `brainbrew`                  | ...        | ...     | +x.xx [+a, +b] *          | ...                  |
| `control`                    | ...        | ...     | ...                       | ...                  |
```

To see whether Brainbrew beats the control directly, run `evaluate` again with
`--models control brainbrew`.

## 6. Check nothing else got worse

```bash
uv tool install lm-eval --with peft
lm_eval --model hf --model_args pretrained=Qwen/Qwen2.5-1.5B-Instruct \
  --tasks arc_challenge,hellaswag --batch_size auto
lm_eval --model hf --model_args pretrained=Qwen/Qwen2.5-1.5B-Instruct,peft=adapters/brainbrew \
  --tasks arc_challenge,hellaswag --batch_size auto
```

Stop vLLM first, since this loads the model itself.

## Reading the result

These pass marks are a proposal for this project, not an external standard:

| Question | Pass when |
|---|---|
| Did the training teach the documents? | `brainbrew` beats the base model by at least **+1.0** mean grade, and the 95% interval excludes zero. |
| Do Brainbrew's quality steps matter? | `brainbrew` beats `control` with the interval above zero. A tie means the judge and filters cost tokens without helping. |
| Is the student still a general model? | ARC-Challenge and HellaSwag drop by no more than 1–2 points. |

Two further checks make the result firmer:

- Write a second exam with a different `--seed` and check that the conclusion holds.
- Read 10–20 low-graded answers in `downstream-report.json` before trusting the
  numbers. A judge can be wrong in systematic ways.
