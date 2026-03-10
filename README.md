<div align="center">
  <img src="images/brainbrew_logo.png" alt="Brainbrew Logo" width="420">
  <h1>🧠 Brainbrew</h1>
  <p><strong>the ridiculously easy, stupidly powerful no-code machine that turns your boring PDFs and TXT files into god-tier synthetic LLM training data</strong></p>
  
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
  [![Python 3.12+](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org)
  [![Docker Ready](https://img.shields.io/badge/Docker-Ready-blue.svg)](Dockerfile)
  [![distilabel Powered](https://img.shields.io/badge/distilabel-Powered-purple.svg)](https://distilabel.argilla.io)
</div>

**Brainbrew** — Think of it like a mad scientist + coffee machine combo: you dump in documents, hit one button, and **BOOM** — fresh, high-quality instruction datasets appear like magic. No coding. No spreadsheets. No crying over JSON formatting at 3 a.m.

We took the original prototype, **slayed every bug**, switched to production-grade distilabel magic, added semantic chunking, progress bars, Docker, and a bunch of other goodies… then wrapped it in a shiny Streamlit UI that even your grandma could use.

**Current version: v1.0.0 Production-Ready** 🔥

---

## 🚀 Why Brainbrew Slaps

- **Zero coding** — literally just upload files and click “Generate Dataset”
- **Distilabel-powered evolution** — real Evol-Instruct (not that broken loop from the old code)
- **Semantic chunking** — your documents actually get understood, not chopped like a bad haircut
- **vLLM or OpenAI** — choose speed (GPU) or zero-setup (API)
- **Auto LoRA training** — optional one-click fine-tune with Unsloth
- **Hugging Face publish** — one checkbox and your dataset is live on the Hub
- **Error handling & progress bars** — because crashes are for amateurs
- **Docker ready** — run it anywhere without summoning the dependency demon

In short: it’s what the original repo *wanted* to be when it grew up.

---

## ✨ Features That Make You Look Cool

- **Quality Modes**: Fast (cheap & quick), Balanced (sweet spot), Research (maximum brain juice)
- **Smart Filtering**: Automatic refusal cleaning + quality scoring
- **Export**: Clean Alpaca-format `dataset.jsonl` ready for training
- **Live Stats**: See token counts and dataset health (if you’re into that nerd stuff)
- **Temp Files + Cleanup**: No more leftover `input.txt` disasters
- **Full Logging**: So you can flex on your friends with pretty terminal output
- **Pydantic Config**: Type-safe everything (no more surprise crashes)

---

## 📦 Quick Start (Takes 2 Minutes)

### 1. Clone & Setup
```bash
git clone https://github.com/YOURNAME/Brainbrew.git   # or your fork
cd Brainbrew
