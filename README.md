<div align="center">
  <img src="brainbrew_logo.png" alt="Brainbrew Logo" width="420" style="margin-bottom: 20px;">
  <h1>🧠 Brainbrew</h1>
  <p><strong>The ridiculously easy, stupidly powerful no-code machine that turns your boring PDFs and TXT files into god-tier synthetic LLM training data</strong></p>

  <p>
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">
    <img src="https://img.shields.io/badge/Python-3.12+-blue.svg" alt="Python 3.12+">
    <img src="https://img.shields.io/badge/Docker-Ready-blue.svg" alt="Docker Ready">
    <img src="https://img.shields.io/badge/distilabel-Powered-purple.svg" alt="distilabel Powered">
  </p>
</div>

<hr>

<p><strong>Brainbrew</strong> — Think of it like a mad scientist + coffee machine combo: you dump in documents, hit one button, and <strong>BOOM</strong> — fresh, high-quality instruction datasets appear like magic. No coding. No spreadsheets. No crying over JSON formatting at 3 a.m.</p>

<p>We took the original prototype, <strong>slayed every bug</strong>, switched to production-grade distilabel magic, added semantic chunking, progress bars, Docker, and a bunch of other goodies… then wrapped it in a shiny Streamlit UI that even your grandma could use.</p>

<p><strong>Current version: v1.0.0 Production-Ready</strong> 🔥</p>

<hr>

<h2>🚀 Why Brainbrew Slaps</h2>
<ul>
  <li><strong>Zero coding</strong> — literally just upload files and click “Generate Dataset”</li>
  <li><strong>Distilabel-powered evolution</strong> — Evol-Instruct</li>
  <li><strong>Semantic chunking</strong> — your documents  get understood, not chopped like a bad haircut</li>
  <li><strong>vLLM or OpenAI</strong> — choose speed (GPU) or zero-setup (API)</li>
  <li><strong>Auto LoRA training</strong> — optional one-click fine-tune with Unsloth</li>
  <li><strong>Hugging Face publish</strong> — one checkbox and your dataset is live on the Hub</li>
  <li><strong>Error handling & progress bars</strong> — because crashes are for amateurs</li>
  <li><strong>Docker ready</strong> — run it anywhere without summoning the dependency demon</li>
</ul>

<p>In short: it’s what every AI guy <em>wanted</em> and never found anywhere.</p>

<hr>

<h2>✨ Features That Make You Look Cool</h2>
<ul>
  <li><strong>Quality Modes</strong>: Fast (cheap & quick), Balanced (sweet spot), Research (maximum brain juice)</li>
  <li><strong>Smart Filtering</strong>: Automatic refusal cleaning + quality scoring</li>
  <li><strong>Export</strong>: Clean Alpaca-format <code>dataset.jsonl</code> ready for training</li>
  <li><strong>Live Stats</strong>: See token counts and dataset health (if you’re into that nerd stuff)</li>
  <li><strong>Temp Files + Cleanup</strong>: No more leftover <code>input.txt</code> disasters</li>
  <li><strong>Full Logging</strong>: So you can flex on your friends with pretty terminal output</li>
  <li><strong>Pydantic Config</strong>: Type-safe everything (no more surprise crashes)</li>
</ul>

<hr>

<h2>📦 Quick Start (Takes 2 Minutes)</h2>

<h3>1. Clone & Setup</h3>
<pre><code>git clone https://github.com/YOURNAME/Brainbrew.git   # or your fork
cd Brainbrew</code></pre>

<h3>2. Install (Python 3.12+)</h3>
<pre><code>pip install -r requirements.txt</code></pre>

<h3>3. Add Your Keys</h3>
<pre><code>cp .env.example .env</code></pre>
<p>Edit <code>.env</code>:</p>
<pre><code>OPENAI_API_KEY=sk-...
HF_TOKEN=hf_...
HF_USERNAME=yourusername</code></pre>

<h3>4. Run It</h3>
<pre><code>streamlit run app.py</code></pre>

<p><strong>Boom.</strong> Browser opens. You’re now a dataset wizard.</p>

<hr>

<h2>🐳 Docker (For the Cool Kids)</h2>
<pre><code>docker build -t brainbrew .
docker run --gpus all -p 8501:8501 --env-file .env brainbrew</code></pre>

<p>Open <code>http://localhost:8501</code> and flex.</p>

<hr>

<h2>🎮 How to Use (So Easy It’s Embarrassing)</h2>
<ol>
  <li>Upload your PDFs or TXT files (multiple OK!)</li>
  <li>Pick your teacher model (GPT-4o for no GPU, or Llama-3.1-8B for vLLM speed)</li>
  <li>Choose quality mode</li>
  <li>Slide to desired dataset size</li>
  <li>Optional: tick “Auto-train LoRA” and/or “Publish to HF”</li>
  <li>Smash the big <strong>🚀 Generate Dataset</strong> button</li>
  <li>Watch the magic + download your <code>alpaca_dataset.jsonl</code></li>
</ol>
<p>Done. Go train a model that actually knows your niche.</p>

<hr>

<h2>⚙️ Advanced Settings (Sidebar)</h2>
<ul>
  <li><strong>Use vLLM</strong> → Lightning fast (needs ≥24 GB VRAM)</li>
  <li><strong>OpenAI API Key</strong> → fallback for laptop warriors</li>
  <li><strong>HF Token</strong> → for publishing</li>
  <li>Everything else (temperature, LoRA rank, etc.) is smart-defaulted but tweakable in code if you’re fancy</li>
</ul>

<hr>

<h2>🛠️ Tech Stack (The Secret Sauce)</h2>
<ul>
  <li><strong>Streamlit</strong> – beautiful UI in 50 lines</li>
  <li><strong>distilabel</strong> – the real MVP (evolution + generation + filtering)</li>
  <li><strong>vLLM</strong> – GPU wizardry</li>
  <li><strong>Unsloth</strong> – fastest LoRA training on the planet</li>
  <li><strong>LangChain text splitters</strong> – semantic chunking</li>
  <li><strong>Pydantic + Structlog</strong> – no more “it worked on my machine” excuses</li>
</ul>

<hr>

<h2>📋 Hardware Requirements (Be Honest)</h2>
<table>
  <thead>
    <tr>
      <th>Mode</th>
      <th>GPU Needed?</th>
      <th>Speed</th>
      <th>Cost</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>OpenAI</td>
      <td>None</td>
      <td>Medium</td>
      <td>$$ (API)</td>
    </tr>
    <tr>
      <td>vLLM (8B)</td>
      <td>24 GB+ VRAM</td>
      <td>Blazing</td>
      <td>Free</td>
    </tr>
    <tr>
      <td>LoRA training</td>
      <td>Same as above</td>
      <td>Fast</td>
      <td>Free</td>
    </tr>
  </tbody>
</table>

<p><em>Pro tip: Start with OpenAI mode. Once it works, flex with vLLM on RunPod/Modal.</em></p>

<hr>

<h2>🐛 Troubleshooting (The Funny Edition)</h2>
<ul>
  <li><strong>“CUDA out of memory”</strong> → Turn off vLLM or use smaller model</li>
  <li><strong>OpenAI rate limit</strong> → Chill, use smaller batch or wait</li>
  <li><strong>Nothing happens</strong> → Check console + make sure you uploaded files</li>
  <li><strong>HF publish fails</strong> → Token wrong? Repo name taken? Classic.</li>
</ul>
<p>Still stuck? Open an issue. I (or the original author) will roast the bug with you.</p>

<hr>

<h2>🤝 Contributing</h2>
<p>Love it? Want to make it even cooler?</p>
<ol>
  <li>Fork it</li>
  <li>Make changes (we love clean PRs)</li>
  <li>Add tests? (we don’t have any yet — first contributor gets legendary status)</li>
  <li>Submit PR</li>
</ol>
<p>Ideas welcome: RAG retrieval, multi-modal support, cost estimator, web UI for cloud, etc.</p>

<hr>

<h2>📜 License</h2>
<p>MIT — do whatever you want. Just don’t blame us if your model becomes too powerful and takes over the world.</p>

<div align="center" style="margin-top: 50px;">
  <img src="brainbrew_logo.png" alt="Brainbrew Logo" width="280">
  <h2>Now go brew some brains.</h2>
  <p><strong>Made with chaos, coffee, and zero patience for bad datasets.</strong></p>
  <p><em>Star the repo if it saved you 20 hours this week. You know you want to. ⭐</em></p>
  Yog-Sotho
</div>
