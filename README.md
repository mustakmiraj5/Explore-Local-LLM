# Local LLM with Python

A hands-on project for learning how local large language models work, using [Ollama](https://ollama.com) and **Qwen 3.5 4B** on macOS.

It has two parts:

- **Python experiments:** small scripts that talk to a locally running model.
- **A study syllabus:** step-by-step lessons on LLM fundamentals, in [`docs/`](docs/README.md).

## Requirements

- Python 3.14
- Ollama, with the `qwen3.5:4b` model pulled

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

ollama pull qwen3.5:4b
ollama serve        # skip if the Ollama app is already running

export LLM_BASE_URL=http://localhost:11434/v1
export LLM_API_KEY=ollama   # Ollama ignores the key, but the client requires one
```

Add the two `export` lines to `~/.zshrc` to keep them set in new terminals.

## Scripts

| Script | What it does |
|---|---|
| `main.py` | An interactive chat agent that can call tools to get the current date and time, the weather for any city (via [Open-Meteo](https://open-meteo.com), no key needed), system info (OS, CPU, RAM) and the current directory. Type `exit` to quit. |
| `web_reader.py` | Fetches a webpage, extracts its main content, and asks the model to do what you tell it with it (summarize by default, or e.g. "list the 5 most important points"). |
| `test-token.py` | Encodes and decodes a sentence with `tiktoken` to show how text becomes token IDs. |

```bash
python main.py
python web_reader.py
python test-token.py
```

`main.py` connects to Ollama's OpenAI-compatible API at `http://localhost:11434/v1`, so it uses the standard `openai` Python library without needing an OpenAI key.

## Syllabus

[LLM Fundamentals — Phase 1](docs/README.md) covers seven topics:

1. LLM Architecture
2. Tokenization
3. Token Generation
4. Inference
5. Quantization
6. Context Window
7. Ollama Internals

Each lesson explains the idea, shows it on the local Qwen model, and ends with questions and a practical assignment. Progress is tracked in [`docs/README.md`](docs/README.md).
